#!/usr/bin/env python3
"""Model profiler: architecture sketch, param count, memory, latency, accuracy."""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from statistics import fmean, median

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _pct(measurements: list[float], p: float) -> float:
    s = sorted(measurements)
    return s[min(int((len(s) - 1) * p), len(s) - 1)]


def _run_latency(predictor, samples: list[tuple[str, str | float]], warmup: int, n: int) -> dict:
    fallback = [("Coffee shop purchase", "12.50")]
    pool = samples if samples else fallback

    for i in range(warmup):
        item, price = pool[i % len(pool)]
        predictor.predict(item_name=item, price=price)

    times_ms: list[float] = []
    for i in range(n):
        item, price = pool[i % len(pool)]
        t0 = time.perf_counter()
        predictor.predict(item_name=item, price=price)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg": fmean(times_ms),
        "min": times_ms[0] if len(times_ms) == 1 else min(times_ms),
        "p50": median(times_ms),
        "p99": _pct(times_ms, 0.99),
        "max": max(times_ms),
    }


def _peak_inference_memory_mb(predictor, item: str, price: str) -> float:
    tracemalloc.start()
    predictor.predict(item_name=item, price=price)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a trained model pack.")
    parser.add_argument("--model", required=True, help="Model name or path under --models-dir.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data", help="CSV/TSV used for realistic latency samples and accuracy.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    from category_classifier.model_pack import resolve_model_pack_path
    from category_classifier.predictor import Predictor

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = (_ROOT / models_dir).resolve()

    model_pack_path = resolve_model_pack_path(args.model, models_dir=models_dir)
    predictor = Predictor(model_pack_path=str(model_pack_path), device="cpu")

    pack = predictor._pack
    manifest = pack.manifest
    model_state = pack.model_state
    encoder_name = manifest.get("encoder_model_name", "unknown")
    input_dim = int(model_state["input_dim"])
    num_classes = int(model_state["num_classes"])
    embed_dim = input_dim - 1  # last feature is price

    param_count = sum(p.numel() for p in predictor.model.parameters())
    model_file_kb = (model_pack_path / "model.pt").stat().st_size / 1024

    # Load data for realistic latency samples and accuracy
    samples: list[tuple[str, str | float]] = []
    df = None
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = (_ROOT / data_path).resolve()
        from category_classifier.dataset import load_transactions
        df = load_transactions(data_path)
        samples = [(str(r["item_name"]), r["price"]) for _, r in df.iterrows()]

    latency = _run_latency(predictor, samples, warmup=args.warmup, n=args.iterations)
    peak_mb = _peak_inference_memory_mb(predictor, "Coffee shop purchase", "12.50")

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\nModel:        {args.model}")
    print(f"Architecture: text → {encoder_name} ({embed_dim}-dim) → Linear({input_dim}, {num_classes})")
    print(f"Parameters:   {param_count:,}  (linear head only)")
    print(f"Model file:   {model_file_kb:.1f} KB")
    print(f"Peak mem:     {peak_mb:.2f} MB  (tracemalloc, excludes encoder)")

    print(f"\nLatency  (n={args.iterations}, warmup={args.warmup}, device=cpu):")
    print(f"  avg  {latency['avg']:6.2f} ms")
    print(f"  min  {latency['min']:6.2f} ms")
    print(f"  p50  {latency['p50']:6.2f} ms")
    print(f"  p99  {latency['p99']:6.2f} ms")
    print(f"  max  {latency['max']:6.2f} ms")

    if df is not None:
        correct = sum(
            1
            for _, row in df.iterrows()
            if predictor.predict(item_name=str(row["item_name"]), price=row["price"])
            == str(row["category_display"])
        )
        total = len(df)
        print(f"\nAccuracy:     {correct / total * 100:.1f}%  ({correct}/{total} samples)")
    else:
        print("\nAccuracy:     pass --data <csv> to compute")


if __name__ == "__main__":
    main()
