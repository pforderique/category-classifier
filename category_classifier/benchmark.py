"""Prediction benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import torch

from category_classifier.artifacts import resolve_model_pack_path
from category_classifier.predictor import Predictor
from category_classifier.runtime import is_mps_available


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency benchmark output for a single device."""

    device: str
    iterations: int
    warmup: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    throughput_rows_per_sec: float | None
    skipped_reason: str | None = None


def _run_latency_benchmark(
    predictor: Predictor,
    item_name: str,
    price: object,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        predictor.predict(item_name=item_name, price=price)
    if predictor.device == "mps":
        torch.mps.synchronize()

    measurements_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        predictor.predict(item_name=item_name, price=price)
        if predictor.device == "mps":
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        measurements_ms.append(elapsed_ms)

    mean_ms = statistics.fmean(measurements_ms)
    p50_ms = statistics.median(measurements_ms)
    p95_ms = sorted(measurements_ms)[int((len(measurements_ms) - 1) * 0.95)]
    throughput = 1000.0 / mean_ms if mean_ms > 0 else None

    return BenchmarkResult(
        device=predictor.device,
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        throughput_rows_per_sec=throughput,
    )


def benchmark_model_pack(
    model_pack: str,
    artifacts_dir: str | Path,
    devices: list[str],
    item_name: str,
    price: object,
    warmup: int = 20,
    iterations: int = 200,
) -> list[BenchmarkResult]:
    """Benchmark end-to-end predict latency across selected devices."""
    model_pack_path = resolve_model_pack_path(model_pack=model_pack, artifacts_dir=artifacts_dir)
    results: list[BenchmarkResult] = []

    for device in devices:
        normalized = device.strip().lower()
        if normalized == "mps" and not is_mps_available():
            results.append(
                BenchmarkResult(
                    device="mps",
                    iterations=iterations,
                    warmup=warmup,
                    mean_ms=None,
                    p50_ms=None,
                    p95_ms=None,
                    throughput_rows_per_sec=None,
                    skipped_reason="mps is not available on this machine",
                )
            )
            continue
        if normalized not in {"cpu", "mps"}:
            raise ValueError(f"Unsupported benchmark device '{device}'. Use cpu,mps.")

        predictor = Predictor(
            model_pack_path=str(model_pack_path),
            encoder=None,
            device=normalized,
        )
        result = _run_latency_benchmark(
            predictor=predictor,
            item_name=item_name,
            price=price,
            warmup=warmup,
            iterations=iterations,
        )
        results.append(result)

    return results
