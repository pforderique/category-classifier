"""Command line interface."""

from __future__ import annotations

# Suppress noisy warnings BEFORE any imports
import io
import logging
import os
import sys
import warnings

# Capture and suppress HF Hub warning during model download
_original_stderr = sys.stderr
_suppressed_output = io.StringIO()

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

from category_classifier.model_pack import resolve_model_pack_path, save_model_pack
from category_classifier.benchmark import benchmark_model_pack
from category_classifier.dataset import load_transactions
from category_classifier.evaluate import evaluate_model
from category_classifier.encoder import SentenceTransformerEncoder
from category_classifier.predictor import Predictor
from category_classifier.runtime import resolve_device
from category_classifier.training import TrainConfig, split_dataset, train_model

DEFAULT_ENCODER_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def _resolve_models_dir(models_dir: str) -> Path:
    """Resolve models directory for CLI commands."""
    candidate = Path(models_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    package_root = Path(__file__).resolve().parent.parent
    return (package_root / candidate).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="category-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model pack from CSV/TSV data.")
    train_parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV/TSV with transaction rows.",
    )
    train_parser.add_argument("--model-name", required=True, help="Name of the output model pack.")
    train_parser.add_argument(
        "--models-dir",
        dest="models_dir",
        default="models",
        help="Directory where model packs are stored.",
    )
    train_parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--learning-rate", type=float, default=1e-2)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--encoder-model",
        default=DEFAULT_ENCODER_MODEL,
        help="Sentence-transformers model name.",
    )

    predict_parser = subparsers.add_parser("predict", help="Run single prediction from a model pack.")
    predict_parser.add_argument(
        "--model-pack",
        required=True,
        help="Model pack path or model name under --models-dir.",
    )
    predict_parser.add_argument("--item-name", required=True)
    predict_parser.add_argument("--price", required=True)
    predict_parser.add_argument("--models-dir", dest="models_dir", default="models")
    predict_parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])

    bench_parser = subparsers.add_parser(
        "benchmark", help="Measure end-to-end prediction latency by device."
    )
    bench_parser.add_argument("--model-pack", required=True)
    bench_parser.add_argument("--models-dir", dest="models_dir", default="models")
    bench_parser.add_argument("--devices", default="cpu", help="Comma-separated list, e.g. cpu,mps")
    bench_parser.add_argument("--item-name", default="Coffee shop purchase")
    bench_parser.add_argument("--price", default="12.50")
    bench_parser.add_argument("--warmup", type=int, default=20)
    bench_parser.add_argument("--iterations", type=int, default=200)

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    df = load_transactions(Path(args.data))
    device = resolve_device(args.device)
    encoder = SentenceTransformerEncoder(model_name=args.encoder_model, device=device)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        test_size=args.test_size,
    )
    split = split_dataset(df, test_size=config.test_size, seed=config.seed)
    trained = train_model(
        train_df=split.train_df,
        encoder=encoder,
        model_name=args.model_name,
        config=config,
        device=device,
    )
    result = evaluate_model(
        trained,
        test_df=split.test_df,
        encoder=encoder,
        class_counts_total=split.class_counts_total,
        device=device,
    )

    resolved_models_dir = _resolve_models_dir(args.models_dir)
    resolved_models_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolved_models_dir / args.model_name
    model_pack_path = save_model_pack(model_dir=model_dir, result=result)

    if result.mappings.warnings:
        for warning in result.mappings.warnings:
            print(f"warning: {warning}", file=sys.stderr)

    payload = {
        "model_pack": str(model_pack_path),
        "metrics": result.metrics,
        "manifest": result.manifest,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    resolved_models_dir = _resolve_models_dir(args.models_dir)
    model_pack_path = resolve_model_pack_path(args.model_pack, models_dir=resolved_models_dir)
    predictor = Predictor(model_pack_path=str(model_pack_path), device=args.device)
    predicted = predictor.predict(item_name=args.item_name, price=args.price)
    print(predicted)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    resolved_models_dir = _resolve_models_dir(args.models_dir)
    results = benchmark_model_pack(
        model_pack=args.model_pack,
        models_dir=resolved_models_dir,
        devices=devices,
        item_name=args.item_name,
        price=args.price,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    payload = [result.__dict__ for result in results]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
