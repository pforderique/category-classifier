"""Card classifier command line interface."""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

from category_classifier.encoder import SentenceTransformerEncoder
from category_classifier.model_pack import save_model_pack
from category_classifier.runtime import resolve_device
from card_classifier.dataset import load_card_transactions
from card_classifier.training import TrainConfig, evaluate_card_model, split_card_dataset, train_card_model

DEFAULT_ENCODER_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def _resolve_dir(dir_str: str) -> Path:
    candidate = Path(dir_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    package_root = Path(__file__).resolve().parent.parent
    return (package_root / candidate).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="category-classifier-train-card")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a card classifier model pack from CSV data.")
    train_parser.add_argument("--data", required=True, help="Path to CSV with card transaction rows.")
    train_parser.add_argument("--model-name", required=True, help="Name of the output model pack.")
    train_parser.add_argument("--card-models-dir", dest="card_models_dir", default="card-models")
    train_parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--learning-rate", type=float, default=1e-2)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--encoder-model", default=DEFAULT_ENCODER_MODEL)

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    df = load_card_transactions(Path(args.data))
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

    split = split_card_dataset(df, test_size=config.test_size, seed=config.seed)
    trained = train_card_model(
        train_df=split.train_df,
        encoder=encoder,
        model_name=args.model_name,
        config=config,
        device=device,
    )
    result = evaluate_card_model(
        trained,
        test_df=split.test_df,
        encoder=encoder,
        class_counts_total=split.class_counts_total,
        device=device,
    )

    resolved_dir = _resolve_dir(args.card_models_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolved_dir / args.model_name
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
