"""Card classifier training pipeline."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import loguru

from category_classifier.dataset import CategoryMappings
from category_classifier.encoder import TextEncoder
from category_classifier.model_pack import TrainResult
from category_classifier.runtime import Device, resolve_device
from category_classifier.training import TrainConfig, TrainedModel, SplitDataset, _train_head, prepare_features


_logger = loguru.logger.bind(module="card_training")


def prepare_card_features(
    encoder: TextEncoder,
    item_names: list[str],
    categories: list[str],
    prices: np.ndarray,
    price_mean: float,
    price_std: float,
    iso_dates: list[str],
    category_order: list[str],
) -> np.ndarray:
    """Build feature vector: [item_embed | price | date(4) | category_onehot(N)]."""
    base = prepare_features(encoder, item_names, prices, price_mean, price_std, iso_dates)

    cat_to_idx = {c: i for i, c in enumerate(category_order)}
    n = len(item_names)
    onehot = np.zeros((n, len(category_order)), dtype=np.float32)
    for i, cat in enumerate(categories):
        if cat in cat_to_idx:
            onehot[i, cat_to_idx[cat]] = 1.0

    return np.concatenate([base, onehot], axis=1)


def split_card_dataset(df: pd.DataFrame, test_size: float, seed: int) -> SplitDataset:
    """Stratified train/test split by card label."""
    card_counts = df["card"].value_counts()
    too_small = card_counts[card_counts < 2]
    if not too_small.empty:
        joined = ", ".join(f"{label}={count}" for label, count in too_small.items())
        raise ValueError(
            f"Each card must have at least 2 rows for stratified split. Too small: {joined}"
        )

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["card"],
    )
    return SplitDataset(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        class_counts_total={str(c): int(n) for c, n in card_counts.items()},
    )


def train_card_model(
    train_df: pd.DataFrame,
    encoder: TextEncoder,
    model_name: str,
    config: TrainConfig | None = None,
    *,
    device: Device = Device.AUTO,
    show_progress: bool = True,
) -> TrainedModel:
    """Train a card classifier linear head on a pre-split training frame."""
    config = config or TrainConfig()
    resolved_device = resolve_device(str(device))

    # Build card label mappings (first-seen order, card values serve as both clean and display)
    seen_cards: dict[str, int] = {}
    for card in train_df["card"]:
        if card not in seen_cards:
            seen_cards[card] = len(seen_cards)
    id_to_card = {v: k for k, v in seen_cards.items()}
    mappings = CategoryMappings(
        clean_to_id=seen_cards,
        id_to_clean=id_to_card,
        clean_to_display={c: c for c in seen_cards},
        warnings=[],
    )

    # category_order = unique input category values from training data (for one-hot)
    seen_cats: dict[str, bool] = {}
    for cat in train_df["category_clean"]:
        seen_cats[cat] = True
    category_order = list(seen_cats.keys())

    price_mean = float(train_df["price"].mean())
    price_std = float(train_df["price"].std(ddof=0))
    if price_std == 0.0:
        price_std = 1.0

    train_labels = np.array([seen_cards[c] for c in train_df["card"]], dtype=np.int64)
    train_features = prepare_card_features(
        encoder=encoder,
        item_names=train_df["item_name"].tolist(),
        categories=train_df["category_clean"].tolist(),
        prices=train_df["price"].to_numpy(dtype=np.float32),
        price_mean=price_mean,
        price_std=price_std,
        iso_dates=train_df["date"].tolist(),
        category_order=category_order,
    )

    _logger.info(
        "Training card model on {} rows with {} classes using device: {}",
        len(train_df), len(seen_cards), resolved_device,
    )
    started_at = time.perf_counter()
    model = _train_head(
        features=train_features,
        labels=train_labels,
        num_classes=len(seen_cards),
        device=str(resolved_device),
        config=config,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - started_at
    _logger.info("Card model training completed in {:.2f}s", elapsed)

    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "encoder_model_name": encoder.name,
        "device_used": str(resolved_device),
        "seed": config.seed,
        "input_dim": int(train_features.shape[1]),
        "num_classes": int(len(seen_cards)),
        "price_mean": price_mean,
        "price_std": price_std,
        "class_order": [id_to_card[i] for i in range(len(id_to_card))],
        "has_date_features": True,
        "has_category_features": True,
        "category_order": category_order,
    }
    model_state = {
        "state_dict": model.state_dict(),
        "input_dim": manifest["input_dim"],
        "num_classes": manifest["num_classes"],
    }

    return TrainedModel(
        model=model,
        mappings=mappings,
        manifest=manifest,
        model_state=model_state,
        training_wall_time_sec=elapsed,
    )


def evaluate_card_model(
    trained: TrainedModel,
    test_df: pd.DataFrame,
    encoder: TextEncoder,
    class_counts_total: dict[str, int],
    device: Device,
    generate_graphs: bool = True,
) -> TrainResult:
    """Evaluate a trained card model on an explicit test frame."""
    category_order: list[str] = list(trained.manifest["category_order"])

    test_labels = np.array(
        [trained.mappings.clean_to_id[c] for c in test_df["card"]], dtype=np.int64
    )
    test_features = prepare_card_features(
        encoder=encoder,
        item_names=test_df["item_name"].tolist(),
        categories=test_df["category_clean"].tolist(),
        prices=test_df["price"].to_numpy(dtype=np.float32),
        price_mean=float(trained.manifest["price_mean"]),
        price_std=float(trained.manifest["price_std"]),
        iso_dates=test_df["date"].tolist(),
        category_order=category_order,
    )

    trained.model.eval()
    with torch.no_grad():
        logits = trained.model(
            torch.tensor(test_features, dtype=torch.float32, device=device)
        )
    pred_ids = torch.argmax(logits, dim=1).cpu().numpy()

    accuracy = float(accuracy_score(test_labels, pred_ids))
    macro_f1 = float(f1_score(test_labels, pred_ids, average="macro", zero_division=0))
    confusion = confusion_matrix(
        test_labels, pred_ids, labels=np.arange(len(trained.mappings.clean_to_id))
    )
    id_to_display = {
        idx: trained.mappings.clean_to_display[clean]
        for idx, clean in trained.mappings.id_to_clean.items()
    }
    metrics = {
        "top1_accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_counts_total": class_counts_total,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": [id_to_display[i] for i in range(len(id_to_display))],
        "training_wall_time_sec": trained.training_wall_time_sec,
    }

    figures = None
    if generate_graphs:
        from category_classifier.evaluate import _generate_figures
        labels = [id_to_display[i] for i in range(len(id_to_display))]
        figures = _generate_figures(confusion, accuracy, macro_f1, labels)

    return TrainResult(
        model=trained.model,
        mappings=trained.mappings,
        metrics=metrics,
        manifest=trained.manifest,
        model_state=trained.model_state,
        figures=figures,
    )
