"""Training pipeline and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from category_classifier.dataset import CategoryMappings, build_category_mappings
from category_classifier.encoder import TextEncoder
from category_classifier.model import LinearClassifier
from category_classifier.runtime import Device, resolve_device


DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class TrainConfig:
    """Trainer hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    seed: int = DEFAULT_SEED
    test_size: float = DEFAULT_TEST_SIZE


@dataclass(frozen=True)
class TrainResult:
    """Outputs from train + evaluate stages."""

    model: LinearClassifier
    mappings: CategoryMappings
    metrics: dict[str, object]
    manifest: dict[str, object]
    model_state: dict[str, object]


@dataclass(frozen=True)
class TrainedModel:
    """Outputs from the training stage only."""

    model: LinearClassifier
    mappings: CategoryMappings
    manifest: dict[str, object]
    model_state: dict[str, object]
    holdout_features: np.ndarray
    holdout_labels: np.ndarray
    class_counts_total: dict[Any, int]
    training_wall_time_sec: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_dataset(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified train/test split."""
    class_counts = df["category_clean"].value_counts()
    too_small = class_counts[class_counts < 2]
    if not too_small.empty:
        joined = ", ".join([f"{label}={count}" for label, count in too_small.items()])
        raise ValueError(
            "Each category must have at least 2 rows for stratified split. "
            f"Too small: {joined}"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["category_clean"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _prepare_features(
    encoder: TextEncoder,
    item_names: list[str],
    prices: np.ndarray,
    price_mean: float,
    price_std: float,
) -> np.ndarray:
    embeddings = encoder.encode(item_names)
    price_norm = ((prices - price_mean) / price_std).reshape(-1, 1)
    features = np.concatenate([embeddings, price_norm], axis=1)
    return features.astype(np.float32)


def _train_head(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    device: str,
    config: TrainConfig,
) -> LinearClassifier:
    _set_seed(config.seed)
    model = LinearClassifier(input_dim=features.shape[1], num_classes=num_classes).to(device)

    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model


def _predict_ids(model: LinearClassifier, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32, device=device))
    return torch.argmax(logits, dim=1).cpu().numpy()


def train_model(
    df: pd.DataFrame,
    encoder: TextEncoder,
    model_name: str,
    device: Device = Device.AUTO,
    config: TrainConfig | None = None,
) -> TrainedModel:
    """Train a fixed-encoder linear head model."""
    config = config or TrainConfig()
    started_at = time.perf_counter()
    resolved_device = resolve_device(str(device))

    mappings = build_category_mappings(df)
    train_df, test_df = _split_dataset(df, test_size=config.test_size, seed=config.seed)

    price_mean = float(train_df["price"].mean())
    price_std = float(train_df["price"].std(ddof=0))
    if price_std == 0.0:
        price_std = 1.0

    train_labels = train_df["category_clean"].map(mappings.clean_to_id).to_numpy(dtype=np.int64)
    test_labels = test_df["category_clean"].map(mappings.clean_to_id).to_numpy(dtype=np.int64)

    train_features = _prepare_features(
        encoder=encoder,
        item_names=train_df["item_name"].tolist(),
        prices=train_df["price"].to_numpy(dtype=np.float32),
        price_mean=price_mean,
        price_std=price_std,
    )
    test_features = _prepare_features(
        encoder=encoder,
        item_names=test_df["item_name"].tolist(),
        prices=test_df["price"].to_numpy(dtype=np.float32),
        price_mean=price_mean,
        price_std=price_std,
    )

    model = _train_head(
        features=train_features,
        labels=train_labels,
        num_classes=len(mappings.clean_to_id),
        device=str(resolved_device),
        config=config,
    )

    class_counts_total = {
        clean: int(count) for clean, count in df["category_clean"].value_counts().to_dict().items()
    }

    elapsed = time.perf_counter() - started_at
    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "encoder_model_name": encoder.name,
        "device_used": str(resolved_device),
        "seed": config.seed,
        "test_size": config.test_size,
        "input_dim": int(train_features.shape[1]),
        "num_classes": int(len(mappings.clean_to_id)),
        "price_mean": price_mean,
        "price_std": price_std,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "class_order": [mappings.id_to_clean[i] for i in range(len(mappings.id_to_clean))],
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
        holdout_features=test_features,
        holdout_labels=test_labels,
        class_counts_total=class_counts_total,
        training_wall_time_sec=elapsed,
    )


def evaluate_model(trained: TrainedModel, device: Device) -> TrainResult:
    """Evaluate a trained model and build the full training result payload."""
    resolved_device = (
        resolve_device(str(device)) if device is not None else resolve_device(trained.manifest["device_used"])
    )
    pred_ids = _predict_ids(trained.model, trained.holdout_features, device=str(resolved_device))

    accuracy = float(accuracy_score(trained.holdout_labels, pred_ids))
    macro_f1 = float(
        f1_score(
            trained.holdout_labels,
            pred_ids,
            average="macro",
            zero_division=0,
        )
    )
    confusion = confusion_matrix(
        trained.holdout_labels,
        pred_ids,
        labels=np.arange(len(trained.mappings.clean_to_id)),
    )
    id_to_display = {
        idx: trained.mappings.clean_to_display[clean]
        for idx, clean in trained.mappings.id_to_clean.items()
    }
    metrics = {
        "top1_accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_counts_total": trained.class_counts_total,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": [id_to_display[i] for i in range(len(id_to_display))],
        "training_wall_time_sec": trained.training_wall_time_sec,
    }

    return TrainResult(
        model=trained.model,
        mappings=trained.mappings,
        metrics=metrics,
        manifest=trained.manifest,
        model_state=trained.model_state,
    )
