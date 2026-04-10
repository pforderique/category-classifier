"""Training pipeline and split utilities."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import loguru

from category_classifier.dataset import CategoryMappings, build_category_mappings
from category_classifier.encoder import TextEncoder
from category_classifier.model import LinearClassifier
from category_classifier.runtime import Device, resolve_device


_logger = loguru.logger.bind(module="training")
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
class SplitDataset:
    """Caller-managed train/test split plus total class counts."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    class_counts_total: dict[str, int]


@dataclass(frozen=True)
class TrainedModel:
    """Outputs from the training stage only."""

    model: LinearClassifier
    mappings: CategoryMappings
    manifest: dict[str, object]
    model_state: dict[str, object]
    training_wall_time_sec: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> SplitDataset:
    """Deterministic stratified train/test split with total class counts."""
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
    return SplitDataset(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        class_counts_total={str(clean): int(count) for clean, count in class_counts.to_dict().items()},
    )


def prepare_features(
    encoder: TextEncoder,
    item_names: list[str],
    prices: np.ndarray,
    price_mean: float,
    price_std: float,
) -> np.ndarray:
    """Prepare model features by encoding item names and normalizing prices."""
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
    show_progress: bool,
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

    epoch_iterator = tqdm(
        range(config.epochs), desc="Training classifier", unit="epoch", leave=False
    ) if show_progress else range(config.epochs)

    for _ in epoch_iterator:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model


def train_model(
    train_df: pd.DataFrame,
    encoder: TextEncoder,
    model_name: str,
    config: TrainConfig | None = None,
    *,
    device: Device = Device.AUTO,
    show_progress: bool = True,
) -> TrainedModel:
    """Train a fixed-encoder linear head model on a pre-split training frame."""
    config = config or TrainConfig()
    resolved_device = resolve_device(str(device))
    mappings = build_category_mappings(train_df)

    price_mean = float(train_df["price"].mean())
    price_std = float(train_df["price"].std(ddof=0))
    if price_std == 0.0:
        price_std = 1.0

    train_labels = train_df["category_clean"].map(mappings.clean_to_id).to_numpy(dtype=np.int64)

    train_features = prepare_features(
        encoder=encoder,
        item_names=train_df["item_name"].tolist(),
        prices=train_df["price"].to_numpy(dtype=np.float32),
        price_mean=price_mean,
        price_std=price_std,
    )

    _logger.info(f"Training on {len(train_df)} rows with {len(mappings.clean_to_id)} classes using device: {resolved_device}")
    started_at = time.perf_counter()
    model = _train_head(
        features=train_features,
        labels=train_labels,
        num_classes=len(mappings.clean_to_id),
        device=str(resolved_device),
        config=config,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - started_at
    _logger.info(f"Training completed in {elapsed:.2f} seconds")

    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "encoder_model_name": encoder.name,
        "device_used": str(resolved_device),
        "seed": config.seed,
        "input_dim": int(train_features.shape[1]),
        "num_classes": int(len(mappings.clean_to_id)),
        "price_mean": price_mean,
        "price_std": price_std,
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
        training_wall_time_sec=elapsed,
    )
