"""Evaluation utilities for trained models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

from category_classifier.dataset import CategoryMappings
from category_classifier.encoder import TextEncoder
from category_classifier.runtime import Device
from category_classifier.training import TrainedModel, prepare_features
from category_classifier.model import LinearClassifier


@dataclass(frozen=True)
class TrainResult:
    """Outputs from train + evaluate stages."""

    model: LinearClassifier
    mappings: CategoryMappings
    metrics: dict[str, object]
    manifest: dict[str, object]
    model_state: dict[str, object]


def _predict_ids(model: LinearClassifier, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32, device=device))
    return torch.argmax(logits, dim=1).cpu().numpy()


def evaluate_model(
    trained: TrainedModel,
    test_df: pd.DataFrame,
    encoder: TextEncoder,
    class_counts_total: dict[str, int],
    device: Device,
) -> TrainResult:
    """Evaluate a trained model on an explicit test frame."""

    test_labels = test_df["category_clean"].map(trained.mappings.clean_to_id).to_numpy(dtype=np.int64)

    price_mean = trained.manifest["price_mean"]
    assert isinstance(price_mean, float)

    price_std = trained.manifest["price_std"]
    assert isinstance(price_std, float)

    test_features = prepare_features(
        encoder=encoder,
        item_names=test_df["item_name"].tolist(),
        prices=test_df["price"].to_numpy(dtype=np.float32),
        price_mean=price_mean,
        price_std=price_std,
    )
    pred_ids = _predict_ids(trained.model, test_features, device=device)

    accuracy = float(accuracy_score(test_labels, pred_ids))
    macro_f1 = float(
        f1_score(
            test_labels,
            pred_ids,
            average="macro",
            zero_division=0,
        )
    )
    confusion = confusion_matrix(
        test_labels,
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
        "per_class_counts_total": class_counts_total,
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