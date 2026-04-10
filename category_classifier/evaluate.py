"""Evaluation utilities for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import warnings

import matplotlib.pyplot as plt
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
    figures: dict[str, bytes] | None = None


def _predict_ids(model: LinearClassifier, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32, device=device))
    return torch.argmax(logits, dim=1).cpu().numpy()


def _generate_figures(
    confusion: np.ndarray,
    accuracy: float,
    macro_f1: float,
    labels: list[str],
) -> dict[str, bytes]:
    """Generate performance visualization figures as PNG bytes."""
    # Suppress matplotlib font glyph warnings about emoji rendering
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing from font.*")
        
        figures = {}

        # Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(confusion, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, confusion[i, j], ha="center", va="center", color="black", fontsize=10)
        fig.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figures["confusion_matrix.png"] = buf.getvalue()
        plt.close(fig)

        # Metrics summary figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        metrics_text = f"Evaluation Metrics\n\nTop-1 Accuracy: {accuracy:.4f}\nMacro F1: {macro_f1:.4f}"
        ax.text(0.5, 0.5, metrics_text, ha="center", va="center", fontsize=14, family="monospace")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figures["metrics_summary.png"] = buf.getvalue()
        plt.close(fig)

        # Per-class accuracy
        per_class_acc = np.diag(confusion) / confusion.sum(axis=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, per_class_acc)
        ax.set_xlabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.set_xlim(0, 1)
        for i, v in enumerate(per_class_acc):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center")
        fig.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figures["per_class_accuracy.png"] = buf.getvalue()
        plt.close(fig)

        return figures


def evaluate_model(
    trained: TrainedModel,
    test_df: pd.DataFrame,
    encoder: TextEncoder,
    class_counts_total: dict[str, int],
    device: Device,
    generate_graphs: bool = True,
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

    figures = None
    if generate_graphs:
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
