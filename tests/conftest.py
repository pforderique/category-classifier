from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from category_classifier.artifacts import save_model_pack
from category_classifier.training import TrainConfig, evaluate_model, train_model
from tests.helpers import DummyEncoder


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(8):
        rows.append(
            {
                "item_name": f"February Rent {idx}",
                "date": "2024-01-02",
                "price": 2200.0 + idx,
                "category_display": "\U0001F3E0Housing",
                "category_clean": "Housing",
            }
        )
    for idx in range(8):
        rows.append(
            {
                "item_name": f"Streaming Subscription {idx}",
                "date": "2024-01-04",
                "price": 14.99 + idx,
                "category_display": "\U0001F4FDSubscription",
                "category_clean": "Subscription",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def trained_pack(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    encoder = DummyEncoder(embedding_dim=8)
    config = TrainConfig(epochs=6, batch_size=4, seed=42)
    trained = train_model(
        df=sample_df,
        encoder=encoder,
        model_name="test-pack",
        device="cpu",
        config=config,
    )
    result = evaluate_model(trained)
    model_dir = tmp_path / "artifacts" / "test-pack"
    return save_model_pack(model_dir=model_dir, result=result)
