from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from category_classifier.model_pack import save_model_pack
from category_classifier.evaluate import evaluate_model
from category_classifier.runtime import Device
from category_classifier.training import TrainConfig, split_dataset, train_model
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
    split = split_dataset(sample_df, test_size=config.test_size, seed=config.seed)
    trained = train_model(
        train_df=split.train_df,
        encoder=encoder,
        model_name="test-pack",
        config=config,
        device=Device.CPU,
    )
    result = evaluate_model(
        trained,
        test_df=split.test_df,
        encoder=encoder,
        class_counts_total=split.class_counts_total,
        device=Device.CPU,
        generate_graphs=False,
    )
    model_dir = tmp_path / "seed-models" / "test-pack"
    return save_model_pack(model_dir=model_dir, result=result)
