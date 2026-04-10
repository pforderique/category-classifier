from __future__ import annotations

from pathlib import Path

import pandas as pd

from category_classifier.training import split_dataset


class TestTraining:
    """Training pipeline tests."""

    def test_outputs_model_pack(self, trained_pack: Path) -> None:
        assert (trained_pack / "model.pt").exists()
        assert (trained_pack / "manifest.json").exists()
        assert (trained_pack / "label_map.json").exists()
        assert (trained_pack / "metrics.json").exists()


class TestSplit:
    """Dataset splitting tests."""

    @staticmethod
    def _dataset() -> pd.DataFrame:
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "item_name": f"housing-{idx}",
                    "date": "2024-01-02",
                    "price": 1000.0 + idx,
                    "category_display": "\U0001F3E0Housing",
                    "category_clean": "Housing",
                }
            )
        for idx in range(10):
            rows.append(
                {
                    "item_name": f"subscriptions-{idx}",
                    "date": "2024-01-03",
                    "price": 20.0 + idx,
                    "category_display": "\U0001F4FDSubscription",
                    "category_clean": "Subscription",
                }
            )
        return pd.DataFrame(rows)

    def test_split_dataset_is_deterministic(self) -> None:
        df = self._dataset()
        split_a = split_dataset(df, test_size=0.2, seed=42)
        split_b = split_dataset(df, test_size=0.2, seed=42)

        assert split_a.train_df["item_name"].tolist() == split_b.train_df["item_name"].tolist()
        assert split_a.test_df["item_name"].tolist() == split_b.test_df["item_name"].tolist()
