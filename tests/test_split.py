from __future__ import annotations

import pandas as pd

from category_classifier.training import _split_dataset


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


def test_split_dataset_is_deterministic() -> None:
    df = _dataset()
    train_a, test_a = _split_dataset(df, test_size=0.2, seed=42)
    train_b, test_b = _split_dataset(df, test_size=0.2, seed=42)

    assert train_a["item_name"].tolist() == train_b["item_name"].tolist()
    assert test_a["item_name"].tolist() == test_b["item_name"].tolist()
