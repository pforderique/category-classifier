from __future__ import annotations

from pathlib import Path

from category_classifier.predictor import Predictor
from tests.helpers import DummyEncoder


def test_predictor_returns_display_label_with_emoji(trained_pack: Path) -> None:
    predictor = Predictor(
        model_pack_path=str(trained_pack),
        encoder=DummyEncoder(embedding_dim=8),
        device="cpu",
    )
    predicted = predictor.predict(item_name="Monthly Rent", price="$2200.00")
    assert predicted in {"\U0001F3E0Housing", "\U0001F4FDSubscription"}
