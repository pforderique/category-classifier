from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from tests.helpers import DummyEncoder


def test_healthz_reports_ready(trained_pack: Path) -> None:
    app = create_app(
        model_pack_path=str(trained_pack),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ready"] is True
    assert payload["device"] == "cpu"


def test_prediction_success(trained_pack: Path) -> None:
    app = create_app(
        model_pack_path=str(trained_pack),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get(
            "/prediction/",
            params={"item_name": "Monthly Rent", "price": "$2200.00"},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] in {"\U0001F3E0Housing", "\U0001F4FDSubscription"}


def test_prediction_rejects_invalid_price(trained_pack: Path) -> None:
    app = create_app(
        model_pack_path=str(trained_pack),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get(
            "/prediction/",
            params={"item_name": "Monthly Rent", "price": "not-a-price"},
        )

    assert response.status_code == 422
    assert "Could not parse price" in response.json()["detail"]


def test_startup_fails_when_model_pack_is_missing(tmp_path: Path) -> None:
    app = create_app(
        model_pack_path=str(tmp_path / "missing-pack"),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass