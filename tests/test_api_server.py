from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from tests.helpers import DummyEncoder


def _copy_model_pack(src: Path, dst_root: Path, name: str) -> Path:
    dst = dst_root / name
    shutil.copytree(src, dst)
    return dst


def test_healthz_reports_ready(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model = _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model=model.name,
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
    assert payload["active_model"] == "test-pack"


def test_prediction_success(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model = _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model=model.name,
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


def test_prediction_rejects_invalid_price(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model = _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model=model.name,
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


def test_startup_fails_when_default_model_is_missing(tmp_path: Path) -> None:
    app = create_app(
        models_dir=str(tmp_path / "models"),
        default_model="missing-pack",
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


def test_available_models_and_current_model_endpoints(
    trained_pack: Path,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")
    _copy_model_pack(trained_pack, models_dir, "backup-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model="test-pack",
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        available_response = client.get("/available_models")
        current_response = client.get("/model")

    assert available_response.status_code == 200
    models = available_response.json()
    assert [model["model_name"] for model in models] == ["backup-pack", "test-pack"]
    assert any(model["active"] is True for model in models)
    assert all(model["size_mb"] > 0 for model in models)
    assert all(model["num_params"] > 0 for model in models)

    assert current_response.status_code == 200
    current_model = current_response.json()
    assert current_model["model_name"] == "test-pack"
    assert current_model["active"] is True
    assert current_model["size_mb"] > 0
    assert current_model["num_params"] > 0


def test_model_endpoint_when_no_model_loaded(tmp_path: Path) -> None:
    app = create_app(
        models_dir=str(tmp_path / "models"),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get("/model")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": None,
        "size_mb": None,
        "num_params": None,
        "active": False,
    }


def test_switch_model(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")
    _copy_model_pack(trained_pack, models_dir, "backup-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model="test-pack",
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        switch_response = client.post("/switch", json={"model_name": "backup-pack"})
        current_response = client.get("/model")

    assert switch_response.status_code == 200
    assert switch_response.json()["model_name"] == "backup-pack"
    assert switch_response.json()["active"] is True
    assert current_response.status_code == 200
    assert current_response.json()["model_name"] == "backup-pack"


def test_switch_missing_model_fails_early(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        default_model="test-pack",
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        switch_response = client.post("/switch", json={"model_name": "missing-pack"})
        current_response = client.get("/model")

    assert switch_response.status_code == 404
    assert "missing-pack" in switch_response.json()["detail"]
    assert current_response.status_code == 200
    assert current_response.json()["model_name"] == "test-pack"
