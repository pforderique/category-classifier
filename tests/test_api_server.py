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
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=1,
    )
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ready"] is True
    assert payload["device"] == "cpu"
    assert payload["loaded_models"] == ["test-pack"]


def test_healthz_no_models_when_dir_empty(tmp_path: Path) -> None:
    app = create_app(
        models_dir=str(tmp_path / "models"),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["loaded_models"] == []


def test_prediction_success(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=1,
    )
    with TestClient(app) as client:
        response = client.get(
            "/models/test-pack/prediction",
            params={"item_name": "Monthly Rent", "price": "$2200.00"},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] in {"\U0001F3E0Housing", "\U0001F4FDSubscription"}


def test_prediction_rejects_invalid_price(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    app = create_app(
        models_dir=str(models_dir),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=1,
    )
    with TestClient(app) as client:
        response = client.get(
            "/models/test-pack/prediction",
            params={"item_name": "Monthly Rent", "price": "not-a-price"},
        )

    assert response.status_code == 422
    assert "Could not parse price" in response.json()["detail"]


def test_prediction_missing_model_returns_404(tmp_path: Path) -> None:
    app = create_app(
        models_dir=str(tmp_path / "models"),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
    )
    with TestClient(app) as client:
        response = client.get(
            "/models/nonexistent/prediction",
            params={"item_name": "Coffee", "price": "5.00"},
        )

    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]


def test_available_models_shows_loaded_status(
    trained_pack: Path,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "backup-pack")
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    # max_loaded_models=1 so only the first alphabetically (backup-pack) is loaded at startup.
    app = create_app(
        models_dir=str(models_dir),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=1,
    )
    with TestClient(app) as client:
        available_response = client.get("/available_models")

    assert available_response.status_code == 200
    models = available_response.json()
    assert [m["model_name"] for m in models] == ["backup-pack", "test-pack"]
    assert models[0]["active"] is True   # backup-pack loaded at startup
    assert models[1]["active"] is False  # test-pack not yet loaded
    assert all(m["size_mb"] > 0 for m in models)
    assert all(m["num_params"] > 0 for m in models)


def test_lru_eviction(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "alpha-pack")
    _copy_model_pack(trained_pack, models_dir, "beta-pack")

    # Cache holds only 1; alpha-pack loaded at startup.
    app = create_app(
        models_dir=str(models_dir),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=1,
    )
    with TestClient(app) as client:
        # Request beta-pack — alpha-pack should be evicted.
        r = client.get(
            "/models/beta-pack/prediction",
            params={"item_name": "Coffee", "price": "5.00"},
        )
        assert r.status_code == 200

        # Now available_models: only beta-pack is active.
        available = client.get("/available_models").json()
        by_name = {m["model_name"]: m for m in available}
        assert by_name["alpha-pack"]["active"] is False
        assert by_name["beta-pack"]["active"] is True


def test_prediction_loads_model_on_demand(trained_pack: Path, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _copy_model_pack(trained_pack, models_dir, "test-pack")

    # Start with max_loaded_models=0 is invalid; use an empty dir so nothing is loaded at startup,
    # then request a model to trigger on-demand loading.
    models_dir2 = tmp_path / "models2"
    _copy_model_pack(trained_pack, models_dir2, "on-demand-pack")

    app = create_app(
        models_dir=str(models_dir2),
        device="cpu",
        encoder=DummyEncoder(embedding_dim=8),
        max_loaded_models=3,
    )
    with TestClient(app) as client:
        # on-demand-pack is loaded at startup since max_loaded_models=3 and there's 1 model.
        r = client.get(
            "/models/on-demand-pack/prediction",
            params={"item_name": "Coffee", "price": "5.00"},
        )
        assert r.status_code == 200
        assert "prediction" in r.json()
