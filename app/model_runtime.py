"""Application state helpers for model discovery and switching."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import RLock

from fastapi import FastAPI
from loguru import logger

from category_classifier.encoder import TextEncoder
from category_classifier.predictor import Predictor

from app.model_registry import build_model_info, list_models, resolve_model_path


def configure_runtime_state(
    app: FastAPI,
    *,
    models_dir: Path,
    device: str,
    encoder: TextEncoder | None = None,
    default_model: str | None = None,
) -> None:
    """Initialize mutable model state on the FastAPI app."""
    app.state.models_dir = models_dir
    app.state.device = device
    app.state.encoder = encoder
    app.state.model_lock = RLock()
    app.state.predictor = None
    app.state.active_model_name = None
    app.state.active_model_path = None
    if default_model:
        switch_model(app, default_model)


def get_predictor(app: FastAPI) -> Predictor | None:
    """Get the current predictor snapshot."""
    with app.state.model_lock:
        return app.state.predictor


def get_health_state(app: FastAPI) -> dict[str, object]:
    """Get lightweight health/config information."""
    with app.state.model_lock:
        predictor = app.state.predictor
        active_model_name = app.state.active_model_name
        models_dir = app.state.models_dir
        configured_device = app.state.device

    return {
        "status": "ok",
        "ready": predictor is not None,
        "models_dir": str(models_dir),
        "active_model": active_model_name,
        "device": str(getattr(predictor, "device", configured_device)),
    }


def available_models(app: FastAPI) -> list[dict[str, object]]:
    """List available model packs with active status and metadata."""
    with app.state.model_lock:
        models_dir = app.state.models_dir
        active_model_name = app.state.active_model_name

    infos = list_models(models_dir=models_dir, active_model_name=active_model_name)
    return [info.as_dict() for info in infos]


def current_model(app: FastAPI) -> dict[str, object]:
    """Return metadata for the currently active model, if any."""
    with app.state.model_lock:
        active_model_name = app.state.active_model_name
        active_model_path = app.state.active_model_path

    if not active_model_name or not active_model_path:
        return {
            "model_name": None,
            "size_mb": None,
            "num_params": None,
            "active": False,
        }

    info = build_model_info(
        model_path=Path(active_model_path),
        active_model_name=active_model_name,
    )
    return info.as_dict()


def switch_model(app: FastAPI, model_name: str) -> dict[str, object]:
    """Switch to a new model by name after unloading any active model."""
    with app.state.model_lock:
        models_dir = app.state.models_dir
        device = app.state.device
        encoder = app.state.encoder

        target_model_path = resolve_model_path(models_dir=models_dir, model_name=model_name)

        previous_predictor = app.state.predictor
        app.state.predictor = None
        app.state.active_model_name = None
        app.state.active_model_path = None

        if previous_predictor is not None:
            del previous_predictor
            gc.collect()

        logger.info("Loading model '{}'", target_model_path.name)
        predictor = Predictor(model_pack_path=str(target_model_path), encoder=encoder, device=device)
        logger.info("Model '{}' loaded successfully", target_model_path.name)

        app.state.predictor = predictor
        app.state.active_model_name = target_model_path.name
        app.state.active_model_path = str(target_model_path)

    model_info = build_model_info(
        model_path=target_model_path,
        active_model_name=target_model_path.name,
    )
    return model_info.as_dict()
