"""Application state helpers for multi-model LRU cache."""

from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from fastapi import FastAPI
from loguru import logger

from category_classifier.encoder import TextEncoder
from category_classifier.predictor import Predictor

from app.model_registry import build_model_info, is_valid_model_pack_dir, list_models, resolve_model_path


def configure_runtime_state(
    app: FastAPI,
    *,
    models_dir: Path,
    device: str,
    encoder: TextEncoder | None = None,
    max_loaded_models: int = 3,
) -> None:
    """Initialize LRU model cache state on the FastAPI app."""
    app.state.models_dir = models_dir
    app.state.device = device
    app.state.encoder = encoder
    app.state.model_lock = RLock()
    app.state.max_loaded_models = max_loaded_models
    app.state.model_cache = OrderedDict()  # model_name -> Predictor, LRU order (oldest first)
    _initialize_cache(app)


def _initialize_cache(app: FastAPI) -> None:
    """Load up to max_loaded_models models from models_dir at startup."""
    models_dir: Path = app.state.models_dir
    max_n: int = app.state.max_loaded_models

    if not models_dir.exists():
        return

    loaded = 0
    for child in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if loaded >= max_n:
            break
        if not is_valid_model_pack_dir(child):
            continue
        try:
            predictor = _load_predictor(app, child.name)
            app.state.model_cache[child.name] = predictor
            loaded += 1
        except Exception:
            logger.exception("Failed to load model '{}' at startup", child.name)


def _load_predictor(app: FastAPI, model_name: str) -> Predictor:
    """Resolve and load a Predictor, reusing the shared encoder."""
    target_path = resolve_model_path(models_dir=app.state.models_dir, model_name=model_name)
    logger.info("Loading model '{}'", model_name)
    predictor = Predictor(
        model_pack_path=str(target_path),
        encoder=app.state.encoder,
        device=app.state.device,
    )
    # Store the encoder after first load so all subsequent models share it.
    if app.state.encoder is None:
        app.state.encoder = predictor.encoder
    logger.info("Model '{}' loaded", model_name)
    return predictor


def get_or_load_predictor(app: FastAPI, model_name: str) -> Predictor:
    """Return the predictor for model_name, loading it (with LRU eviction) if needed."""
    with app.state.model_lock:
        cache: OrderedDict[str, Predictor] = app.state.model_cache

        if model_name in cache:
            cache.move_to_end(model_name)
            return cache[model_name]

        # Evict least-recently-used entry if at capacity.
        if len(cache) >= app.state.max_loaded_models:
            evicted_name, evicted_predictor = cache.popitem(last=False)
            logger.info("Evicting model '{}' from cache (LRU)", evicted_name)
            del evicted_predictor
            gc.collect()

        predictor = _load_predictor(app, model_name)
        cache[model_name] = predictor
        return predictor


def get_health_state(app: FastAPI) -> dict[str, object]:
    """Get lightweight health/config information."""
    with app.state.model_lock:
        cache: OrderedDict = app.state.model_cache
        loaded_models = list(cache.keys())
        configured_device = app.state.device
        first_predictor = next(iter(cache.values()), None)

    return {
        "status": "ok",
        "ready": len(loaded_models) > 0,
        "models_dir": str(app.state.models_dir),
        "loaded_models": loaded_models,
        "device": str(getattr(first_predictor, "device", configured_device)),
    }


def available_models(app: FastAPI) -> list[dict[str, object]]:
    """List available model packs; active=True means currently in the LRU cache."""
    with app.state.model_lock:
        models_dir = app.state.models_dir
        cached_names: set[str] = set(app.state.model_cache.keys())

    infos = list_models(models_dir=models_dir, active_model_name=None)
    result = []
    for info in infos:
        d = info.as_dict()
        d["active"] = info.model_name in cached_names
        result.append(d)
    return result
