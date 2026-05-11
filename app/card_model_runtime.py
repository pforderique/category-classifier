"""Card model LRU cache runtime."""

from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from fastapi import FastAPI
from loguru import logger

from card_classifier.predictor import CardPredictor
from app.model_registry import is_valid_model_pack_dir, list_models, resolve_model_path


def configure_card_runtime_state(
    app: FastAPI,
    *,
    card_models_dir: Path,
    device: str,
    max_loaded_models: int = 3,
) -> None:
    """Initialize card model LRU cache state on the FastAPI app."""
    app.state.card_models_dir = card_models_dir
    app.state.card_device = device
    app.state.card_model_lock = RLock()
    app.state.card_max_loaded_models = max_loaded_models
    app.state.card_model_cache = OrderedDict()
    _initialize_card_cache(app)


def _initialize_card_cache(app: FastAPI) -> None:
    models_dir: Path = app.state.card_models_dir
    max_n: int = app.state.card_max_loaded_models

    if not models_dir.exists():
        return

    loaded = 0
    for child in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if loaded >= max_n:
            break
        if not is_valid_model_pack_dir(child):
            continue
        try:
            predictor = _load_card_predictor(app, child.name)
            app.state.card_model_cache[child.name] = predictor
            loaded += 1
        except Exception:
            logger.exception("Failed to load card model '{}' at startup", child.name)


def _load_card_predictor(app: FastAPI, model_name: str) -> CardPredictor:
    """Resolve and load a CardPredictor, reusing the shared encoder."""
    target_path = resolve_model_path(models_dir=app.state.card_models_dir, model_name=model_name)
    logger.info("Loading card model '{}'", model_name)
    predictor = CardPredictor(
        model_pack_path=str(target_path),
        encoder=app.state.encoder,
        device=app.state.card_device,
    )
    # Bootstrap the shared encoder if not yet initialized.
    if app.state.encoder is None:
        app.state.encoder = predictor.encoder
    logger.info("Card model '{}' loaded", model_name)
    return predictor


def get_or_load_card_predictor(app: FastAPI, model_name: str) -> CardPredictor:
    """Return the card predictor for model_name, loading it (with LRU eviction) if needed."""
    with app.state.card_model_lock:
        cache: OrderedDict[str, CardPredictor] = app.state.card_model_cache

        if model_name in cache:
            cache.move_to_end(model_name)
            return cache[model_name]

        if len(cache) >= app.state.card_max_loaded_models:
            evicted_name, evicted = cache.popitem(last=False)
            logger.info("Evicting card model '{}' from cache (LRU)", evicted_name)
            del evicted
            gc.collect()

        predictor = _load_card_predictor(app, model_name)
        cache[model_name] = predictor
        return predictor


def available_card_models(app: FastAPI) -> list[dict[str, object]]:
    """List available card model packs; active=True means currently in LRU cache."""
    with app.state.card_model_lock:
        cached_names: set[str] = set(app.state.card_model_cache.keys())

    infos = list_models(models_dir=app.state.card_models_dir, active_model_name=None)
    result = []
    for info in infos:
        d = info.as_dict()
        d["active"] = info.model_name in cached_names
        result.append(d)
    return result
