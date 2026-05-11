"""FastAPI app factory for the inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from category_classifier.encoder import TextEncoder

from app.api import router as api_router
from app.card_api import router as card_api_router
from app.config import resolve_models_dir
from app.model_runtime import configure_runtime_state
from app.card_model_runtime import configure_card_runtime_state


def create_app(
    models_dir: str = "models",
    card_models_dir: str = "card-models",
    device: str = "auto",
    encoder: TextEncoder | None = None,
    max_loaded_models: int = 3,
) -> FastAPI:
    """Create a FastAPI app with LRU model caches for category and card classifiers."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_models_dir = resolve_models_dir(models_dir)
        resolved_card_models_dir = resolve_models_dir(card_models_dir)
        app.state.config = {
            "models_dir": str(resolved_models_dir),
            "card_models_dir": str(resolved_card_models_dir),
            "device": device,
            "max_loaded_models": max_loaded_models,
        }
        configure_runtime_state(
            app=app,
            models_dir=resolved_models_dir,
            device=device,
            encoder=encoder,
            max_loaded_models=max_loaded_models,
        )
        configure_card_runtime_state(
            app=app,
            card_models_dir=resolved_card_models_dir,
            device=device,
            max_loaded_models=max_loaded_models,
        )
        yield

    app = FastAPI(title="category-classifier", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    app.include_router(card_api_router)
    return app
