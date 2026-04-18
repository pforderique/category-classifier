"""FastAPI app factory for the inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from category_classifier.encoder import TextEncoder

from app.api import router as api_router
from app.config import resolve_models_dir
from app.model_runtime import configure_runtime_state


def create_app(
    models_dir: str = "models",
    default_model: str | None = None,
    device: str = "auto",
    encoder: TextEncoder | None = None,
) -> FastAPI:
    """Create a FastAPI app with model discovery and switch support."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_models_dir = resolve_models_dir(models_dir)
        app.state.config = {
            "models_dir": str(resolved_models_dir),
            "device": device,
            "default_model": default_model,
        }
        configure_runtime_state(
            app=app,
            models_dir=resolved_models_dir,
            device=device,
            encoder=encoder,
            default_model=default_model,
        )
        yield

    app = FastAPI(title="category-classifier", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app
