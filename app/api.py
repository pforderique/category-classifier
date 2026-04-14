"""FastAPI app factory and route definitions."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from loguru import logger

from category_classifier.encoder import TextEncoder
from category_classifier.predictor import Predictor

from app.config import resolve_model_pack_path


def create_app(
    model_pack_path: str,
    device: str = "auto",
    encoder: TextEncoder | None = None,
) -> FastAPI:
    """Create a FastAPI app that serves a single loaded model pack."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_model_pack_path = resolve_model_pack_path(model_pack_path)
        logger.info("Loading model pack from {}", resolved_model_pack_path)
        app.state.config = {
            "model_pack_path": str(resolved_model_pack_path),
            "device": device,
        }
        app.state.predictor = Predictor(
            model_pack_path=str(resolved_model_pack_path),
            encoder=encoder,
            device=device,
        )
        logger.info("Model pack loaded successfully")
        yield

    app = FastAPI(title="category-classifier", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        predictor = getattr(app.state, "predictor", None)
        state = getattr(app.state, "config", {})
        return {
            "status": "ok",
            "ready": predictor is not None,
            "model_pack_path": state.get("model_pack_path"),
            "device": str(getattr(predictor, "device", device)),
        }

    @app.get("/prediction/")
    def prediction(
        item_name: str = Query(..., min_length=1),
        price: str = Query(..., min_length=1),
    ) -> dict[str, str]:
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready.",
            )

        cleaned_item_name = item_name.strip()
        if not cleaned_item_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="item_name cannot be empty.",
            )

        try:
            prediction = predictor.predict(item_name=cleaned_item_name, price=price)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Could not parse price.",
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed.",
            ) from exc

        return {"prediction": prediction}

    return app