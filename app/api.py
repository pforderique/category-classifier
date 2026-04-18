"""FastAPI route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from loguru import logger
from pydantic import BaseModel

from category_classifier.errors import ModelPackError

from app.model_runtime import (
    available_models,
    current_model,
    get_health_state,
    get_predictor,
    switch_model,
)

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> dict[str, object]:
    """Health Check"""
    return get_health_state(request.app)


@router.get("/available_models")
def get_available_models(request: Request) -> list[dict[str, object]]:
    return available_models(request.app)


@router.get("/model")
def get_current_model(request: Request) -> dict[str, object]:
    return current_model(request.app)


class SwitchModelRequest(BaseModel):
    model_name: str


@router.post("/switch")
def switch_active_model(request: Request, payload: SwitchModelRequest) -> dict[str, object]:
    try:
        return switch_model(request.app, payload.model_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ModelPackError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.exception("Model switch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load requested model.",
        ) from exc


@router.get("/prediction")
def prediction(
    request: Request,
    item_name: str = Query(..., min_length=1),
    price: str = Query(..., min_length=1),
) -> dict[str, str]:
    """Returns a prediction given item_name and price in the query"""
    predictor = get_predictor(request.app)
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
        resolved_prediction = predictor.predict(item_name=cleaned_item_name, price=price)
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

    return {"prediction": resolved_prediction}
