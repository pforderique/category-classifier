"""FastAPI route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from loguru import logger

from category_classifier.errors import ModelPackError

from app.model_runtime import (
    available_models,
    get_health_state,
    get_or_load_predictor,
)

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> dict[str, object]:
    """Health Check"""
    return get_health_state(request.app)


@router.get("/available_models")
def get_available_models(request: Request) -> list[dict[str, object]]:
    return available_models(request.app)


@router.get("/models/{model_name}/prediction")
@router.get("/models/{model_name}/prediction/", include_in_schema=False)
def model_prediction(
    request: Request,
    model_name: str,
    item_name: str = Query(..., min_length=1),
    price: str = Query(..., min_length=1),
) -> dict[str, str]:
    """Returns a prediction from the named model given item_name and price."""
    cleaned_item_name = item_name.strip()
    if not cleaned_item_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="item_name cannot be empty.",
        )

    try:
        predictor = get_or_load_predictor(request.app, model_name)
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
        logger.exception("Failed to load model '{}'", model_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load requested model.",
        ) from exc

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
