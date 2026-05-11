"""FastAPI card classifier route definitions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from loguru import logger

from category_classifier.errors import ModelPackError
from app.card_model_runtime import available_card_models, get_or_load_card_predictor

router = APIRouter(prefix="/card")


@router.get("/available_models")
def get_available_card_models(request: Request) -> list[dict[str, object]]:
    return available_card_models(request.app)


@router.get("/models/{model_name}/prediction")
@router.get("/models/{model_name}/prediction/", include_in_schema=False)
def card_prediction(
    request: Request,
    model_name: str,
    item_name: str = Query(..., min_length=1),
    price: str = Query(..., min_length=1),
    category: str = Query(..., min_length=1),
    date: str | None = Query(None),
) -> dict[str, str]:
    """Returns a card prediction given item_name, price, category, and optional date."""
    cleaned_item_name = item_name.strip()
    if not cleaned_item_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="item_name cannot be empty.",
        )

    try:
        predictor = get_or_load_card_predictor(request.app, model_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ModelPackError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to load card model '{}'", model_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load requested model.",
        ) from exc

    try:
        prediction = predictor.predict(
            item_name=cleaned_item_name, price=price, iso_date=date, category=category
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Could not parse price or date.",
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Card prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed."
        ) from exc

    return {"prediction": prediction}
