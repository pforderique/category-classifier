"""Card classifier dataset loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from category_classifier.dataset import _detect_delimiter
from category_classifier.errors import DataValidationError
from category_classifier.preprocessing import normalize_category, parse_date, parse_price


REQUIRED_COLUMNS = ("item", "cost", "date", "category", "card")


def _resolve_card_columns(raw_df: pd.DataFrame) -> dict[str, str]:
    normalized_to_original: dict[str, str] = {}
    for original_name in raw_df.columns:
        normalized = str(original_name).strip().lower()
        if normalized in normalized_to_original:
            raise DataValidationError(
                [f"duplicate columns after case normalization: '{normalized_to_original[normalized]}' and '{original_name}'"]
            )
        normalized_to_original[normalized] = str(original_name)

    missing = [c for c in REQUIRED_COLUMNS if c not in normalized_to_original]
    if missing:
        raise DataValidationError([f"missing required columns: {', '.join(missing)}"])

    return {c: normalized_to_original[c] for c in REQUIRED_COLUMNS}


def load_card_transactions(file_path: Path) -> pd.DataFrame:
    """Load card transaction CSV.

    Returns DataFrame with columns:
    - item_name, date, price, category_display, category_clean, card
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    delimiter = _detect_delimiter(file_path)
    raw_df = pd.read_csv(file_path, sep=delimiter, dtype=str, keep_default_na=False)
    resolved = _resolve_card_columns(raw_df)

    errors: list[str] = []
    rows: list[dict[str, object]] = []

    for row_number, (_, row) in enumerate(raw_df.iterrows(), start=2):
        item_raw = str(row[resolved["item"]]).strip()
        cost_raw = str(row[resolved["cost"]]).strip()
        date_raw = str(row[resolved["date"]]).strip()
        category_raw = str(row[resolved["category"]]).strip()
        card_raw = str(row[resolved["card"]]).strip()

        missing_fields = [
            name for name, val in [
                ("item", item_raw), ("cost", cost_raw), ("date", date_raw),
                ("category", category_raw), ("card", card_raw),
            ] if not val
        ]
        if missing_fields:
            logger.warning(
                "Dropping row {} because required fields are missing: {}",
                row_number, ", ".join(missing_fields),
            )
            continue

        row_errors: list[str] = []

        try:
            parsed_date = parse_date(date_raw)
        except ValueError as exc:
            row_errors.append(str(exc))
            parsed_date = ""

        try:
            parsed_price = parse_price(cost_raw)
        except ValueError as exc:
            row_errors.append(str(exc))
            parsed_price = 0.0

        try:
            clean_category = normalize_category(category_raw)
        except ValueError as exc:
            row_errors.append(str(exc))
            clean_category = ""

        if row_errors:
            errors.append(f"row {row_number}: {', '.join(row_errors)}")
            continue

        rows.append({
            "item_name": item_raw,
            "date": parsed_date,
            "price": parsed_price,
            "category_display": category_raw,
            "category_clean": clean_category,
            "card": card_raw,
        })

    if errors:
        raise DataValidationError(errors)
    if not rows:
        raise DataValidationError(["dataset is empty after validation"])

    return pd.DataFrame(rows)
