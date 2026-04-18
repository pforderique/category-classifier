"""Dataset loading and label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from loguru import logger
import pandas as pd

from category_classifier.errors import DataValidationError
from category_classifier.preprocessing import normalize_category, parse_date, parse_price


REQUIRED_COLUMNS = ("item", "cost", "date", "category")


@dataclass(frozen=True)
class CategoryMappings:
    """Category mappings used by the model pack."""

    clean_to_id: dict[str, int]
    id_to_clean: dict[int, str]
    clean_to_display: dict[str, str]
    warnings: list[str]


def _detect_delimiter(path: Path) -> str:
    """Infer comma- or tab-delimited format."""
    sample_size_kb = 4 * 1024
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(sample_size_kb)

    if not sample:
        return ","

    try:
        guessed = csv.Sniffer().sniff(sample, delimiters=",\t")
        if guessed.delimiter in {",", "\t"}:
            return guessed.delimiter
    except csv.Error:
        pass

    return "\t" if sample.count("\t") > sample.count(",") else ","


def _normalize_header(value: str) -> str:
    return value.strip().lower()


def _resolve_required_columns(raw_df: pd.DataFrame) -> dict[str, str]:
    normalized_to_original: dict[str, str] = {}

    for original_name in raw_df.columns:
        normalized_name = _normalize_header(str(original_name))
        if normalized_name in normalized_to_original:
            raise DataValidationError(
                [
                    "duplicate columns after case normalization: "
                    f"'{normalized_to_original[normalized_name]}' and '{original_name}'"
                ]
            )
        normalized_to_original[normalized_name] = str(original_name)

    missing = [column for column in REQUIRED_COLUMNS if column not in normalized_to_original]
    if missing:
        raise DataValidationError([f"missing required columns: {', '.join(missing)}"])

    return {column: normalized_to_original[column] for column in REQUIRED_COLUMNS}


def load_transactions(file_path: Path) -> pd.DataFrame:
    """Load and validate transaction data from CSV/TSV file.

    Returns:
        DataFrame with columns:
        - item_name: original item name string
        - date: parsed datetime object
        - price: parsed float value
        - category_display: original category label string
        - category_clean: normalized category label string used for modeling
    """
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    delimiter = _detect_delimiter(file_path)
    raw_df = pd.read_csv(file_path, sep=delimiter, dtype=str, keep_default_na=False)

    resolved_columns = _resolve_required_columns(raw_df)

    errors: list[str] = []
    rows: list[dict[str, object]] = []

    for row_number, (_, row) in enumerate(raw_df.iterrows(), start=2):
        row_errors: list[str] = []
        missing_fields: list[str] = []

        item_raw = str(row[resolved_columns["item"]]).strip()
        cost_raw = str(row[resolved_columns["cost"]]).strip()
        date_raw = str(row[resolved_columns["date"]]).strip()
        category_raw = str(row[resolved_columns["category"]]).strip()

        if not item_raw:
            missing_fields.append("item")
        if not cost_raw:
            missing_fields.append("cost")
        if not date_raw:
            missing_fields.append("date")
        if not category_raw:
            missing_fields.append("category")

        if missing_fields:
            logger.warning(
                "Dropping row {} because required fields are missing: {}",
                row_number,
                ", ".join(missing_fields),
            )
            continue

        item_name = item_raw

        try:
            parsed_date = parse_date(date_raw)
        except ValueError as exc:
            row_errors.append(str(exc))

        try:
            parsed_price = parse_price(cost_raw)
        except ValueError as exc:
            row_errors.append(str(exc))

        display_category = category_raw

        try:
            clean_category = normalize_category(category_raw)
        except ValueError as exc:
            row_errors.append(str(exc))

        if row_errors:
            errors.append(f"row {row_number}: {', '.join(row_errors)}")
            continue

        rows.append(
            {
                "item_name": item_name,
                "date": parsed_date,
                "price": parsed_price,
                "category_display": display_category,
                "category_clean": clean_category,
            }
        )
    if errors:
        raise DataValidationError(errors)

    if not rows:
        raise DataValidationError(["dataset is empty after validation"])

    return pd.DataFrame(rows)


def build_category_mappings(df: pd.DataFrame) -> CategoryMappings:
    """Build clean/display category mappings and deterministic ids."""
    clean_to_display: dict[str, str] = {}
    warnings: list[str] = []

    for clean, display in zip(df["category_clean"], df["category_display"], strict=True):
        if clean not in clean_to_display:
            clean_to_display[clean] = display
            continue
        if clean_to_display[clean] != display:
            warnings.append(
                "category label collision: "
                f"'{clean}' seen as '{clean_to_display[clean]}' and '{display}'. "
                f"Keeping first display label '{clean_to_display[clean]}'."
            )

    clean_labels = list(clean_to_display.keys())
    clean_to_id = {clean: idx for idx, clean in enumerate(clean_labels)}
    id_to_clean = {idx: clean for clean, idx in clean_to_id.items()}

    return CategoryMappings(
        clean_to_id=clean_to_id,
        id_to_clean=id_to_clean,
        clean_to_display=clean_to_display,
        warnings=warnings,
    )
