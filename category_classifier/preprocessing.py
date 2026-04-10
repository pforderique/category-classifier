"""Data parsing and normalization helpers."""

from __future__ import annotations

from datetime import datetime
import unicodedata


DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d")


def parse_price(value: object) -> float:
    """Parse a signed currency-like price into float."""
    if isinstance(value, (int, float)):
        return float(value)

    if value is None:
        raise ValueError("price is missing")

    text = str(value).strip()
    if not text:
        raise ValueError("price is empty")

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace("$", "").replace(",", "").replace(" ", "")
    if not text:
        raise ValueError("price has no numeric value")

    parsed = float(text)
    return -parsed if negative else parsed


def parse_date(value: object) -> str:
    """Parse a date and return an ISO date string."""
    if value is None:
        raise ValueError("date is missing")

    text = str(value).strip()
    if not text:
        raise ValueError("date is empty")

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError as exc:
        raise ValueError(f"date '{text}' is not in a supported format") from exc


def strip_leading_emoji(text: str) -> str:
    """Strip leading emoji or pictograph symbols, preserving trailing text."""
    value = text.strip()
    idx = 0
    while idx < len(value):
        ch = value[idx]
        category = unicodedata.category(ch)
        if ch.isspace() or ch in ("\ufe0f", "\u200d"):
            idx += 1
            continue
        if category in {"So", "Sk"}:
            idx += 1
            continue
        break
    return value[idx:].strip()


def normalize_category(value: object) -> str:
    """Normalize category for internal label usage."""
    if value is None:
        raise ValueError("category is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("category is empty")
    cleaned = strip_leading_emoji(text).strip()
    if not cleaned:
        raise ValueError("category became empty after emoji stripping")
    return cleaned
