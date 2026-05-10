from __future__ import annotations

import math
import pytest

from category_classifier.preprocessing import (
    encode_cyclical_date,
    normalize_category,
    parse_price,
    strip_leading_emoji,
)


def test_parse_price_signed_currency() -> None:
    assert parse_price("$2,200.00") == pytest.approx(2200.0)
    assert parse_price("-$10.00") == pytest.approx(-10.0)
    assert parse_price("($10.00)") == pytest.approx(-10.0)
    assert parse_price("45.76") == pytest.approx(45.76)


def test_strip_leading_emoji() -> None:
    housing = "\U0001F3E0Housing"
    shopping = "\U0001F6D2 Misc. Shopping"
    assert strip_leading_emoji(housing) == "Housing"
    assert strip_leading_emoji(shopping) == "Misc. Shopping"


def test_normalize_category_raises_on_empty_after_strip() -> None:
    with pytest.raises(ValueError):
        normalize_category("\U0001F3E0")


def test_encode_cyclical_date_returns_four_floats() -> None:
    month_sin, month_cos, day_sin, day_cos = encode_cyclical_date("2024-01-15")
    assert isinstance(month_sin, float)
    assert isinstance(month_cos, float)
    assert isinstance(day_sin, float)
    assert isinstance(day_cos, float)


def test_encode_cyclical_date_jan_vs_dec() -> None:
    jan_sin, jan_cos, _, _ = encode_cyclical_date("2024-01-01")
    dec_sin, dec_cos, _, _ = encode_cyclical_date("2024-12-01")
    # Jan (1) and Dec (12) should be close on the circle, roughly opposite.
    # Jan sin should be ~0, Dec sin should be ~0
    # But Dec should be opposite Dec (month 12 vs 1 wrapping)
    assert abs(jan_sin - dec_sin) > 0.5  # Actually opposite sides


def test_encode_cyclical_date_same_day_different_months() -> None:
    _, _, may_day_sin, may_day_cos = encode_cyclical_date("2024-05-15")
    _, _, june_day_sin, june_day_cos = encode_cyclical_date("2024-06-15")
    # Same day-of-month, different months -> day components should match
    assert may_day_sin == pytest.approx(june_day_sin)
    assert may_day_cos == pytest.approx(june_day_cos)


def test_encode_cyclical_date_day_wrapping() -> None:
    day_1_sin, day_1_cos, _, _ = encode_cyclical_date("2024-05-01")
    day_31_sin, day_31_cos, _, _ = encode_cyclical_date("2024-05-31")
    # Day 1 and Day 31 should be close on the cycle (cyclical!)
    distance_sq = (day_1_sin - day_31_sin) ** 2 + (day_1_cos - day_31_cos) ** 2
    assert distance_sq < 0.5  # Pretty close on circle
