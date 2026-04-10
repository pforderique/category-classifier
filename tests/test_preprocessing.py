from __future__ import annotations

import pytest

from category_classifier.preprocessing import normalize_category, parse_price, strip_leading_emoji


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
