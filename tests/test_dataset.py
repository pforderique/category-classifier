from __future__ import annotations

from pathlib import Path

from loguru import logger
import pytest

from category_classifier.dataset import build_category_mappings, load_transactions
from category_classifier.errors import DataValidationError


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_csv_and_tsv(tmp_path: Path) -> None:
    csv_path = tmp_path / "transactions.csv"
    tsv_path = tmp_path / "transactions.tsv"

    payload = (
        "item,cost,date,category\n"
        "February Rent,\"$2,200.00\",1/2/2024,\U0001F3E0Housing\n"
        "Amex Gold Renewal Fee,$250.00,1/4/2024,\U0001F4FDSubscription\n"
    )
    _write(csv_path, payload)

    payload_tsv = (
        "ITEM\tCOST\tDATE\tCATEGORY\n"
        "February Rent\t$2,200.00\t1/2/2024\t\U0001F3E0Housing\n"
        "Amex Gold Renewal Fee\t$250.00\t1/4/2024\t\U0001F4FDSubscription\n"
    )
    _write(tsv_path, payload_tsv)

    csv_df = load_transactions(csv_path)
    tsv_df = load_transactions(tsv_path)
    assert len(csv_df) == 2
    assert len(tsv_df) == 2
    assert list(csv_df["category_clean"]) == ["Housing", "Subscription"]
    assert list(tsv_df["category_clean"]) == ["Housing", "Subscription"]


def test_load_transactions_drops_rows_missing_any_required_fields(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tsv"
    _write(
        broken,
        "item\tcost\tdate\tcategory\n"
        "Rent\t$2200.00\t1/2/2024\t\U0001F3E0Housing\n"
        "\t\t1/4/2024\t\n",
    )

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        df = load_transactions(broken)
    finally:
        logger.remove(sink_id)

    assert len(df) == 1
    assert any(
        "Dropping row 3 because required fields are missing" in message
        for message in messages
    )


def test_load_transactions_invalid_cost_still_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken-cost.tsv"
    _write(
        broken,
        "item\tcost\tdate\tcategory\n"
        "Rent\t$2200.00\t1/2/2024\t\U0001F3E0Housing\n"
        "Bad Cost\tbad-cost\t1/4/2024\t\U0001F4FDSubscription\n",
    )

    with pytest.raises(DataValidationError) as exc:
        load_transactions(broken)
    assert "row 3:" in str(exc.value)


def test_category_mapping_collision_keeps_first_display(tmp_path: Path) -> None:
    tsv = (
        "item\tcost\tdate\tcategory\n"
        "Rent\t$2200.00\t1/2/2024\t\U0001F3E0Housing\n"
        "Rent Interest\t-$10.00\t1/2/2024\t\U0001F3E0 Housing\n"
    )
    path = tmp_path / "tmp-collision.tsv"
    path.write_text(tsv, encoding="utf-8")
    try:
        df = load_transactions(path)
        mappings = build_category_mappings(df)
    finally:
        path.unlink(missing_ok=True)

    assert mappings.clean_to_display["Housing"] == "\U0001F3E0Housing"
    assert mappings.warnings
