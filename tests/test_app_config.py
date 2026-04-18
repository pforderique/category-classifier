from __future__ import annotations

import os
from pathlib import Path

from app.config import load_dotenv


def test_load_dotenv_reads_from_cwd(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MODELS_DIR=./custom-models\nPORT=9001\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    loaded = load_dotenv()

    assert loaded == env_path
    assert os.environ["MODELS_DIR"] == "./custom-models"
    assert os.environ["PORT"] == "9001"


def test_load_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("PORT=9001\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PORT", "8000")

    loaded = load_dotenv()

    assert loaded == env_path
    assert os.environ["PORT"] == "8000"
