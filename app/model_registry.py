"""Helpers for discovering and loading model packs for the API service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from category_classifier.artifacts import load_model_pack
from category_classifier.errors import ModelPackError


REQUIRED_MODEL_PACK_FILES = ("model.pt", "manifest.json", "label_map.json", "metrics.json")


@dataclass(frozen=True)
class ModelInfo:
    """Serializable information about an available model pack."""

    model_name: str
    size_mb: float
    num_params: int
    active: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "size_mb": self.size_mb,
            "num_params": self.num_params,
            "active": self.active,
        }


def is_valid_model_pack_dir(path: Path) -> bool:
    """Return whether a directory contains all required model pack files."""
    return path.is_dir() and all((path / required).exists() for required in REQUIRED_MODEL_PACK_FILES)


def resolve_model_path(models_dir: Path, model_name: str) -> Path:
    """Resolve a model name to a valid model pack directory inside models_dir."""
    name = model_name.strip()
    if not name:
        raise ValueError("model_name cannot be empty.")
    if "/" in name or "\\" in name:
        raise ValueError("model_name must be a simple directory name.")

    path = (models_dir / name).resolve()
    if path.parent != models_dir.resolve():
        raise ValueError("model_name must resolve to a direct child of models directory.")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Model '{name}' not found.")
    if not is_valid_model_pack_dir(path):
        raise ModelPackError(
            f"Model '{name}' is missing required files: {', '.join(REQUIRED_MODEL_PACK_FILES)}."
        )
    return path


def model_size_mb(model_path: Path) -> float:
    """Compute approximate model pack size in MB."""
    total_bytes = 0
    for candidate in model_path.rglob("*"):
        if candidate.is_file():
            total_bytes += candidate.stat().st_size
    if total_bytes == 0:
        return 0.0
    return max(round(total_bytes / (1024 * 1024), 2), 0.01)


def model_num_params(model_path: Path) -> int:
    """Compute the parameter count from the packed model state."""
    pack = load_model_pack(model_path)
    state_dict = pack.model_state.get("state_dict")
    if not isinstance(state_dict, dict):
        return 0

    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel"):
            total += int(tensor.numel())
    return total


def build_model_info(model_path: Path, active_model_name: str | None) -> ModelInfo:
    """Build model metadata for API responses."""
    return ModelInfo(
        model_name=model_path.name,
        size_mb=model_size_mb(model_path),
        num_params=model_num_params(model_path),
        active=model_path.name == active_model_name,
    )


def list_models(models_dir: Path, active_model_name: str | None) -> list[ModelInfo]:
    """Discover valid model packs under models_dir."""
    if not models_dir.exists():
        return []

    infos: list[ModelInfo] = []
    for child in sorted(models_dir.iterdir(), key=lambda path: path.name.lower()):
        if not is_valid_model_pack_dir(child):
            continue
        infos.append(build_model_info(child, active_model_name=active_model_name))
    return infos
