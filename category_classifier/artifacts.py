"""Model pack save/load utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from category_classifier.evaluate import TrainResult
from category_classifier.errors import ModelPackError


@dataclass(frozen=True)
class ModelPack:
    """Loaded model pack."""

    path: Path
    manifest: dict[str, object]
    label_map: dict[str, object]
    metrics: dict[str, object]
    model_state: dict[str, object]


def save_model_pack(model_dir: Path, result: TrainResult) -> Path:
    """Write model pack files to disk."""
    pack_dir = Path(model_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    model_path = pack_dir / "model.pt"
    manifest_path = pack_dir / "manifest.json"
    label_map_path = pack_dir / "label_map.json"
    metrics_path = pack_dir / "metrics.json"

    torch.save(result.model_state, model_path)

    manifest = dict(result.manifest)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    label_map = {
        "clean_to_id": result.mappings.clean_to_id,
        "id_to_clean": {str(idx): clean for idx, clean in result.mappings.id_to_clean.items()},
        "clean_to_display": result.mappings.clean_to_display,
    }
    with label_map_path.open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2, ensure_ascii=False, sort_keys=True)

    metrics_payload = dict(result.metrics)
    metrics_payload["mapping_warnings"] = result.mappings.warnings
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

    # Save figures if they exist
    if result.figures:
        figures_dir = pack_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        for fig_name, fig_bytes in result.figures.items():
            fig_path = figures_dir / fig_name
            with fig_path.open("wb") as handle:
                handle.write(fig_bytes)

    return pack_dir


def load_model_pack(model_dir: str | Path) -> ModelPack:
    """Load model pack files from disk."""
    pack_dir = Path(model_dir)
    model_path = pack_dir / "model.pt"
    manifest_path = pack_dir / "manifest.json"
    label_map_path = pack_dir / "label_map.json"
    metrics_path = pack_dir / "metrics.json"

    for required in (model_path, manifest_path, label_map_path, metrics_path):
        if not required.exists():
            raise ModelPackError(f"Model pack is missing required file: {required}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with label_map_path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    model_state = torch.load(model_path, map_location="cpu")

    return ModelPack(
        path=pack_dir,
        manifest=manifest,
        label_map=label_map,
        metrics=metrics,
        model_state=model_state,
    )


def resolve_model_pack_path(model_pack: str, artifacts_dir: str | Path) -> Path:
    """Resolve either an explicit path or a model name under artifacts dir."""
    candidate = Path(model_pack)
    if candidate.exists():
        return candidate

    named = Path(artifacts_dir) / model_pack
    if named.exists():
        return named

    raise FileNotFoundError(
        f"Model pack '{model_pack}' not found as a path or under artifacts dir '{artifacts_dir}'."
    )
