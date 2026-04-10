from __future__ import annotations

from pathlib import Path


def test_training_smoke_outputs_model_pack(trained_pack: Path) -> None:
    assert (trained_pack / "model.pt").exists()
    assert (trained_pack / "manifest.json").exists()
    assert (trained_pack / "label_map.json").exists()
    assert (trained_pack / "metrics.json").exists()
