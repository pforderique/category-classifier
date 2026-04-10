from __future__ import annotations

from pathlib import Path

from category_classifier.benchmark import benchmark_model_pack
from category_classifier.runtime import is_mps_available


def test_benchmark_cpu_runs(trained_pack: Path, monkeypatch) -> None:
    from category_classifier import predictor as predictor_module

    class _DummyRuntimeEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            self.name = model_name
            self._embedding_dim = 8

        @property
        def embedding_dim(self) -> int:
            return self._embedding_dim

        def encode(self, texts: list[str]):
            import numpy as np

            arr = np.zeros((len(texts), self._embedding_dim), dtype=np.float32)
            for idx, text in enumerate(texts):
                arr[idx, 0] = float(len(text))
            return arr

    monkeypatch.setattr(predictor_module, "SentenceTransformerEncoder", _DummyRuntimeEncoder)

    results = benchmark_model_pack(
        model_pack=str(trained_pack),
        artifacts_dir=trained_pack.parent,
        devices=["cpu"],
        item_name="Coffee purchase",
        price="$4.50",
        warmup=2,
        iterations=5,
    )

    assert len(results) == 1
    assert results[0].device == "cpu"
    assert results[0].mean_ms is not None
    assert results[0].skipped_reason is None


def test_benchmark_mps_skips_when_unavailable(trained_pack: Path, monkeypatch) -> None:
    from category_classifier import benchmark as bench_module

    monkeypatch.setattr(bench_module, "is_mps_available", lambda: False)
    results = benchmark_model_pack(
        model_pack=str(trained_pack),
        artifacts_dir=trained_pack.parent,
        devices=["mps"],
        item_name="Coffee purchase",
        price="$4.50",
        warmup=1,
        iterations=2,
    )
    assert len(results) == 1
    assert results[0].device == "mps"
    assert results[0].skipped_reason


def test_mps_probe_matches_runtime_type() -> None:
    assert isinstance(is_mps_available(), bool)
