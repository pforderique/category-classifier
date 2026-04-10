from __future__ import annotations

import numpy as np


class DummyEncoder:
    """Deterministic local encoder used in tests."""

    def __init__(self, embedding_dim: int = 8) -> None:
        self.name = "dummy-encoder"
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self._embedding_dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            seed = sum(ord(ch) for ch in text) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            rows[idx] = rng.normal(loc=0.0, scale=1.0, size=self._embedding_dim)
        return rows
