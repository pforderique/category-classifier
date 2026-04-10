"""Text embedding interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    """Protocol for text embedders."""

    name: str

    @property
    def embedding_dim(self) -> int: # type: ignore
        """Embedding width."""

    def encode(self, texts: list[str]) -> np.ndarray: # type: ignore
        """Encode text values into float embeddings."""


class SentenceTransformerEncoder:
    """SentenceTransformer wrapper with lazy import."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        if (dim := self._model.get_sentence_embedding_dimension()) is not None:
            self.embedding_dim = dim
        else:
            raise ValueError(f"Failed to get embedding dimension for model {model_name}.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text values into float embeddings."""
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
