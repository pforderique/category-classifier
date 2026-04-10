"""Prediction runtime."""

from __future__ import annotations

import numpy as np
import torch

from category_classifier.artifacts import load_model_pack
from category_classifier.encoder import SentenceTransformerEncoder, TextEncoder
from category_classifier.model import LinearClassifier
from category_classifier.preprocessing import parse_price
from category_classifier.runtime import resolve_device


class Predictor:
    """Loaded model pack plus encoder for single-row prediction."""

    def __init__(
        self,
        model_pack_path: str,
        encoder: TextEncoder | None = None,
        device: str = "cpu",
    ) -> None:
        self._pack = load_model_pack(model_pack_path)
        self.device = resolve_device(device)

        manifest = self._pack.manifest
        label_map = self._pack.label_map

        if encoder is None:
            encoder = SentenceTransformerEncoder(
                model_name=str(manifest["encoder_model_name"]),
                device=self.device,
            )
        self.encoder = encoder

        model_state = self._pack.model_state
        expected_input_dim = int(model_state["input_dim"])
        if encoder.embedding_dim + 1 != expected_input_dim:
            raise ValueError(
                "Encoder embedding size does not match model pack: "
                f"expected {expected_input_dim - 1}, got {encoder.embedding_dim}."
            )

        self.model = LinearClassifier(
            input_dim=expected_input_dim,
            num_classes=int(model_state["num_classes"]),
        )
        self.model.load_state_dict(model_state["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.price_mean = float(manifest["price_mean"])
        self.price_std = float(manifest["price_std"])
        self.id_to_clean = {int(k): v for k, v in label_map["id_to_clean"].items()}
        clean_to_display = dict(label_map["clean_to_display"])
        self.id_to_display = {
            idx: clean_to_display[clean] for idx, clean in self.id_to_clean.items()
        }

    def predict(self, item_name: str, price: object) -> str:
        """Predict display category label (including emoji when present)."""
        text = item_name.strip()
        if not text:
            raise ValueError("item_name cannot be empty")
        parsed_price = parse_price(price)
        price_norm = (parsed_price - self.price_mean) / self.price_std

        embedding = self.encoder.encode([text])
        feature = np.concatenate([embedding, np.array([[price_norm]], dtype=np.float32)], axis=1)
        tensor = torch.tensor(feature, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        pred_id = int(torch.argmax(logits, dim=1).item())
        return self.id_to_display[pred_id]
