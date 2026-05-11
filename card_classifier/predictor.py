"""Card classifier prediction runtime."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from loguru import logger

from category_classifier.encoder import SentenceTransformerEncoder, TextEncoder
from category_classifier.model import LinearClassifier
from category_classifier.model_pack import load_model_pack
from category_classifier.preprocessing import encode_cyclical_date, normalize_category, parse_price
from category_classifier.runtime import resolve_device


class CardPredictor:
    """Loaded card model pack plus shared encoder for single-row prediction."""

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

        self.category_order: list[str] = list(manifest["category_order"])
        self.category_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.category_order)}

        model_state = self._pack.model_state
        expected_input_dim = int(model_state["input_dim"])
        n_cats = len(self.category_order)
        computed_dim = encoder.embedding_dim + 5 + n_cats  # embed + price + 4 date + onehot
        if expected_input_dim != computed_dim:
            raise ValueError(
                f"Encoder/manifest mismatch: model expects input_dim={expected_input_dim}, "
                f"computed {computed_dim} (embed={encoder.embedding_dim}, cats={n_cats})."
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

    def predict(
        self,
        item_name: str,
        price: object,
        iso_date: str | None = None,
        category: str = "",
    ) -> str:
        """Predict card label given item_name, price, date, and category."""
        text = item_name.strip()
        if not text:
            raise ValueError("item_name cannot be empty")

        parsed_price = parse_price(price)
        price_norm = (parsed_price - self.price_mean) / self.price_std

        embedding = self.encoder.encode([text])

        if iso_date is None:
            iso_date = datetime.now().date().isoformat()
        date_features = np.array([encode_cyclical_date(iso_date)], dtype=np.float32).reshape(1, -1)

        cat_clean = normalize_category(category) if category.strip() else ""
        onehot = np.zeros((1, len(self.category_order)), dtype=np.float32)
        if cat_clean in self.category_to_idx:
            onehot[0, self.category_to_idx[cat_clean]] = 1.0
        elif cat_clean:
            logger.warning("Unknown category '{}' at inference; treating as zeros.", category)

        feature = np.concatenate(
            [embedding, np.array([[price_norm]], dtype=np.float32), date_features, onehot],
            axis=1,
        )
        tensor = torch.tensor(feature, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        pred_id = int(torch.argmax(logits, dim=1).item())
        return self.id_to_display[pred_id]
