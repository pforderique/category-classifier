"""Configuration helpers for the API service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration for the inference service."""

    model_pack_path: str
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load server configuration from environment variables."""
        model_pack_path = os.environ.get("MODEL_PACK_PATH")
        if not model_pack_path:
            raise ValueError("MODEL_PACK_PATH is required.")

        device = os.environ.get("INFERENCE_DEVICE", "auto")
        logger.info("Using inference device: {}", device)

        host = os.environ.get("HOST", "0.0.0.0")
        logger.info("API server will listen on host: {}", host)

        raw_port = os.environ.get("PORT", "8000")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(f"PORT must be an integer, got '{raw_port}'.") from exc
        if port < 1 or port > 65535:
            raise ValueError(f"PORT must be between 1 and 65535, got {port}.")
        logger.info("API server will listen on port: {}", port)

        return cls(model_pack_path=model_pack_path, device=device, host=host, port=port)


def resolve_model_pack_path(model_pack_path: str) -> Path:
    """Resolve a model pack path from an absolute or relative config value."""
    candidate = Path(model_pack_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if candidate.is_absolute():
        raise FileNotFoundError(f"Model pack path does not exist: {candidate}")

    package_root = Path(__file__).resolve().parent.parent
    search_roots = [Path.cwd(), package_root, package_root / "artifacts"]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Model pack path does not exist: {model_pack_path}")