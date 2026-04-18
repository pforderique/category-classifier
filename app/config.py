"""Configuration helpers for the API service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class ServerConfig:
    """Runtime configuration for the inference service."""

    models_dir: str = "models"
    default_model: str | None = None
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load server configuration from environment variables."""
        models_dir = os.environ.get("MODELS_DIR", "models")
        logger.info("Using models directory: {}", models_dir)

        default_model = os.environ.get("DEFAULT_MODEL")
        if default_model:
            logger.info("Default model requested at startup: {}", default_model)

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

        return cls(
            models_dir=models_dir,
            default_model=default_model,
            device=device,
            host=host,
            port=port,
        )


def resolve_models_dir(models_dir: str) -> Path:
    """Resolve and create the models directory from an absolute or relative config value."""
    candidate = Path(models_dir).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        package_root = Path(__file__).resolve().parent.parent
        cwd_candidate = (Path.cwd() / candidate).resolve()
        package_candidate = (package_root / candidate).resolve()
        resolved = cwd_candidate if cwd_candidate.exists() else package_candidate

    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Models directory is not a directory: {resolved}")
    return resolved.resolve()
