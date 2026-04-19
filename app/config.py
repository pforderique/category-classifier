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
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    max_loaded_models: int = 3

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load server configuration from environment variables."""
        models_dir = os.environ.get("MODELS_DIR")

        legacy_model_pack_path = os.environ.get("MODEL_PACK_PATH")
        if legacy_model_pack_path:
            legacy_path = Path(legacy_model_pack_path).expanduser()
            if models_dir is None:
                models_dir = str(legacy_path.parent)
            logger.warning(
                "MODEL_PACK_PATH is deprecated; prefer MODELS_DIR."
            )

        if models_dir is None:
            models_dir = "models"
        logger.info("Using models directory: {}", models_dir)

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

        raw_max = os.environ.get("MAX_LOADED_MODELS", "3")
        try:
            max_loaded_models = int(raw_max)
        except ValueError as exc:
            raise ValueError(f"MAX_LOADED_MODELS must be an integer, got '{raw_max}'.") from exc
        if max_loaded_models < 1:
            raise ValueError(f"MAX_LOADED_MODELS must be at least 1, got {max_loaded_models}.")
        logger.info("Max models held in memory: {}", max_loaded_models)

        return cls(
            models_dir=models_dir,
            device=device,
            host=host,
            port=port,
            max_loaded_models=max_loaded_models,
        )


def load_dotenv() -> Path | None:
    """Load environment variables from a local .env file when present.

    Existing process environment variables take precedence.
    """
    project_root = Path(__file__).resolve().parent.parent
    candidates = [Path.cwd() / ".env", project_root / ".env"]

    env_path: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            env_path = candidate
            break

    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if key:
            os.environ.setdefault(key, value)

    logger.info("Loaded environment from {}", env_path)
    return env_path


def resolve_models_dir(models_dir: str) -> Path:
    """Resolve and create the models directory from an absolute or relative config value."""
    candidate = Path(models_dir).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        package_root = Path(__file__).resolve().parent.parent
        resolved = (package_root / candidate).resolve()

    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Models directory is not a directory: {resolved}")
    return resolved.resolve()
