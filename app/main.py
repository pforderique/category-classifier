"""Server entrypoint for the FastAPI inference service."""

from __future__ import annotations

import uvicorn

from app.config import ServerConfig
from app.server import create_app


def main() -> None:
    """Run the API server using environment configuration."""
    config = ServerConfig.from_env()
    app = create_app(
        models_dir=config.models_dir,
        default_model=config.default_model,
        device=config.device,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
