"""Server entrypoint for the FastAPI inference service."""

from __future__ import annotations

import uvicorn
from app.api import create_app
from app.config import ServerConfig


def main() -> None:
    """Run the API server using environment configuration."""
    config = ServerConfig.from_env()
    app = create_app(
        model_pack_path=config.model_pack_path,
        device=config.device,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
