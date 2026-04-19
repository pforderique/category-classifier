"""Server entrypoint for the FastAPI inference service."""

from __future__ import annotations

import uvicorn

from app.config import ServerConfig, load_dotenv
from app.server import create_app


def main() -> None:
    """Run the API server using environment configuration."""
    load_dotenv()
    config = ServerConfig.from_env()
    app = create_app(
        models_dir=config.models_dir,
        device=config.device,
        max_loaded_models=config.max_loaded_models,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
