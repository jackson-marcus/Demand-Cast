"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from demandcast import __version__
from demandcast.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="demandcast",
        description="Retail demand forecasting API (probabilistic, precomputed)",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
