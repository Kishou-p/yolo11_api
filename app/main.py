from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.exceptions.exception_handlers import register_exception_handlers
from app.routes.health_routes import router as health_router
from app.routes.inference_routes import router as inference_router
from app.routes.model_routes import router as model_router
from app.routes.result_routes import router as result_router


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()

    setup_logging(app_settings.log_level)

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
    )

    app.state.settings = app_settings
    app.state.model_registry = None

    _register_middlewares(app, app_settings)
    _register_routes(app, app_settings)
    register_exception_handlers(app)

    return app


def _register_middlewares(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


def _register_routes(app: FastAPI, settings: Settings) -> None:
    app.include_router(
        health_router,
        prefix=settings.api_prefix,
        tags=["health"],
    )

    app.include_router(
        model_router,
        prefix=settings.api_prefix,
        tags=["models"],
    )

    app.include_router(
        inference_router,
        prefix=settings.api_prefix,
        tags=["inference"],
    )

    app.include_router(
        result_router,
        prefix=settings.api_prefix,
        tags=["results"],
    )