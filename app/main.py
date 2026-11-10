import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.core.metrics import Metrics
from app.core.request_context import RequestContextMiddleware
from app.exceptions.app_exceptions import AppError
from app.exceptions.exception_handlers import register_exception_handlers
from app.routes.health_routes import router as health_router
from app.routes.inference_routes import router as inference_router
from app.routes.metrics_routes import router as metrics_router
from app.routes.model_routes import router as model_router
from app.routes.result_routes import router as result_router
from app.services.model_registry_service import ModelRegistryService
from app.services.retention_service import RetentionService
from app.services.yolo_service import YoloService
from app.utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()

    setup_logging(app_settings.log_level, app_settings.log_format)

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        docs_url=None if app_settings.is_production_like else "/docs",
        redoc_url=None if app_settings.is_production_like else "/redoc",
        openapi_url=(
            None if app_settings.is_production_like else "/openapi.json"
        ),
        lifespan=_lifespan,
    )

    app.state.settings = app_settings
    app.state.metrics = Metrics()
    app.state.model_registry = ModelRegistryService(
        yolo_service=YoloService(
            model_path=app_settings.yolo_model_path,
            expected_sha256=app_settings.yolo_model_sha256,
            params=app_settings.inference_params,
        ),
    )

    _register_middlewares(app, app_settings)
    _register_routes(app, app_settings)
    register_exception_handlers(app)

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    model_registry: ModelRegistryService = app.state.model_registry

    ensure_directory_exists(settings.upload_dir)
    ensure_directory_exists(settings.output_dir)

    if settings.load_model_on_startup:
        await run_in_threadpool(_load_and_warmup_model, model_registry)

    retention_task = _start_retention_task(settings)

    try:
        yield
    finally:
        if retention_task is not None:
            retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retention_task


def _start_retention_task(settings: Settings) -> asyncio.Task[None] | None:
    if settings.retention_days <= 0:
        return None

    retention_service = RetentionService(
        directories=[settings.upload_dir, settings.output_dir],
        retention_days=settings.retention_days,
    )

    return asyncio.create_task(
        _run_retention_loop(
            retention_service,
            settings.retention_interval_minutes * 60,
        )
    )


async def _run_retention_loop(
    retention_service: RetentionService,
    interval_seconds: float,
) -> None:
    while True:
        try:
            await run_in_threadpool(retention_service.purge_expired_files)
        except Exception:
            logger.exception("Retention cleanup failed.")

        await asyncio.sleep(interval_seconds)


def _load_and_warmup_model(model_registry: ModelRegistryService) -> None:
    # A failed load keeps the API up but not ready, so /health/ready
    # reports it and POST /models/load can retry without a restart.
    try:
        model_registry.load_model()
        model_registry.warmup()
    except AppError as exc:
        logger.error("Model not loaded on startup: %s", exc.log_detail or exc)
        return

    logger.info("Model loaded and warmed up on startup.")


def _register_middlewares(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    # Added last so it is the outermost middleware and sees every request.
    app.add_middleware(RequestContextMiddleware, metrics=app.state.metrics)


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

    if settings.metrics_enabled:
        app.include_router(metrics_router, tags=["metrics"])
