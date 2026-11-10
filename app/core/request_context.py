import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging_config import request_id_var
from app.core.metrics import Metrics

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    """Assigns a request id (reusing a valid incoming X-Request-ID), returns
    it in the response, logs one access line and records HTTP metrics."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self._app = app
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _get_request_id(scope)
        token = request_id_var.set(request_id)
        started_at = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )

            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - started_at
            route = _get_route_template(scope)
            method = scope["method"]

            self._metrics.http_requests.labels(
                method=method,
                route=route,
                status=str(status_code),
            ).inc()
            self._metrics.http_request_duration.labels(
                method=method,
                route=route,
            ).observe(duration)
            logger.info(
                "%s %s %s",
                method,
                scope["path"],
                status_code,
                extra={
                    "method": method,
                    "path": scope["path"],
                    "route": route,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "client": _get_client(scope),
                },
            )
            request_id_var.reset(token)


def _get_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate

    return uuid.uuid4().hex


def _get_route_template(scope: Scope) -> str:
    # The route template (e.g. /results/{execution_id}) keeps metric label
    # cardinality bounded, unlike the raw path.
    # Recent FastAPI versions keep the full path (with the router prefix)
    # in the effective route context; older ones in scope["route"].
    fastapi_scope = scope.get("fastapi")
    if isinstance(fastapi_scope, dict):
        effective_route = fastapi_scope.get("effective_route_context")
        path = getattr(effective_route, "path", None)
        if isinstance(path, str):
            return path

    route = scope.get("route")
    return getattr(route, "path", "unmatched")


def _get_client(scope: Scope) -> str | None:
    client = scope.get("client")
    return client[0] if client else None
