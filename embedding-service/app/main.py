"""FastAPI entrypoint for the deployment-local retrieval gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings
from app.gateway import RetrievalEngineError, RetrievalGateway, RetrievalRequestError
from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    ReadinessResponse,
    RerankRequest,
    RerankResponse,
)


def create_app(
    settings: Settings | None = None,
    *,
    embedding_transport: httpx.AsyncBaseTransport | None = None,
    reranker_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        gateway = RetrievalGateway(
            configured,
            embedding_transport=embedding_transport,
            reranker_transport=reranker_transport,
        )
        app.state.gateway = gateway
        try:
            yield
        finally:
            await gateway.close()

    application = FastAPI(
        title="Argus Panopt Retrieval Gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid retrieval request."})

    @application.exception_handler(RetrievalRequestError)
    async def request_error_handler(_: Request, __: RetrievalRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "Invalid retrieval request."})

    @application.exception_handler(RetrievalEngineError)
    async def engine_error_handler(_: Request, __: RetrievalEngineError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Local retrieval engine is unavailable."},
        )

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready", response_model=ReadinessResponse)
    async def ready(request: Request) -> JSONResponse | ReadinessResponse:
        gateway: RetrievalGateway = request.app.state.gateway
        embedding_ready, reranker_ready = await gateway.readiness()
        response = ReadinessResponse(
            status="ready" if embedding_ready and reranker_ready else "unavailable",
            embedding_ready=embedding_ready,
            reranker_ready=reranker_ready,
        )
        if not embedding_ready or not reranker_ready:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @application.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def embeddings(payload: EmbeddingRequest, request: Request) -> dict[str, object]:
        gateway: RetrievalGateway = request.app.state.gateway
        return await gateway.embed(payload)

    @application.post("/v1/rerank", response_model=RerankResponse)
    async def rerank(payload: RerankRequest, request: Request) -> dict[str, object]:
        gateway: RetrievalGateway = request.app.state.gateway
        return await gateway.rerank(payload)

    return application


app = create_app()
