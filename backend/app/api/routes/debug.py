"""Development-only inference diagnostics."""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_debug_chat_model
from app.core.config import Settings, get_settings
from app.domain.inference import ChatMessage, GenerationRequest
from app.ports.inference import ChatModelPort

router = APIRouter(prefix="/v1/debug", tags=["debug"])


class DebugMessage(BaseModel):
    """Validated message accepted by the diagnostic endpoint."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_768)


class DebugGenerationRequest(BaseModel):
    """Provider-neutral diagnostic generation input."""

    model_config = ConfigDict(extra="forbid")

    system: str | None = Field(default=None, max_length=32_768)
    messages: list[DebugMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: list[str] = Field(default_factory=list, max_length=8)
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_domain(self) -> GenerationRequest:
        """Convert validated transport data into a domain request."""

        return GenerationRequest(
            system=self.system,
            messages=tuple(
                ChatMessage(role=message.role, content=message.content) for message in self.messages
            ),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=tuple(self.stop),
            metadata=self.metadata,
        )


def _encode_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.post("/generate")
async def generate(
    payload: DebugGenerationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[ChatModelPort, Depends(get_debug_chat_model)],
) -> StreamingResponse:
    """Stream normalized chunks from the configured local model adapter."""

    if payload.max_tokens > settings.model_context_ceiling:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Requested output exceeds the configured context ceiling.",
        )

    model_stream = await model.generate(payload.to_domain())

    async def events() -> AsyncIterator[str]:
        async for chunk in model_stream:
            yield _encode_event(
                "complete" if chunk.finish_reason else "token",
                {
                    "text": chunk.text,
                    "index": chunk.index,
                    "finish_reason": chunk.finish_reason,
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
