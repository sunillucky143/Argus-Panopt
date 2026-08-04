"""Strict PHI-safe request and response contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(min_length=1, max_length=32_768)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=512)]
ModelName = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmbeddingRequest(StrictModel):
    model: ModelName
    texts: list[Text] = Field(min_length=1, max_length=256)
    kind: Literal["query", "passage"]

    @field_validator("texts")
    @classmethod
    def reject_blank_texts(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("embedding text is blank")
        return values


class PassageRequest(StrictModel):
    identifier: Identifier
    text: Text

    @field_validator("identifier", "text")
    @classmethod
    def reject_blank_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("passage value is blank")
        return value


class RerankRequest(StrictModel):
    model: ModelName
    query: Text
    passages: list[PassageRequest] = Field(min_length=1, max_length=256)
    top_k: int = Field(ge=1, le=256)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reranking query is blank")
        return value


class EmbeddingItem(StrictModel):
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class EmbeddingResponse(StrictModel):
    model: str
    data: list[EmbeddingItem]


class RerankItem(StrictModel):
    identifier: str
    score: float


class RerankResponse(StrictModel):
    model: str
    data: list[RerankItem]


class ReadinessResponse(StrictModel):
    status: Literal["ready", "unavailable"]
    embedding_ready: bool
    reranker_ready: bool
