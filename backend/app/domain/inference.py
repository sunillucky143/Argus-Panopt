"""Provider-neutral inference domain values."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length"]
InferenceOutcome = Literal["success", "failure", "cancelled"]
EmbeddingKind = Literal["query", "passage"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One structured conversational message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ImageRef:
    """Opaque reference to an image already held inside the deployment."""

    identifier: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A provider-independent generation request."""

    messages: tuple[ChatMessage, ...]
    system: str | None = None
    images: tuple[ImageRef, ...] = ()
    max_tokens: int = 512
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Provider-normalized token accounting for one completed generation."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """A normalized unit from a streaming model response."""

    text: str
    index: int
    finish_reason: FinishReason | None = None
    usage: GenerationUsage | None = None


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities reported consistently by every model adapter."""

    provider: str
    model_name: str
    max_context: int
    vision: bool
    streaming: bool
    quantization: str
    speculative_decoding: bool
    prefix_caching: bool


@dataclass(frozen=True, slots=True)
class InferenceMetricDimensions:
    """Bounded, configuration-derived labels permitted on inference metrics."""

    provider: str
    model_name: str


@dataclass(frozen=True, slots=True)
class InferenceRequestMetrics:
    """Content-free measurements emitted when an inference request terminates."""

    outcome: InferenceOutcome
    total_latency_seconds: float
    time_to_first_token_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    context_utilization: float | None = None
    finish_reason: FinishReason | None = None


@dataclass(frozen=True, slots=True)
class Passage:
    """Text plus trusted, structured source metadata for reranking."""

    identifier: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    """A passage with its adapter-normalized relevance score."""

    passage: Passage
    score: float


@dataclass(frozen=True, slots=True)
class VisionResult:
    """Normalized text produced from deployment-local image analysis."""

    text: str
    model_name: str


Vector = tuple[float, ...]
