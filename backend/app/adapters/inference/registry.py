"""Configuration-driven model adapter registry."""

from collections.abc import Callable

from app.adapters.inference.fake import FakeModelAdapter
from app.core.config import Settings
from app.ports.inference import ChatModelPort

ModelFactory = Callable[[Settings], ChatModelPort]


class ModelAdapterUnavailableError(RuntimeError):
    """Raised when a configured provider has not been registered."""


class ModelAdapterRegistry:
    """Resolve and cache process-level adapters by provider configuration."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}
        self._instances: dict[tuple[str, str, str, int], ChatModelPort] = {}

    def register(self, provider: str, factory: ModelFactory) -> None:
        """Register one provider factory."""

        self._factories[provider] = factory

    def resolve(self, settings: Settings) -> ChatModelPort:
        """Return the configured adapter without leaking provider details upward."""

        key = (
            settings.model_provider,
            settings.model_name,
            settings.model_endpoint,
            settings.model_context_ceiling,
        )
        if key in self._instances:
            return self._instances[key]

        factory = self._factories.get(settings.model_provider)
        if factory is None:
            raise ModelAdapterUnavailableError("Configured local model provider is unavailable.")

        adapter = factory(settings)
        self._instances[key] = adapter
        return adapter


def create_default_registry() -> ModelAdapterRegistry:
    """Create the registry; real local-engine factories arrive in the next work item."""

    registry = ModelAdapterRegistry()
    registry.register(
        "fake",
        lambda settings: FakeModelAdapter(
            model_name=settings.model_name,
            max_context=settings.model_context_ceiling,
        ),
    )
    return registry
