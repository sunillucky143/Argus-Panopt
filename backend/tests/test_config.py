import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_origins_are_normalized() -> None:
    settings = Settings(cors_origins=" https://one.test,https://two.test, ")

    assert settings.allowed_origins == ("https://one.test", "https://two.test")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://inference-cpu:8080/v1",
        "http://localhost:8080/v1",
        "http://10.20.30.40:8080/v1",
        f"http://{'a' * 63}:8080/v1",
    ],
)
def test_local_model_endpoints_are_accepted(endpoint: str) -> None:
    assert Settings(model_endpoint=endpoint).model_endpoint == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.vendor.example/v1",
        "https://example.com/v1",
        "http://8.8.8.8/v1",
        "ftp://inference-cpu/v1",
        "http://user:secret@inference-cpu/v1",
        f"http://{'a' * 64}:8080/v1",
    ],
)
def test_non_local_model_endpoints_are_rejected(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="model endpoint"):
        Settings(model_endpoint=endpoint)
