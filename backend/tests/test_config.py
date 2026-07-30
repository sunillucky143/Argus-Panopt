from app.core.config import Settings


def test_origins_are_normalized() -> None:
    settings = Settings(cors_origins=" https://one.test,https://two.test, ")

    assert settings.allowed_origins == ("https://one.test", "https://two.test")
