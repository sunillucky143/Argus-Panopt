"""Validation helpers for deployment-local service endpoints."""

import re
from ipaddress import ip_address
from urllib.parse import urlsplit


def validate_internal_service_endpoint(value: str, *, label: str) -> str:
    """Return a normalized endpoint only when it cannot name a public host."""

    endpoint = urlsplit(value)
    try:
        _ = endpoint.port
    except ValueError:
        raise ValueError(f"{label} port is invalid") from None

    hostname = endpoint.hostname
    if (
        endpoint.scheme not in {"http", "https"}
        or hostname is None
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or endpoint.netloc.endswith(":")
        or any(segment in {".", ".."} for segment in endpoint.path.split("/"))
    ):
        raise ValueError(f"{label} must be a simple internal HTTP(S) URL")

    try:
        address = ip_address(hostname)
    except ValueError:
        is_internal_name = hostname == "localhost" or (
            bool(
                re.fullmatch(
                    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                    hostname,
                    flags=re.IGNORECASE,
                )
            )
            and not hostname.startswith("-")
            and not hostname.endswith("-")
        )
        if not is_internal_name:
            raise ValueError(f"{label} hostname must be local or private") from None
    else:
        if not (address.is_loopback or address.is_private or address.is_link_local):
            raise ValueError(f"{label} hostname must be local or private")

    return value.rstrip("/")
