"""Provision a pinned local model artifact after strict integrity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf")
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_EXPECTED_KEYS = {
    "schema_version",
    "model_id",
    "tier",
    "engine",
    "repository",
    "revision",
    "filename",
    "quantization",
    "license",
    "license_url",
    "source_url",
    "size_bytes",
    "sha256",
}


class ManifestError(ValueError):
    """The model manifest is malformed or violates the provisioning policy."""


class DownloadError(RuntimeError):
    """The model artifact could not be installed safely."""


class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class Writable(Protocol):
    def write(self, data: bytes) -> int: ...


class DownloadResponse(Readable, Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None: ...


OpenUrl = Callable[[Request, float], DownloadResponse]


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    model_id: str
    tier: str
    engine: str
    repository: str
    revision: str
    filename: str
    quantization: str
    license: str
    license_url: str
    source_url: str
    size_bytes: int
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _validate_source_url(manifest: ModelManifest) -> None:
    try:
        parsed = urlsplit(manifest.source_url)
        port = parsed.port
    except ValueError:
        raise ManifestError(
            "source_url must use the approved HTTPS model host"
        ) from None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise ManifestError("source_url must use the approved HTTPS model host")

    expected_path = (
        f"/{manifest.repository}/resolve/{manifest.revision}/{manifest.filename}"
    )
    if unquote(parsed.path) != expected_path:
        raise ManifestError("source_url must match the pinned repository artifact")
    if parsed.query not in ("", "download=true"):
        raise ManifestError("source_url contains unsupported query parameters")


def _validate_manifest(data: Mapping[str, Any]) -> ModelManifest:
    if set(data) != _EXPECTED_KEYS:
        missing = sorted(_EXPECTED_KEYS - set(data))
        unexpected = sorted(set(data) - _EXPECTED_KEYS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ManifestError(f"manifest keys do not match schema ({'; '.join(details)})")

    schema_version = data.get("schema_version")
    size_bytes = data.get("size_bytes")
    if type(schema_version) is not int or schema_version != 1:
        raise ManifestError("schema_version must be 1")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ManifestError("size_bytes must be a positive integer")

    manifest = ModelManifest(
        schema_version=schema_version,
        model_id=_required_string(data, "model_id"),
        tier=_required_string(data, "tier"),
        engine=_required_string(data, "engine"),
        repository=_required_string(data, "repository"),
        revision=_required_string(data, "revision"),
        filename=_required_string(data, "filename"),
        quantization=_required_string(data, "quantization"),
        license=_required_string(data, "license"),
        license_url=_required_string(data, "license_url"),
        source_url=_required_string(data, "source_url"),
        size_bytes=size_bytes,
        sha256=_required_string(data, "sha256"),
    )

    if not _IDENTIFIER_PATTERN.fullmatch(manifest.model_id):
        raise ManifestError("model_id contains unsupported characters")
    if manifest.tier != "S" or manifest.engine != "llama.cpp":
        raise ManifestError(
            "this manifest schema supports only the Tier S llama.cpp model"
        )
    if not _REPOSITORY_PATTERN.fullmatch(manifest.repository):
        raise ManifestError("repository must be an owner/name identifier")
    if not _REVISION_PATTERN.fullmatch(manifest.revision):
        raise ManifestError("revision must be a lowercase 40-character commit SHA")
    if not _FILENAME_PATTERN.fullmatch(manifest.filename):
        raise ManifestError("filename must be a basename ending in .gguf")
    if not _SHA256_PATTERN.fullmatch(manifest.sha256):
        raise ManifestError("sha256 must be a lowercase hexadecimal digest")
    if manifest.quantization != "Q4_K_M":
        raise ManifestError("Tier S quantization must be Q4_K_M")

    try:
        license_url = urlsplit(manifest.license_url)
        license_port = license_url.port
    except ValueError:
        raise ManifestError(
            "license_url must use the official HTTPS license host"
        ) from None
    if (
        license_url.scheme != "https"
        or license_url.hostname != "ai.google.dev"
        or license_url.username is not None
        or license_url.password is not None
        or license_port not in (None, 443)
        or license_url.fragment
    ):
        raise ManifestError("license_url must use the official HTTPS license host")

    _validate_source_url(manifest)
    return manifest


def load_manifest(path: Path) -> ModelManifest:
    """Load and strictly validate a JSON model manifest."""

    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ManifestError("manifest exceeds the maximum supported size")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(
            f"unable to read manifest: {type(error).__name__}"
        ) from None

    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    return _validate_manifest(data)


def _copy_and_hash(
    source: Readable,
    destination: Writable,
    size_limit: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    copied = 0
    while chunk := source.read(_CHUNK_SIZE):
        copied += len(chunk)
        if copied > size_limit:
            break
        digest.update(chunk)
        destination.write(chunk)
    return copied, digest.hexdigest()


def verify_artifact(path: Path, manifest: ModelManifest) -> bool:
    """Return whether an artifact snapshot has the expected size and digest."""

    if not path.is_file() or path.is_symlink():
        return False
    try:
        with path.open("rb") as artifact:
            if not stat.S_ISREG(os.fstat(artifact.fileno()).st_mode):
                return False
            size, digest = _copy_and_hash(artifact, _NullWriter(), manifest.size_bytes)
        return size == manifest.size_bytes and digest == manifest.sha256
    except OSError:
        return False


class _NullWriter:
    def write(self, data: bytes) -> int:
        return len(data)


def _validate_redirect_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise DownloadError("download redirect targeted an unapproved host") from None
    hostname = parsed.hostname or ""
    approved_host = (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
        or hostname == "cdn-lfs.hf.co"
        or hostname.endswith(".xethub.hf.co")
    )
    if (
        parsed.scheme != "https"
        or not approved_host
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise DownloadError("download redirect targeted an unapproved host")


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_redirect_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(request: Request, timeout: float) -> DownloadResponse:
    return cast(
        DownloadResponse,
        build_opener(_SafeRedirectHandler()).open(request, timeout=timeout),
    )


def download_model(
    manifest: ModelManifest,
    output_dir: Path,
    *,
    timeout: float = 120.0,
    opener: OpenUrl = _open_url,
) -> Path:
    """Download, verify, and atomically install a model artifact."""

    if not math.isfinite(timeout) or timeout <= 0:
        raise DownloadError("timeout must be finite and positive")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_root = output_dir.resolve(strict=True)
    except OSError as error:
        raise DownloadError(
            f"unable to prepare output directory: {type(error).__name__}"
        ) from None
    if not output_root.is_dir():
        raise DownloadError("output path is not a directory")

    destination = output_root / manifest.filename
    if destination.is_symlink():
        raise DownloadError("refusing to replace a symbolic link")

    request = Request(  # noqa: S310 - URL and redirects are restricted above.
        manifest.source_url,
        headers={"User-Agent": "Argus-Panopt-model-provisioner/1"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_root,
            prefix=f".{manifest.filename}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            existing_verified = False
            if destination.exists():
                try:
                    with destination.open("rb") as existing:
                        if stat.S_ISREG(os.fstat(existing.fileno()).st_mode):
                            downloaded, digest = _copy_and_hash(
                                existing,
                                temporary,
                                manifest.size_bytes,
                            )
                            existing_verified = (
                                downloaded == manifest.size_bytes
                                and digest == manifest.sha256
                            )
                except OSError:
                    pass

            if not existing_verified:
                temporary.seek(0)
                temporary.truncate()
                with opener(request, timeout) as response:
                    downloaded, digest = _copy_and_hash(
                        response,
                        temporary,
                        manifest.size_bytes,
                    )
                    if downloaded > manifest.size_bytes:
                        raise DownloadError(
                            "downloaded artifact exceeds the expected size"
                        )
            temporary.flush()
            os.fsync(temporary.fileno())

        if downloaded != manifest.size_bytes:
            raise DownloadError("downloaded artifact size does not match the manifest")
        if digest != manifest.sha256:
            raise DownloadError(
                "downloaded artifact checksum does not match the manifest"
            )

        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError(f"download failed: {type(error).__name__}") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                raise DownloadError(
                    f"unable to remove partial artifact: {type(error).__name__}"
                ) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify a pinned Argus Panopt model artifact."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        installed_path = download_model(
            manifest,
            args.output_dir,
            timeout=args.timeout,
        )
    except (ManifestError, DownloadError) as error:
        print(f"model provisioning failed: {error}", file=sys.stderr)
        return 1

    print(f"verified model artifact: {installed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
