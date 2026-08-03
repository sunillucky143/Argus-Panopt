"""Provision an immutable multi-file model bundle with strict verification."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request

from inference.download_model import (
    DownloadError,
    OpenUrl,
    _copy_and_hash,
    _open_url,
)

_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_ARTIFACTS = 32
_MAX_ARTIFACT_BYTES = 5 * 1024**3
_MAX_BUNDLE_BYTES = 8 * 1024**3
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_EXPECTED_KEYS = {
    "schema_version",
    "bundle_id",
    "model_id",
    "engine",
    "repository",
    "revision",
    "format",
    "license",
    "license_url",
    "artifacts",
}
_ARTIFACT_KEYS = {"source_path", "filename", "size_bytes", "sha256"}
_APPROVED_MODELS = {
    "bge-m3": ("BAAI/bge-m3", "onnx-fp32", "MIT"),
    "bge-reranker-v2-m3": (
        "BAAI/bge-reranker-v2-m3",
        "safetensors-fp32",
        "Apache-2.0",
    ),
}


class BundleManifestError(ValueError):
    """The model bundle manifest is malformed or violates policy."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    source_path: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelBundleManifest:
    schema_version: int
    bundle_id: str
    model_id: str
    engine: str
    repository: str
    revision: str
    format: str
    license: str
    license_url: str
    artifacts: tuple[ArtifactSpec, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleManifestError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise BundleManifestError(f"{key} must be a non-empty string")
    return value


def _safe_source_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        value == path.as_posix()
        and not path.is_absolute()
        and len(path.parts) <= 4
        and all(part not in ("", ".", "..") for part in path.parts)
        and all(_FILENAME_PATTERN.fullmatch(part) for part in path.parts)
    )


def _parse_artifact(data: Any) -> ArtifactSpec:
    if not isinstance(data, dict) or set(data) != _ARTIFACT_KEYS:
        raise BundleManifestError("artifact keys do not match schema")
    size_bytes = data.get("size_bytes")
    if type(size_bytes) is not int or not 0 < size_bytes <= _MAX_ARTIFACT_BYTES:
        raise BundleManifestError("artifact size_bytes is invalid")
    artifact = ArtifactSpec(
        source_path=_required_string(data, "source_path"),
        filename=_required_string(data, "filename"),
        size_bytes=size_bytes,
        sha256=_required_string(data, "sha256"),
    )
    if not _safe_source_path(artifact.source_path):
        raise BundleManifestError("artifact source_path is invalid")
    if not _FILENAME_PATTERN.fullmatch(artifact.filename):
        raise BundleManifestError("artifact filename is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256):
        raise BundleManifestError("artifact sha256 is invalid")
    return artifact


def _validate_manifest(data: Mapping[str, Any]) -> ModelBundleManifest:
    if set(data) != _EXPECTED_KEYS:
        raise BundleManifestError("manifest keys do not match schema")
    if data.get("schema_version") != 1 or type(data.get("schema_version")) is not int:
        raise BundleManifestError("schema_version must be 1")
    raw_artifacts = data.get("artifacts")
    if (
        not isinstance(raw_artifacts, list)
        or not 0 < len(raw_artifacts) <= _MAX_ARTIFACTS
    ):
        raise BundleManifestError("artifacts must be a bounded non-empty list")

    artifacts = tuple(_parse_artifact(item) for item in raw_artifacts)
    manifest = ModelBundleManifest(
        schema_version=1,
        bundle_id=_required_string(data, "bundle_id"),
        model_id=_required_string(data, "model_id"),
        engine=_required_string(data, "engine"),
        repository=_required_string(data, "repository"),
        revision=_required_string(data, "revision"),
        format=_required_string(data, "format"),
        license=_required_string(data, "license"),
        license_url=_required_string(data, "license_url"),
        artifacts=artifacts,
    )
    if not _IDENTIFIER_PATTERN.fullmatch(manifest.bundle_id):
        raise BundleManifestError("bundle_id contains unsupported characters")
    approved = _APPROVED_MODELS.get(manifest.model_id)
    if approved != (manifest.repository, manifest.format, manifest.license):
        raise BundleManifestError("model bundle is not approved")
    if manifest.engine != "text-embeddings-inference":
        raise BundleManifestError("model bundle engine is not approved")
    if not _REVISION_PATTERN.fullmatch(manifest.revision):
        raise BundleManifestError("revision must be an immutable commit SHA")
    expected_license_url = (
        f"https://huggingface.co/{manifest.repository}/blob/"
        f"{manifest.revision}/README.md"
    )
    if manifest.license_url != expected_license_url:
        raise BundleManifestError("license_url must reference the pinned model card")
    if len({item.source_path for item in artifacts}) != len(artifacts):
        raise BundleManifestError("artifact source paths must be unique")
    if len({item.filename for item in artifacts}) != len(artifacts):
        raise BundleManifestError("artifact filenames must be unique")
    if sum(item.size_bytes for item in artifacts) > _MAX_BUNDLE_BYTES:
        raise BundleManifestError("model bundle exceeds the maximum supported size")
    return manifest


def load_bundle_manifest(path: Path) -> ModelBundleManifest:
    """Load and strictly validate a model bundle manifest."""

    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise BundleManifestError("manifest exceeds the maximum supported size")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except BundleManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleManifestError(
            f"unable to read manifest: {type(error).__name__}"
        ) from None
    if not isinstance(data, dict):
        raise BundleManifestError("manifest root must be a JSON object")
    return _validate_manifest(data)


def verify_bundle(path: Path, manifest: ModelBundleManifest) -> bool:
    """Return whether a local bundle exactly matches its manifest."""

    if not path.is_dir() or path.is_symlink():
        return False
    expected = {artifact.filename for artifact in manifest.artifacts}
    try:
        entries = list(path.iterdir())
        if {entry.name for entry in entries} != expected:
            return False
        for artifact in manifest.artifacts:
            candidate = path / artifact.filename
            if candidate.is_symlink() or not candidate.is_file():
                return False
            with candidate.open("rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    return False
                size, digest = _copy_and_hash(
                    source, _NullWriter(), artifact.size_bytes
                )
            if size != artifact.size_bytes or digest != artifact.sha256:
                return False
        return True
    except OSError:
        return False


class _NullWriter:
    def write(self, data: bytes) -> int:
        return len(data)


def _artifact_url(manifest: ModelBundleManifest, artifact: ArtifactSpec) -> str:
    path = quote(artifact.source_path, safe="/")
    return (
        f"https://huggingface.co/{manifest.repository}/resolve/"
        f"{manifest.revision}/{path}?download=true"
    )


def download_bundle(
    manifest: ModelBundleManifest,
    output_dir: Path,
    *,
    timeout: float = 120.0,
    opener: OpenUrl = _open_url,
) -> Path:
    """Download, verify, and atomically install a model bundle."""

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

    destination = output_root / manifest.bundle_id
    if destination.is_symlink():
        raise DownloadError("refusing to replace a symbolic link")
    if destination.exists():
        if verify_bundle(destination, manifest):
            return destination
        raise DownloadError("existing model bundle is incomplete or invalid")

    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{manifest.bundle_id}.", dir=output_root)
    )
    cleanup_path: Path | None = temporary_path
    try:
        for artifact in manifest.artifacts:
            request = Request(  # noqa: S310 - repository and revision are restricted.
                _artifact_url(manifest, artifact),
                headers={"User-Agent": "Argus-Panopt-model-provisioner/1"},
            )
            target = temporary_path / artifact.filename
            with target.open("xb") as output, opener(request, timeout) as response:
                size, digest = _copy_and_hash(response, output, artifact.size_bytes)
                output.flush()
                os.fsync(output.fileno())
            if size != artifact.size_bytes:
                raise DownloadError(
                    "downloaded artifact size does not match the manifest"
                )
            if digest != artifact.sha256:
                raise DownloadError(
                    "downloaded artifact checksum does not match the manifest"
                )
            os.chmod(target, 0o644)
        if not verify_bundle(temporary_path, manifest):
            raise DownloadError("downloaded model bundle failed final verification")
        os.replace(temporary_path, destination)
        cleanup_path = None
        return destination
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError(f"bundle download failed: {type(error).__name__}") from None
    finally:
        if cleanup_path is not None:
            try:
                shutil.rmtree(cleanup_path)
            except OSError as error:
                raise DownloadError(
                    f"unable to remove partial bundle: {type(error).__name__}"
                ) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify a pinned Argus Panopt model bundle."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_bundle_manifest(args.manifest)
        installed_path = args.output_dir.resolve() / manifest.bundle_id
        if args.verify_only:
            if not verify_bundle(installed_path, manifest):
                raise DownloadError("installed model bundle is missing or invalid")
        else:
            installed_path = download_bundle(
                manifest, args.output_dir, timeout=args.timeout
            )
    except (BundleManifestError, DownloadError) as error:
        print(f"model bundle provisioning failed: {error}", file=sys.stderr)
        return 1
    print(f"verified model bundle: {installed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
