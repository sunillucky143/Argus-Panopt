from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.request import Request

from inference.download_bundle import (
    BundleManifestError,
    ModelBundleManifest,
    download_bundle,
    load_bundle_manifest,
    main,
    verify_bundle,
)
from inference.download_model import DownloadError, DownloadResponse

_ARTIFACT = b"verified local bundle fixture"
_REVISION = "a" * 40
_MANIFEST_ROOT = Path(__file__).parents[1] / "manifests"


def _manifest_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_id": "bge-m3-test-a1b2c3d",
        "model_id": "bge-m3",
        "engine": "text-embeddings-inference",
        "repository": "BAAI/bge-m3",
        "revision": _REVISION,
        "format": "onnx-fp32",
        "license": "MIT",
        "license_url": (
            f"https://huggingface.co/BAAI/bge-m3/blob/{_REVISION}/README.md"
        ),
        "artifacts": [
            {
                "source_path": "onnx/config.json",
                "filename": "config.json",
                "size_bytes": len(_ARTIFACT),
                "sha256": hashlib.sha256(_ARTIFACT).hexdigest(),
            }
        ],
    }


def _response(data: bytes) -> DownloadResponse:
    return cast(DownloadResponse, BytesIO(data))


def _write_manifest(directory: Path, data: dict[str, Any]) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class BundleManifestTests(unittest.TestCase):
    def test_repository_manifests_are_valid_and_pinned(self) -> None:
        embedding = load_bundle_manifest(_MANIFEST_ROOT / "bge-m3-onnx-fp32.json")
        reranker = load_bundle_manifest(
            _MANIFEST_ROOT / "bge-reranker-v2-m3-safetensors-fp32.json"
        )

        self.assertEqual(embedding.revision, "5617a9f61b028005a4858fdac845db406aefb181")
        self.assertEqual(len(embedding.artifacts), 8)
        self.assertEqual(embedding.artifacts[3].size_bytes, 2_266_820_608)
        self.assertEqual(
            embedding.artifacts[3].sha256,
            "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
        )
        self.assertEqual(reranker.revision, "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
        self.assertEqual(len(reranker.artifacts), 6)
        self.assertEqual(reranker.artifacts[1].size_bytes, 2_271_071_852)

    def test_rejects_duplicate_manifest_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
            )

            with self.assertRaisesRegex(BundleManifestError, "duplicate manifest key"):
                load_bundle_manifest(path)

    def test_rejects_manifest_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _manifest_data()
            data["unexpected"] = True

            with self.assertRaisesRegex(BundleManifestError, "keys do not match"):
                load_bundle_manifest(_write_manifest(Path(temporary), data))

    def test_rejects_unapproved_or_mutable_model_configuration(self) -> None:
        cases = (
            ("repository", "someone/bge-m3", "not approved"),
            ("engine", "remote-api", "not approved"),
            ("revision", "main", "immutable commit SHA"),
            ("format", "pickle", "not approved"),
            ("license_url", "https://example.com/license", "pinned model card"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                data = _manifest_data()
                data[key] = value
                with self.assertRaisesRegex(BundleManifestError, message):
                    load_bundle_manifest(_write_manifest(Path(temporary), data))

    def test_rejects_unsafe_and_duplicate_artifact_paths(self) -> None:
        cases = (
            ("source_path", "../config.json", "source_path"),
            ("source_path", "onnx//config.json", "source_path"),
            ("source_path", "onnx/./config.json", "source_path"),
            ("filename", "../config.json", "filename"),
            ("sha256", "not-a-digest", "sha256"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                data = _manifest_data()
                data["artifacts"][0][key] = value
                with self.assertRaisesRegex(BundleManifestError, message):
                    load_bundle_manifest(_write_manifest(Path(temporary), data))

        with tempfile.TemporaryDirectory() as temporary:
            data = _manifest_data()
            data["artifacts"].append(dict(data["artifacts"][0]))
            with self.assertRaisesRegex(
                BundleManifestError, "source paths must be unique"
            ):
                load_bundle_manifest(_write_manifest(Path(temporary), data))


class BundleProvisioningTests(unittest.TestCase):
    def _load_fixture(self, root: Path) -> ModelBundleManifest:
        return load_bundle_manifest(_write_manifest(root, _manifest_data()))

    def test_downloads_from_pinned_url_and_installs_verified_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            requests: list[tuple[str, float]] = []

            def opener(request: Request, timeout: float) -> DownloadResponse:
                requests.append((request.full_url, timeout))
                return _response(_ARTIFACT)

            installed = download_bundle(manifest, root / "models", opener=opener)

            self.assertTrue(verify_bundle(installed, manifest))
            self.assertEqual((installed / "config.json").read_bytes(), _ARTIFACT)
            self.assertEqual(
                requests,
                [
                    (
                        "https://huggingface.co/BAAI/bge-m3/resolve/"
                        f"{_REVISION}/onnx/config.json?download=true",
                        120.0,
                    )
                ],
            )

    def test_existing_verified_bundle_skips_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            models = root / "models"

            def first_opener(_: Request, __: float) -> DownloadResponse:
                return _response(_ARTIFACT)

            installed = download_bundle(manifest, models, opener=first_opener)

            def forbidden_opener(_: Request, __: float) -> DownloadResponse:
                raise AssertionError("network must not be used")

            self.assertEqual(
                download_bundle(manifest, models, opener=forbidden_opener), installed
            )

    def test_checksum_failure_preserves_no_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            models = root / "models"

            def opener(_: Request, __: float) -> DownloadResponse:
                return _response(b"x" * len(_ARTIFACT))

            with self.assertRaisesRegex(DownloadError, "checksum"):
                download_bundle(manifest, models, opener=opener)

            self.assertFalse((models / manifest.bundle_id).exists())
            self.assertEqual(list(models.iterdir()), [])

    def test_invalid_existing_bundle_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            destination = root / "models" / manifest.bundle_id
            destination.mkdir(parents=True)
            (destination / "config.json").write_bytes(b"preserve me")

            with self.assertRaisesRegex(DownloadError, "incomplete or invalid"):
                download_bundle(manifest, root / "models")

            self.assertEqual((destination / "config.json").read_bytes(), b"preserve me")

    def test_verification_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            bundle = root / manifest.bundle_id
            bundle.mkdir()
            (bundle / "config.json").write_bytes(_ARTIFACT)
            (bundle / "unexpected.py").write_text("pass", encoding="utf-8")

            self.assertFalse(verify_bundle(bundle, manifest))

    def test_rejects_invalid_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._load_fixture(root)
            with self.assertRaisesRegex(DownloadError, "finite and positive"):
                download_bundle(manifest, root / "models", timeout=float("nan"))

    def test_verify_only_cli_succeeds_for_installed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_manifest(root, _manifest_data())
            manifest = load_bundle_manifest(manifest_path)
            bundle = root / "models" / manifest.bundle_id
            bundle.mkdir(parents=True)
            (bundle / "config.json").write_bytes(_ARTIFACT)

            self.assertEqual(
                main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(root / "models"),
                        "--verify-only",
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
