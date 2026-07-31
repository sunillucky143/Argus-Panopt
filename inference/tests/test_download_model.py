from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any

from inference.download_model import (
    DownloadError,
    ManifestError,
    ModelManifest,
    _validate_redirect_url,
    download_model,
    load_manifest,
    verify_artifact,
)

_ARTIFACT = b"verified local model fixture"
_REPOSITORY_MANIFEST = (
    Path(__file__).parents[1] / "manifests" / "tier-s-gemma-3-4b-it-q4_k_m.json"
)


def _manifest_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": "fixture-model",
        "tier": "S",
        "engine": "llama.cpp",
        "repository": "example/fixture-GGUF",
        "revision": "a" * 40,
        "filename": "fixture-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "license": "gemma",
        "license_url": "https://ai.google.dev/gemma/terms",
        "source_url": (
            "https://huggingface.co/example/fixture-GGUF/resolve/"
            f"{'a' * 40}/fixture-Q4_K_M.gguf?download=true"
        ),
        "size_bytes": len(_ARTIFACT),
        "sha256": hashlib.sha256(_ARTIFACT).hexdigest(),
    }


def _write_manifest(directory: Path, data: dict[str, Any]) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ManifestTests(unittest.TestCase):
    def test_repository_manifest_is_valid(self) -> None:
        manifest = load_manifest(_REPOSITORY_MANIFEST)

        self.assertEqual(manifest.revision, "d0976223747697cb51e056d85c532013931fe52e")
        self.assertEqual(manifest.size_bytes, 2_489_757_856)
        self.assertEqual(
            manifest.sha256,
            "882e8d2db44dc554fb0ea5077cb7e4bc49e7342a1f0da57901c0802ea21a0863",
        )

    def test_rejects_missing_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            del data["sha256"]
            data["extra"] = True

            with self.assertRaisesRegex(ManifestError, "missing: sha256"):
                load_manifest(_write_manifest(root, data))

    def test_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
            )

            with self.assertRaisesRegex(ManifestError, "duplicate manifest key"):
                load_manifest(path)

    def test_rejects_path_traversal_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            data["filename"] = "../fixture.gguf"

            with self.assertRaisesRegex(ManifestError, "filename"):
                load_manifest(_write_manifest(root, data))

    def test_rejects_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            data["revision"] = "main"

            with self.assertRaisesRegex(ManifestError, "commit SHA"):
                load_manifest(_write_manifest(root, data))

    def test_rejects_source_url_that_does_not_match_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            data["source_url"] = (
                f"https://huggingface.co/example/fixture-GGUF/resolve/{'a' * 40}/other.gguf"
            )

            with self.assertRaisesRegex(ManifestError, "pinned repository artifact"):
                load_manifest(_write_manifest(root, data))

    def test_rejects_unapproved_source_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            data["source_url"] = data["source_url"].replace(
                "huggingface.co", "models.example.com"
            )

            with self.assertRaisesRegex(ManifestError, "approved HTTPS model host"):
                load_manifest(_write_manifest(root, data))

    def test_rejects_malformed_source_port_as_manifest_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _manifest_data()
            data["source_url"] = data["source_url"].replace(
                "huggingface.co", "huggingface.co:not-a-port"
            )

            with self.assertRaisesRegex(ManifestError, "approved HTTPS model host"):
                load_manifest(_write_manifest(root, data))


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        data = _manifest_data()
        self.manifest = ModelManifest(**data)

    @staticmethod
    def _opener(content: bytes) -> Any:
        def open_fixture(_request: object, _timeout: float) -> BytesIO:
            return BytesIO(content)

        return open_fixture

    def test_downloads_and_atomically_installs_verified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            installed = download_model(
                self.manifest,
                output,
                opener=self._opener(_ARTIFACT),
            )

            self.assertEqual(installed.read_bytes(), _ARTIFACT)
            self.assertTrue(verify_artifact(installed, self.manifest))
            self.assertEqual(list(output.glob("*.part")), [])

    def test_redirect_policy_allows_only_approved_https_artifact_hosts(self) -> None:
        _validate_redirect_url(
            "https://cas-bridge.xethub.hf.co/signed-artifact?token=test"
        )
        _validate_redirect_url("https://cdn-lfs.huggingface.co/model")

        for url in (
            "http://cas-bridge.xethub.hf.co/artifact",
            "https://example.com/artifact",
            "https://huggingface.co.evil.example/artifact",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                DownloadError, "unapproved host"
            ):
                _validate_redirect_url(url)

    def test_checksum_mismatch_does_not_replace_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / self.manifest.filename
            destination.write_bytes(b"existing invalid artifact")

            with self.assertRaisesRegex(DownloadError, "checksum"):
                download_model(
                    self.manifest,
                    output,
                    opener=self._opener(b"x" * len(_ARTIFACT)),
                )

            self.assertEqual(destination.read_bytes(), b"existing invalid artifact")
            self.assertEqual(list(output.glob("*.part")), [])

    def test_size_mismatch_removes_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            with self.assertRaisesRegex(DownloadError, "size"):
                download_model(
                    self.manifest,
                    output,
                    opener=self._opener(_ARTIFACT[:-1]),
                )

            self.assertFalse((output / self.manifest.filename).exists())
            self.assertEqual(list(output.glob("*.part")), [])

    def test_rejects_non_finite_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for timeout in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(timeout=timeout), self.assertRaisesRegex(
                    DownloadError, "finite and positive"
                ):
                    download_model(
                        self.manifest,
                        Path(temporary),
                        timeout=timeout,
                        opener=self._opener(_ARTIFACT),
                    )

    def test_existing_verified_artifact_skips_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / self.manifest.filename
            destination.write_bytes(_ARTIFACT)

            def fail_if_called(_request: object, _timeout: float) -> Any:
                raise AssertionError("network opener must not be called")

            installed = download_model(self.manifest, output, opener=fail_if_called)

            self.assertEqual(installed, destination)

    def test_refuses_symbolic_link_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            target = output / "target.gguf"
            target.write_bytes(_ARTIFACT)
            destination = output / self.manifest.filename
            try:
                destination.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links are unavailable in this environment")

            with self.assertRaisesRegex(DownloadError, "symbolic link"):
                download_model(
                    self.manifest,
                    output,
                    opener=self._opener(_ARTIFACT),
                )


if __name__ == "__main__":
    unittest.main()
