"""Validate rendered Compose policy for local model runtimes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE_FILE = _ROOT / "compose.yaml"
_ENV_FILE = _ROOT / "deploy" / ".env.example"
_MODEL_FILE = "gemma-3-4b-it-Q4_K_M.gguf"
_PINNED_LLAMA_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-b9445@"
    "sha256:8dd148c53936b6e8b0e75309841e66eab13adc50b004a5e86ab1fec477c17d8e"
)
_PINNED_TEI_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.1@"
    "sha256:b7772cdd9dcbced147b16a7dff17d4aed1ab36333f8d3e686c50d2175e1d2126"
)
_RETRIEVAL_WORKERS = {
    "embedding-engine": "bge-m3-onnx-fp32-5617a9f",
    "reranker-engine": "bge-reranker-v2-m3-safetensors-fp32-953dc6f",
}
_COMPOSE_OVERRIDE_KEYS = (
    "ARGUS_MODEL_PROVIDER",
    "ARGUS_MODEL_NAME",
    "ARGUS_MODEL_ENDPOINT",
    "ARGUS_MODEL_CONTEXT_CEILING",
    "ARGUS_LLAMA_CPUS",
    "ARGUS_LLAMA_MEMORY",
    "ARGUS_LLAMA_THREADS",
    "ARGUS_LLAMA_PARALLEL",
    "ARGUS_EMBEDDING_MODEL",
    "ARGUS_RERANKER_MODEL",
    "ARGUS_RETRIEVAL_TIMEOUT_SECONDS",
    "ARGUS_RETRIEVAL_GATEWAY_CPUS",
    "ARGUS_RETRIEVAL_GATEWAY_MEMORY",
    "ARGUS_EMBEDDING_CPUS",
    "ARGUS_EMBEDDING_MEMORY",
    "ARGUS_EMBEDDING_THREADS",
    "ARGUS_RERANKER_CPUS",
    "ARGUS_RERANKER_MEMORY",
    "ARGUS_RERANKER_THREADS",
)


class PolicyError(RuntimeError):
    """The rendered Compose model runtime violates a required control."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise PolicyError(f"{name} must be a list")
    return value


def _render(
    profile: str, overrides: Mapping[str, str] | None = None
) -> Mapping[str, Any]:
    environment = os.environ.copy()
    for key in _COMPOSE_OVERRIDE_KEYS:
        environment.pop(key, None)
    environment.update(overrides or {})
    docker = shutil.which("docker")
    if docker is None:
        raise PolicyError("docker executable is unavailable")
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            str(_ENV_FILE),
            "--file",
            str(_COMPOSE_FILE),
            "--profile",
            profile,
            "config",
            "--format",
            "json",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PolicyError(f"unable to render the {profile} Compose profile")
    try:
        return _mapping(json.loads(result.stdout), "Compose config")
    except json.JSONDecodeError:
        raise PolicyError("Compose returned invalid JSON") from None


def _validate_hardening(
    service: Mapping[str, Any],
    name: str,
    *,
    pids_limit: int,
    max_cpus: float,
    max_memory: int,
) -> None:
    _require(service.get("user") == "10001:10001", f"{name} must run non-root")
    _require(
        service.get("read_only") is True, f"{name} root filesystem must be read-only"
    )
    _require(service.get("cap_drop") == ["ALL"], f"{name} must drop all capabilities")
    _require(
        "no-new-privileges:true" in (service.get("security_opt") or []),
        f"{name} must set no-new-privileges",
    )
    _require(
        set(_mapping(service.get("networks"), f"{name} networks")) == {"processing"},
        f"{name} must attach only to the processing network",
    )
    _require(not service.get("ports"), f"{name} must not publish host ports")
    _require(
        service.get("pids_limit") == pids_limit, f"{name} must enforce its PID limit"
    )
    cpus = float(service.get("cpus", 0))
    memory = int(service.get("mem_limit", 0))
    _require(0 < cpus <= max_cpus, f"{name} CPU limit exceeds policy")
    _require(0 < memory <= max_memory, f"{name} memory limit exceeds policy")
    tmpfs = _list(service.get("tmpfs"), f"{name} tmpfs")
    _require(
        any(str(mount).startswith("/tmp:") for mount in tmpfs),
        f"{name} must use a bounded /tmp tmpfs",
    )


def _validate_llama(services: Mapping[str, Any]) -> None:
    runtime = _mapping(services.get("inference-cpu"), "inference-cpu")
    api = _mapping(services.get("api"), "api")

    _require(
        runtime.get("image") == _PINNED_LLAMA_IMAGE,
        "llama.cpp image must be digest-pinned",
    )
    _require(
        runtime.get("profiles") == ["cpu"], "inference-cpu must be CPU-profile only"
    )
    _validate_hardening(
        runtime,
        "inference-cpu",
        pids_limit=512,
        max_cpus=6.0,
        max_memory=8 * 1024**3,
    )

    command = _list(runtime.get("command"), "inference-cpu command")
    command_text = " ".join(str(item) for item in command)
    _require(
        f"/models/{_MODEL_FILE}" in command_text,
        "runtime must use the pinned local artifact",
    )
    for forbidden in (
        "http://",
        "https://",
        "--model-url",
        "--hf-repo",
        "--docker-repo",
    ):
        _require(
            forbidden not in command_text,
            f"runtime command contains forbidden source: {forbidden}",
        )

    healthcheck = _mapping(runtime.get("healthcheck"), "runtime healthcheck")
    healthcheck_test = _list(healthcheck.get("test"), "runtime healthcheck test")
    _require(
        "/health" in " ".join(str(item) for item in healthcheck_test),
        "inference-cpu must probe llama.cpp readiness",
    )

    volumes = _list(runtime.get("volumes"), "inference-cpu volumes")
    _require(len(volumes) == 1, "inference-cpu must have exactly one model mount")
    volume = _mapping(volumes[0], "model mount")
    _require(
        Path(str(volume.get("source"))).resolve() == (_ROOT / "models").resolve(),
        "model mount source must be the verified artifact",
    )
    _require(
        volume.get("target") == "/models", "model directory must be mounted at /models"
    )
    _require(
        volume.get("read_only") is True, "model artifact must be mounted read-only"
    )

    api_environment = _mapping(api.get("environment"), "API environment")
    _require(
        api_environment.get("ARGUS_MODEL_PROVIDER") == "llama_cpp",
        "CPU profile must select the llama.cpp adapter",
    )
    _require(
        api_environment.get("ARGUS_MODEL_NAME") == "gemma-3-4b-it",
        "CPU profile must report the pinned model name",
    )
    _require(
        api_environment.get("ARGUS_MODEL_ENDPOINT") == "http://inference-cpu:8080/v1",
        "CPU adapter endpoint must remain internal",
    )


def _validate_retrieval_gateway(services: Mapping[str, Any]) -> None:
    gateway = _mapping(services.get("embedding-service"), "embedding-service")
    _require(
        gateway.get("profiles") == ["cpu", "gpu"],
        "retrieval gateway must be enabled only for real model profiles",
    )
    _validate_hardening(
        gateway,
        "embedding-service",
        pids_limit=64,
        max_cpus=0.5,
        max_memory=256 * 1024**2,
    )

    environment = _mapping(gateway.get("environment"), "retrieval gateway environment")
    expected = {
        "ARGUS_RETRIEVAL_EMBEDDING_ENGINE_ENDPOINT": "http://embedding-engine:80",
        "ARGUS_RETRIEVAL_RERANKER_ENGINE_ENDPOINT": "http://reranker-engine:80",
        "ARGUS_RETRIEVAL_EMBEDDING_MODEL": "bge-m3",
        "ARGUS_RETRIEVAL_RERANKER_MODEL": "bge-reranker-v2-m3",
        "ARGUS_RETRIEVAL_TIMEOUT_SECONDS": "120",
    }
    for key, value in expected.items():
        _require(environment.get(key) == value, f"retrieval gateway must pin {key}")

    dependencies = _mapping(gateway.get("depends_on"), "retrieval gateway dependencies")
    _require(
        set(dependencies) == set(_RETRIEVAL_WORKERS),
        "retrieval gateway must wait only for both local workers",
    )


def _validate_retrieval_worker(
    services: Mapping[str, Any],
    name: str,
    bundle_id: str,
) -> None:
    worker = _mapping(services.get(name), name)
    _require(
        worker.get("profiles") == ["cpu", "gpu"], f"{name} profile policy is invalid"
    )
    _require(
        worker.get("platform") == "linux/amd64",
        f"{name} must pin its supported platform",
    )
    _require(
        worker.get("image") == _PINNED_TEI_IMAGE, f"{name} image must be digest-pinned"
    )
    _validate_hardening(
        worker,
        name,
        pids_limit=256,
        max_cpus=2.0,
        max_memory=4 * 1024**3,
    )

    command = _list(worker.get("command"), f"{name} command")
    command_text = " ".join(str(item) for item in command)
    _require(
        "--model-id /models/model" in command_text, f"{name} must load its local bundle"
    )
    _require("--port 80" in command_text, f"{name} must use its internal port")
    for forbidden in ("http://", "https://", "--revision", "--token"):
        _require(
            forbidden not in command_text,
            f"{name} command contains forbidden source: {forbidden}",
        )
    if name == "embedding-engine":
        _require("--pooling cls" in command_text, "BGE-M3 must use CLS pooling")
    else:
        _require(
            "--pooling" not in command_text, "reranker must use sequence classification"
        )

    environment = _mapping(worker.get("environment"), f"{name} environment")
    _require(
        environment.get("HF_HUB_OFFLINE") == "1", f"{name} must force offline mode"
    )
    _require(
        environment.get("HF_HUB_DISABLE_TELEMETRY") == "1",
        f"{name} must disable telemetry",
    )
    _require(
        environment.get("HF_HOME") == "/tmp/huggingface",
        f"{name} cache must use bounded tmpfs",
    )

    volumes = _list(worker.get("volumes"), f"{name} volumes")
    _require(len(volumes) == 1, f"{name} must have exactly one model mount")
    volume = _mapping(volumes[0], f"{name} model mount")
    _require(
        Path(str(volume.get("source"))).resolve()
        == (_ROOT / "models" / bundle_id).resolve(),
        f"{name} must mount its verified bundle directory",
    )
    _require(
        volume.get("target") == "/models/model", f"{name} model mount target is invalid"
    )
    _require(volume.get("read_only") is True, f"{name} model mount must be read-only")

    healthcheck = _mapping(worker.get("healthcheck"), f"{name} healthcheck")
    healthcheck_test = _list(healthcheck.get("test"), f"{name} healthcheck test")
    _require(
        "http://127.0.0.1:80/health"
        in " ".join(str(item) for item in healthcheck_test),
        f"{name} must probe local readiness",
    )


def _validate_cpu_profile(config: Mapping[str, Any]) -> None:
    services = _mapping(config.get("services"), "services")
    _validate_llama(services)
    _validate_retrieval_gateway(services)
    for name, bundle_id in _RETRIEVAL_WORKERS.items():
        _validate_retrieval_worker(services, name, bundle_id)

    api = _mapping(services.get("api"), "api")
    api_environment = _mapping(api.get("environment"), "API environment")
    _require(
        api_environment.get("ARGUS_EMBEDDING_ENDPOINT")
        == "http://embedding-service:8081",
        "retrieval adapter endpoint must remain internal",
    )
    _require(
        api_environment.get("ARGUS_EMBEDDING_MODEL") == "bge-m3",
        "embedding model name must match the pinned runtime",
    )
    _require(
        api_environment.get("ARGUS_RERANKER_MODEL") == "bge-reranker-v2-m3",
        "reranker model name must match the pinned runtime",
    )

    networks = _mapping(config.get("networks"), "networks")
    processing = _mapping(networks.get("processing"), "processing network")
    _require(processing.get("internal") is True, "processing network must block egress")


def _validate_smoke_profile(config: Mapping[str, Any]) -> None:
    services = _mapping(config.get("services"), "smoke services")
    for name in ("inference-cpu", "embedding-service", *_RETRIEVAL_WORKERS):
        _require(name not in services, f"smoke profile must not start {name}")
    api = _mapping(services.get("api"), "smoke API")
    environment = _mapping(api.get("environment"), "smoke API environment")
    _require(
        environment.get("ARGUS_MODEL_PROVIDER") == "fake",
        "smoke profile must use the deterministic adapter",
    )


def main() -> int:
    try:
        _validate_cpu_profile(_render("cpu"))
        _validate_smoke_profile(
            _render(
                "smoke",
                {
                    "ARGUS_MODEL_PROVIDER": "fake",
                    "ARGUS_MODEL_NAME": "argus-smoke-model",
                },
            )
        )
    except PolicyError as error:
        print(f"Compose model runtime policy failed: {error}", file=sys.stderr)
        return 1

    print("Compose model runtime policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
