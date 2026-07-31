"""Validate the rendered Compose policy for the Tier S model runtime."""

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
_PINNED_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-b9445@"
    "sha256:8dd148c53936b6e8b0e75309841e66eab13adc50b004a5e86ab1fec477c17d8e"
)
_COMPOSE_OVERRIDE_KEYS = (
    "ARGUS_MODEL_PROVIDER",
    "ARGUS_MODEL_NAME",
    "ARGUS_MODEL_ENDPOINT",
    "ARGUS_MODEL_CONTEXT_CEILING",
    "ARGUS_LLAMA_CPUS",
    "ARGUS_LLAMA_MEMORY",
    "ARGUS_LLAMA_THREADS",
    "ARGUS_LLAMA_PARALLEL",
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
    result = subprocess.run(  # noqa: S603 - fixed executable and argument list
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


def _validate_cpu_profile(config: Mapping[str, Any]) -> None:
    services = _mapping(config.get("services"), "services")
    runtime = _mapping(services.get("inference-cpu"), "inference-cpu")
    api = _mapping(services.get("api"), "api")

    _require(
        runtime.get("image") == _PINNED_IMAGE, "llama.cpp image must be digest-pinned"
    )
    _require(
        runtime.get("profiles") == ["cpu"], "inference-cpu must be CPU-profile only"
    )
    _require(runtime.get("user") == "10001:10001", "inference-cpu must run non-root")
    _require(
        runtime.get("read_only") is True,
        "inference-cpu root filesystem must be read-only",
    )
    _require(
        runtime.get("cap_drop") == ["ALL"], "inference-cpu must drop all capabilities"
    )
    _require(
        "no-new-privileges:true" in (runtime.get("security_opt") or []),
        "inference-cpu must set no-new-privileges",
    )
    _require(
        set(_mapping(runtime.get("networks"), "runtime networks")) == {"processing"},
        "inference-cpu must attach only to the processing network",
    )
    _require(not runtime.get("ports"), "inference-cpu must not publish host ports")
    _require(runtime.get("pids_limit") == 512, "inference-cpu must enforce a PID limit")
    _require(
        float(runtime.get("cpus", 0)) <= 6.0, "inference-cpu CPU limit exceeds Tier S"
    )
    _require(
        int(runtime.get("mem_limit", 0)) <= 8 * 1024**3,
        "inference-cpu memory limit exceeds Tier S",
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
    expected_source = (_ROOT / "models").resolve()
    _require(
        Path(str(volume.get("source"))).resolve() == expected_source,
        "model mount source must be the verified artifact",
    )
    _require(
        volume.get("target") == "/models",
        "model directory must be mounted at /models",
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

    networks = _mapping(config.get("networks"), "networks")
    processing = _mapping(networks.get("processing"), "processing network")
    _require(processing.get("internal") is True, "processing network must block egress")


def _validate_smoke_profile(config: Mapping[str, Any]) -> None:
    services = _mapping(config.get("services"), "smoke services")
    _require(
        "inference-cpu" not in services, "smoke profile must not start the real model"
    )
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
