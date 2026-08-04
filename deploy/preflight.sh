#!/usr/bin/env sh
set -eu

profile="${1:-cpu}"
required_disk_kb=10485760
required_disk_gb=10
web_port="${ARGUS_WEB_PORT:-8080}"
environment="${ARGUS_ENVIRONMENT:-}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

info() {
  printf 'OK: %s\n' "$1"
}

read_setting() {
  setting_name="$1"
  setting_value="$(printenv "$setting_name" 2>/dev/null || true)"
  if [ -z "$setting_value" ]; then
    setting_value="$(
      sed -n "s/^${setting_name}=//p" .env | tail -n 1 | tr -d '\r'
    )"
  fi
  printf '%s' "$setting_value"
}

case "$profile" in
  cpu|gpu) ;;
  *) fail "profile must be 'cpu' or 'gpu'" ;;
esac

[ -f compose.yaml ] && [ -f inference/manifests/tier-s-gemma-3-4b-it-q4_k_m.json ] \
  && [ -f inference/manifests/tier-m-qwen2.5-vl-7b-instruct-awq.json ] \
  || fail "run preflight from the repository root"

if [ "$profile" = "cpu" ]; then
  required_disk_kb=15728640
  required_disk_gb=15
else
  required_disk_kb=31457280
  required_disk_gb=30
fi

command -v docker >/dev/null 2>&1 || fail "Docker is not installed"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is unavailable"
docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"
info "Docker and Compose are available"

available_disk_kb="$(df -Pk . | awk 'NR == 2 {print $4}')"
case "$available_disk_kb" in
  ''|*[!0-9]*) fail "could not determine free disk space" ;;
esac
[ "$available_disk_kb" -ge "$required_disk_kb" ] \
  || fail "at least ${required_disk_gb} GB free disk is required"
info "at least ${required_disk_gb} GB free disk is available"

if [ -z "${ARGUS_WEB_PORT:-}" ] && [ -f .env ]; then
  configured_web_port="$(sed -n 's/^ARGUS_WEB_PORT=//p' .env | tail -n 1)"
  [ -z "$configured_web_port" ] || web_port="$configured_web_port"
fi
if [ -z "$environment" ] && [ -f .env ]; then
  environment="$(sed -n 's/^ARGUS_ENVIRONMENT=//p' .env | tail -n 1)"
fi
case "$web_port" in
  ''|*[!0-9]*) fail "ARGUS_WEB_PORT must be numeric" ;;
esac
[ "$web_port" -ge 1 ] && [ "$web_port" -le 65535 ] \
  || fail "ARGUS_WEB_PORT must be between 1 and 65535"

if docker ps --format '{{.Ports}}' | grep -Eq "[:.]${web_port}->"; then
  fail "port ${web_port} is already published by a Docker container"
fi
if command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
  python3 - "$web_port" <<'PY' || fail "host port ${web_port} is already in use"
import socket
import sys

with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
PY
elif command -v nc >/dev/null 2>&1; then
  ! nc -z 127.0.0.1 "$web_port" || fail "host port ${web_port} is already in use"
elif command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -NonInteractive -Command \
    "if (Get-NetTCPConnection -State Listen -LocalPort ${web_port} -ErrorAction SilentlyContinue) { exit 1 }" \
    || fail "host port ${web_port} is already in use"
else
  fail "cannot verify host port availability (install python3 or netcat)"
fi
info "host port ${web_port} is available"

if [ ! -f .env ]; then
  fail "missing .env; copy deploy/.env.example to .env and review every value"
fi

[ -f deploy/secrets/postgres_password.txt ] \
  || fail "missing PostgreSQL secret; run ./deploy/init-secrets.sh"
[ -f deploy/secrets/redis_password.txt ] \
  || fail "missing Redis secret; run ./deploy/init-secrets.sh"

if grep -q 'replace-before-production' .env && [ "${environment:-development}" = "production" ]; then
  fail "example secrets cannot be used in production"
fi
info "environment file is present"

if [ "$profile" = "gpu" ]; then
  [ "$(read_setting ARGUS_TIER)" = "M" ] \
    || fail "GPU profile requires ARGUS_TIER=M"
  [ "$(read_setting ARGUS_MODEL_PROVIDER)" = "vllm" ] \
    || fail "GPU profile requires ARGUS_MODEL_PROVIDER=vllm"
  [ "$(read_setting ARGUS_MODEL_NAME)" = "qwen2.5-vl-7b-instruct-awq" ] \
    || fail "GPU profile requires the pinned Tier M model name"
  [ "$(read_setting ARGUS_MODEL_ENDPOINT)" = "http://inference-gpu:8000/v1" ] \
    || fail "GPU profile requires the internal vLLM endpoint"
  [ "$(read_setting ARGUS_MODEL_CONTEXT_CEILING)" = "65536" ] \
    || fail "GPU profile requires the validated 65536-token context ceiling"
else
  [ "$(read_setting ARGUS_TIER)" = "S" ] \
    || fail "CPU profile requires ARGUS_TIER=S"
  [ "$(read_setting ARGUS_MODEL_PROVIDER)" = "llama_cpp" ] \
    || fail "CPU profile requires ARGUS_MODEL_PROVIDER=llama_cpp"
fi
info "${profile} model adapter configuration is aligned"

if command -v python3 >/dev/null 2>&1; then
  python_command=python3
elif command -v python >/dev/null 2>&1; then
  python_command=python
else
  fail "Python is required to verify local model artifacts"
fi

if [ "$profile" = "cpu" ]; then
  "$python_command" -m inference.download_model \
    --manifest inference/manifests/tier-s-gemma-3-4b-it-q4_k_m.json \
    --output-dir models \
    --verify-only \
    || fail "Tier S model artifact is missing or failed checksum verification"
  info "Tier S model artifact passed checksum verification"
else
  "$python_command" -m inference.download_bundle \
    --manifest inference/manifests/tier-m-qwen2.5-vl-7b-instruct-awq.json \
    --output-dir models \
    --verify-only \
    || fail "Tier M model bundle is missing or failed checksum verification"
  info "Tier M model bundle passed checksum verification"
fi

"$python_command" -m inference.download_bundle \
  --manifest inference/manifests/bge-m3-onnx-fp32.json \
  --output-dir models \
  --verify-only \
  || fail "BGE-M3 bundle is missing or failed checksum verification"
info "BGE-M3 bundle passed checksum verification"

"$python_command" -m inference.download_bundle \
  --manifest inference/manifests/bge-reranker-v2-m3-safetensors-fp32.json \
  --output-dir models \
  --verify-only \
  || fail "BGE reranker bundle is missing or failed checksum verification"
info "BGE reranker bundle passed checksum verification"

if [ "$profile" = "gpu" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
  nvidia-smi >/dev/null 2>&1 || fail "NVIDIA driver is not operational"
  docker info --format '{{json .Runtimes}}' | grep -qi nvidia \
    || fail "NVIDIA Container Toolkit runtime is unavailable"
  gpu_memory_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | awk 'NR == 1 {print int($1)}')"
  case "$gpu_memory_mb" in
    ''|*[!0-9]*) fail "could not determine GPU memory" ;;
  esac
  [ "$gpu_memory_mb" -ge 23000 ] \
    || fail "Tier M requires one GPU with at least 24 GB VRAM"
  info "GPU driver and container runtime are available"
fi

docker compose --profile "$profile" config --quiet \
  || fail "Compose configuration is invalid"
info "Argus Panopt ${profile} profile passed preflight"
