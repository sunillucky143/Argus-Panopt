#!/usr/bin/env sh
set -eu

probe_model=false
case "${1:-}" in
  "") ;;
  --probe-model-image) probe_model=true ;;
  *) printf 'usage: %s [--probe-model-image]\n' "$0" >&2; exit 2 ;;
esac

if docker compose exec -T api python -c \
  "import socket, sys; sock = socket.socket(); sock.settimeout(3); result = sock.connect_ex(('1.1.1.1', 443)); sock.close(); sys.exit(0 if result == 0 else 10)"; then
  printf 'ERROR: API processing network unexpectedly allowed outbound access\n' >&2
  exit 1
else
  probe_status=$?
  [ "$probe_status" -eq 10 ] \
    || { printf 'ERROR: API egress probe failed with status %s\n' "$probe_status" >&2; exit 1; }
fi

if docker compose ps --services --status running | grep -qx 'inference-cpu'; then
  if docker compose exec -T inference-cpu curl \
    --silent --output /dev/null --connect-timeout 3 --max-time 3 \
    https://1.1.1.1/; then
    printf 'ERROR: model processing network unexpectedly allowed outbound access\n' >&2
    exit 1
  else
    probe_status=$?
    case "$probe_status" in
      7|28) ;;
      *) printf 'ERROR: model egress probe failed with status %s\n' "$probe_status" >&2; exit 1 ;;
    esac
  fi
elif [ "$probe_model" = true ]; then
  if docker compose --profile cpu run --rm --no-deps --entrypoint curl inference-cpu \
    --silent --output /dev/null --connect-timeout 3 --max-time 3 \
    https://1.1.1.1/; then
    printf 'ERROR: model processing network unexpectedly allowed outbound access\n' >&2
    exit 1
  else
    probe_status=$?
    case "$probe_status" in
      7|28) ;;
      *) printf 'ERROR: model image egress probe failed with status %s\n' "$probe_status" >&2; exit 1 ;;
    esac
  fi
fi

printf 'Running processing containers could not reach the public internet.\n'
