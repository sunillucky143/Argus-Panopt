#!/usr/bin/env sh
set -eu

if docker compose exec -T api python -c \
  "import socket; socket.create_connection(('example.invalid', 443), timeout=3)"; then
  printf 'ERROR: processing network unexpectedly allowed outbound access\n' >&2
  exit 1
fi

printf 'Processing container outbound connection was blocked as required.\n'
