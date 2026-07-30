#!/usr/bin/env sh
set -eu

force=false
if [ "${1:-}" = "--force" ]; then
  force=true
elif [ "$#" -gt 0 ]; then
  printf 'usage: %s [--force]\n' "$0" >&2
  exit 2
fi

[ -f .env ] || {
  printf 'ERROR: missing .env; copy deploy/.env.example first\n' >&2
  exit 1
}

read_value() {
  key="$1"
  case "$key" in
    POSTGRES_PASSWORD) value="${POSTGRES_PASSWORD:-}" ;;
    REDIS_PASSWORD) value="${REDIS_PASSWORD:-}" ;;
    *) value="" ;;
  esac
  if [ -z "$value" ]; then
    value="$(sed -n "s/^${key}=//p" .env | tail -n 1)"
  fi
  [ -n "$value" ] || {
    printf 'ERROR: %s is missing or empty in .env\n' "$key" >&2
    exit 1
  }
  printf '%s' "$value"
}

write_secret() {
  path="$1"
  value="$2"
  if [ -e "$path" ] && [ "$force" = false ]; then
    printf 'Keeping existing %s (use --force to replace it).\n' "$path"
    return
  fi
  (umask 077 && printf '%s' "$value" >"$path")
  printf 'Wrote %s with owner-only permissions.\n' "$path"
}

mkdir -p deploy/secrets
write_secret deploy/secrets/postgres_password.txt "$(read_value POSTGRES_PASSWORD)"
write_secret deploy/secrets/redis_password.txt "$(read_value REDIS_PASSWORD)"
