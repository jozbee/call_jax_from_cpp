#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <amd64|arm64>" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage

case "$1" in
  amd64) SERVICE=ubuntu-amd64 ;;
  arm64) SERVICE=ubuntu-arm64 ;;
  *) usage ;;
esac

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Ensuring '$SERVICE' is built and running..."
docker compose up -d "$SERVICE"

echo "Connecting to '$SERVICE'..."
exec docker compose exec "$SERVICE" bash
