#!/usr/bin/env bash
# Bring up the development container and drop into a shell. The compose file
# grants the privileges the real-time helpers need; a plain `docker run` does
# not.
set -euo pipefail

COMPOSE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker" && pwd)/compose.yml"
SERVICE="${1:-dev}"

echo "Bringing up '$SERVICE'..."
docker compose -f "$COMPOSE" up -d "$SERVICE"
exec docker compose -f "$COMPOSE" exec "$SERVICE" bash
