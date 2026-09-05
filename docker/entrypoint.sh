#!/usr/bin/env bash
# Keeps /opt/venv in step with the bind-mounted lock file, then runs the
# command. The venv lives outside /workspace on purpose: the workspace is a
# bind mount and would shadow anything baked into the image.
set -euo pipefail

if [[ -f /workspace/pyproject.toml ]]; then
  stamp=/opt/venv/.lock-stamp
  lock=/workspace/uv.lock
  [[ -f "$lock" ]] || lock=/workspace/pyproject.toml
  if [[ ! -f "$stamp" ]] || ! cmp -s "$lock" "$stamp"; then
    echo "entrypoint: syncing /opt/venv from $(basename "$lock")" >&2
    if [[ -f /workspace/uv.lock ]]; then
      (cd /workspace && uv sync --locked --all-groups --extra docs --python /usr/bin/python3) || \
        (cd /workspace && uv sync --all-groups --extra docs --python /usr/bin/python3)
    else
      (cd /workspace && uv sync --all-groups --extra docs --python /usr/bin/python3)
    fi
    cp -f "$lock" "$stamp" 2>/dev/null || true
  fi
fi

exec "$@"
