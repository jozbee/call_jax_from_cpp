#!/usr/bin/env bash
set -euo pipefail

source ~/.virtualenvs/jax/bin/activate

# /workspace is bind-mounted at container start (not baked into the image),
# so the editable install has to happen here rather than in the Dockerfile.
if [[ -f /workspace/pyproject.toml ]]; then
  pip install --quiet -e /workspace
fi

exec "$@"
