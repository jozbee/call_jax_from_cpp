#!/usr/bin/env bash
# Keep /opt/venv in step with the bind-mounted lock file, then run the command.
# The venv lives outside /workspace because a bind mount would shadow it.
set -euo pipefail

sync_venv() {
  (cd /workspace && uv sync "$@" --all-groups --extra docs \
     --python /usr/bin/python3)
}

if [[ -f /workspace/pyproject.toml ]]; then
  stamp=/opt/venv/.lock-stamp
  lock=/workspace/uv.lock
  [[ -f "$lock" ]] || lock=/workspace/pyproject.toml
  if [[ ! -f "$stamp" ]] || ! cmp -s "$lock" "$stamp"; then
    echo "entrypoint: syncing /opt/venv from $(basename "$lock")" >&2
    # A lock that no longer resolves must not strand the container.
    if [[ -f /workspace/uv.lock ]]; then
      sync_venv --locked || sync_venv
    else
      sync_venv
    fi
    cp -f "$lock" "$stamp" 2>/dev/null || true
  fi
fi

# The image sets PJRT_CPU_PLUGIN whether or not a plugin was baked in, and the
# variable outranks the compiled-in path: left pointing at nothing, it would
# mask a good plugin in the workspace.
if [[ -n "${PJRT_CPU_PLUGIN:-}" && ! -f "${PJRT_CPU_PLUGIN}" ]]; then
  unset PJRT_CPU_PLUGIN
fi

# A workspace plugin wins over the image's: its binaries were built with a
# default path that only resolves outside the container.
if [[ -f /workspace/build/plugin/libpjrt_c_api_cpu_plugin.so ]]; then
  export PJRT_CPU_PLUGIN=/workspace/build/plugin/libpjrt_c_api_cpu_plugin.so
fi

exec "$@"
