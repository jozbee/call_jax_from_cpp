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

# The image sets PJRT_CPU_PLUGIN unconditionally, but the plugin is only baked
# in when the image was built with PLUGIN_SOURCE=prebuilt or source. Pointing
# the runtime at a file that is not there would mask a perfectly good plugin in
# the bind-mounted workspace, because the environment variable outranks the
# path compiled into the binary.
if [[ -n "${PJRT_CPU_PLUGIN:-}" && ! -f "${PJRT_CPU_PLUGIN}" ]]; then
  unset PJRT_CPU_PLUGIN
fi

# A workspace plugin wins over anything baked into the image: the bind-mounted
# tree is what the caller is working on, and its binaries were built with a
# default path that only resolves outside the container.
if [[ -f /workspace/build/plugin/libpjrt_c_api_cpu_plugin.so ]]; then
  export PJRT_CPU_PLUGIN=/workspace/build/plugin/libpjrt_c_api_cpu_plugin.so
fi

exec "$@"
