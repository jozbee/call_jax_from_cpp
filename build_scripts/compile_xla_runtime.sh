#!/usr/bin/env bash

# Build the PJRT CPU plugin from the vendored XLA fork.
#
# make sure to use the correct xla commit
# see the `README.md`
#
# The build is an optimized one.  Bazel's default compilation mode is
# `fastbuild` (-O0), and every part of the runtime that is not JIT-generated --
# the thunk executor, buffer management, the Eigen kernels compiled into the
# plugin -- runs at whatever this build produces.  Compiling at -O0 costs both
# throughput and tail latency, so `-c opt` is not optional here.
#
# Environment:
#   BUILD_MODE       `opt` (default) or `fastbuild`, to A/B the two.
#   EXTRA_BAZEL_ARGS extra flags, e.g. `--copt=-march=x86-64-v3` on a machine
#                    whose exact microarchitecture is known.
#   OS_NAME          `Darwin` to produce a .dylib.

set -euo pipefail

BUILD_MODE="${BUILD_MODE:-opt}"
EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS:-}"

# `avx_linux` (-mavx) only means anything on x86; on aarch64 the toolchain
# already targets the host baseline.
ARCH_ARGS=()
if [[ "$(uname -m)" == "x86_64" && "${OS_NAME:-}" != "Darwin" ]]; then
  ARCH_ARGS+=(--config=avx_linux)
fi

cd third_party/xla
./configure.py --backend=CPU --host_compiler=CLANG
# shellcheck disable=SC2086
bazel build \
  -c "$BUILD_MODE" \
  "${ARCH_ARGS[@]}" \
  $EXTRA_BAZEL_ARGS \
  --repo_env=HERMETIC_PYTHON_VERSION=3.11 \
  //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

cd ../..
mkdir -p artifacts

SUFFIX=""
if [[ "$BUILD_MODE" != "opt" ]]; then
  SUFFIX="_${BUILD_MODE}"
fi

if [[ "${OS_NAME:-}" == "Darwin" ]]; then
  OUT="artifacts/libpjrt_c_api_cpu_plugin${SUFFIX}.dylib"
else
  OUT="artifacts/libpjrt_c_api_cpu_plugin${SUFFIX}.so"
fi

# The build tree keeps the unstripped copy for symbolizing profiles; the
# shipped one is stripped of everything the dynamic linker does not need.
cp -f third_party/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so "$OUT"
chmod u+w "$OUT"

if [[ "${OS_NAME:-}" == "Darwin" ]]; then
  install_name_tool -id "$(pwd)/$OUT" "$OUT"
  strip -x "$OUT" || echo "warning: strip failed, keeping unstripped plugin"
else
  strip --strip-unneeded "$OUT" ||
    echo "warning: strip failed, keeping unstripped plugin"
fi

echo "wrote $OUT ($(du -h "$OUT" | cut -f1), mode=$BUILD_MODE)"
