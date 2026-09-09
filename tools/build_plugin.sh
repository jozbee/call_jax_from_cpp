#!/usr/bin/env bash
#
# Build the PJRT CPU plugin from the XLA fork.
#
# The fork adds jaxlib's LAPACK FFI kernels (so jnp.linalg.* executables load
# at all) and the CPU plugin create options that make inline execution
# reachable through the C API; docs/developer/xla-fork.md has the patches.
#
# `-c opt` is the default because it is what ships. Build mode does not
# measurably change call latency: the compute kernels are compiled at export
# time into the artifact, and the plugin only orchestrates.
#
# Environment:
#   XLA_DIR       XLA source tree (default: third_party/xla)
#   OUT_DIR       where to put the .so (default: build/plugin)
#   BUILD_MODE    bazel -c mode: opt (default) or fastbuild
#   ARCH_FLAGS    extra bazel flags, e.g. --config=avx_linux. Release assets
#                 are built WITHOUT these so they run on any x86-64.
#   BAZEL_STARTUP_ARGS  e.g. --output_user_root=/mnt/bazel (CI disk pressure)
#   HERMETIC_PYTHON_VERSION  default 3.12
#
# Usage: tools/build_plugin.sh [--package] [--mode opt|fastbuild]
#                              [--xla-dir DIR] [--out DIR] [--arch-flags "..."]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XLA_DIR="${XLA_DIR:-$REPO_ROOT/third_party/xla}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/build/plugin}"
BUILD_MODE="${BUILD_MODE:-opt}"
ARCH_FLAGS="${ARCH_FLAGS:-}"
HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}"
PACKAGE=0

die() { echo "build_plugin: $*" >&2; exit 1; }

# The header comment is the help text; it ends at the first non-comment line.
usage() { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)    PACKAGE=1; shift ;;
    --mode)       BUILD_MODE="$2"; shift 2 ;;
    --xla-dir)    XLA_DIR="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --arch-flags) ARCH_FLAGS="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Sourced before anything reads JAX_VERSION or the fork metadata.
# shellcheck disable=SC1091
[[ -f "$REPO_ROOT/versions.env" ]] && . "$REPO_ROOT/versions.env"

# ---------------------------------------------------------------- preconditions
[[ -d "$XLA_DIR" ]] || die "no XLA tree at $XLA_DIR
  git submodule update --init --depth 1 third_party/xla
  (or point XLA_DIR at a working clone)"
[[ -f "$XLA_DIR/WORKSPACE" || -f "$XLA_DIR/MODULE.bazel" ]] \
  || die "$XLA_DIR does not look like an XLA checkout (no WORKSPACE)"
command -v bazel >/dev/null || die "bazel/bazelisk not on PATH (the docker image has it:
  docker compose -f docker/compose.yml run --rm plugin-builder tools/build_plugin.sh)"
command -v xxd >/dev/null || die "xxd not found (apt-get install xxd)"

# The LAPACK patch links the plugin against the system LAPACK and BLAS.
have_lapack=0
for pat in /usr/lib/*/liblapack.so* /usr/lib/liblapack.so* /usr/lib64/liblapack.so* \
           /usr/lib/*/liblapack.a /usr/lib/liblapack.a; do
  [[ -e "$pat" ]] && { have_lapack=1; break; }
done
if [[ "$have_lapack" == 0 ]]; then
  echo "build_plugin: warning: no liblapack found; the link step will fail if" >&2
  echo "  the fork's LAPACK patch is present (apt-get install liblapack-dev libblas-dev)" >&2
fi

avail_gb=$(df -BG --output=avail "$XLA_DIR" | tail -1 | tr -dc '0-9')
[[ "${avail_gb:-0}" -ge 25 ]] || echo "build_plugin: warning: only ${avail_gb}G free; bazel wants ~25G" >&2

# ------------------------------------------------------------------- configure
cd "$XLA_DIR"
if [[ -f configure.py ]]; then
  echo "build_plugin: configure.py --backend=CPU --host_compiler=CLANG"
  # configure.py may refuse --host_compiler=CLANG without --clang_path (newer
  # XLA defaults to a hermetic clang). No configure at all is fine: the
  # checked-in .bazelrc already selects a working CPU toolchain.
  python3 configure.py --backend=CPU --host_compiler=CLANG \
    || python3 configure.py --backend=CPU \
    || echo "build_plugin: configure.py declined; using the checked-in .bazelrc"
fi

# ----------------------------------------------------------------------- build
# No --config=bzlmod: XLA at this pin selects WORKSPACE mode in its .bazelrc.
read -r -a startup_args <<< "${BAZEL_STARTUP_ARGS:-}"
read -r -a arch_args <<< "$ARCH_FLAGS"

echo "build_plugin: bazel build -c $BUILD_MODE (bazel $(bazel --version 2>/dev/null | tail -1))"
bazel "${startup_args[@]}" build \
  -c "$BUILD_MODE" \
  "${arch_args[@]}" \
  --repo_env="HERMETIC_PYTHON_VERSION=$HERMETIC_PYTHON_VERSION" \
  //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

BUILT="$XLA_DIR/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
[[ -f "$BUILT" ]] || die "bazel reported success but $BUILT is missing"

# --------------------------------------------------------------------- install
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/libpjrt_c_api_cpu_plugin.so"
cp -f "$BUILT" "$OUT"
chmod u+w "$OUT"
strip --strip-unneeded "$OUT" || echo "build_plugin: warning: strip failed, keeping unstripped plugin" >&2

# A PJRT plugin exports GetPjrtApi; anything else cannot be dlopen-ed as one.
nm -D --defined-only "$OUT" 2>/dev/null | grep -q ' T GetPjrtApi' \
  || die "$OUT does not export GetPjrtApi"

# What the tree is, over what versions.env claims. In the container git fails
# (a submodule's .git is a file pointing into the superproject, which is not
# mounted), so fall back to versions.env rather than record "unknown".
fork_commit="$(git -C "$XLA_DIR" rev-parse HEAD 2>/dev/null || true)"
fork_branch="$(git -C "$XLA_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ -z "$fork_commit" && -n "${XLA_FORK_COMMIT:-}" ]]; then
  fork_commit="$XLA_FORK_COMMIT"
  fork_branch="${XLA_FORK_BRANCH:-}"
  echo "build_plugin: no git metadata in $XLA_DIR; recording the fork commit from versions.env" >&2
fi
fork_commit="${fork_commit:-unknown}"
fork_branch="${fork_branch:-unknown}"
api_minor="$(grep -m1 '^#define PJRT_API_MINOR' "$XLA_DIR/xla/pjrt/c/pjrt_c_api.h" 2>/dev/null | awk '{print $3}')"
glibc_max="$(objdump -T "$OUT" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"

cat > "$OUT_DIR/PLUGIN_INFO.txt" <<INFO
PJRT CPU plugin for call_jax_from_cpp
jax_version      = ${JAX_VERSION:-unknown}
xla_commit       = ${XLA_COMMIT:-unknown}
fork_commit      = $fork_commit
fork_branch      = $fork_branch
pjrt_api_minor   = ${api_minor:-unknown}
build_mode       = $BUILD_MODE
arch_flags       = ${ARCH_FLAGS:-none (baseline)}
host_arch        = $(uname -m)
glibc_max        = ${glibc_max:-unknown}
size             = $(du -h "$OUT" | cut -f1)
runtime_deps     = $(ldd "$OUT" 2>/dev/null | awk '{print $1}' | grep -E '^lib' | sort | tr '\n' ' ')

Built from the XLA fork; the binary is Apache-2.0 (see LICENSE-xla).
INFO

if [[ -f "$XLA_DIR/LICENSE" ]]; then
  cp -f "$XLA_DIR/LICENSE" "$OUT_DIR/LICENSE-xla"
fi

echo "build_plugin: wrote $OUT ($(du -h "$OUT" | cut -f1), mode=$BUILD_MODE)"
cat "$OUT_DIR/PLUGIN_INFO.txt"

# --------------------------------------------------------------------- package
if [[ "$PACKAGE" == 1 ]]; then
  arch="$(uname -m)"
  ver="${JAX_VERSION:-unknown}"
  tarball="$OUT_DIR/pjrt_cpu_plugin-jax-v${ver}-linux-${arch}.tar.gz"
  # The binary is Apache-2.0 XLA; ship its licence whenever the tree had one.
  tar_members=(libpjrt_c_api_cpu_plugin.so PLUGIN_INFO.txt)
  [[ -f "$OUT_DIR/LICENSE-xla" ]] && tar_members+=(LICENSE-xla)
  tar czf "$tarball" -C "$OUT_DIR" "${tar_members[@]}"
  (cd "$OUT_DIR" && sha256sum "$(basename "$tarball")" > "$(basename "$tarball").sha256")
  echo "build_plugin: packaged $tarball"
fi
