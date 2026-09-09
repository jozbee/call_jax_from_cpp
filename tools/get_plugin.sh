#!/usr/bin/env bash
#
# Download the prebuilt PJRT CPU plugin for the pinned JAX version.
#
# There is no official prebuilt CPU PJRT C-API plugin (jaxlib never exports
# GetPjrtApi), so this project publishes its own, built from the XLA fork, as
# a GitHub Release asset per JAX version.
#
# Usage:
#   tools/get_plugin.sh [--dest DIR] [--release TAG] [--url URL] [--check]
#                       [--versions-file F] [--manifest F]
#
# With no asset for this platform it prints the remedies and exits 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$REPO_ROOT/build/plugin}"
VERSIONS_FILE="$REPO_ROOT/versions.env"
MANIFEST="$REPO_ROOT/tools/plugin_versions.txt"
RELEASE=""
URL=""
CHECK_ONLY=0

die() { echo "get_plugin: $*" >&2; exit 1; }

# The header comment is the help text; it ends at the first non-comment line.
usage() { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; }

# A PJRT plugin exports GetPjrtApi; anything else cannot be dlopen-ed as one.
exports_pjrt_api() {
  nm -D --defined-only "$1" 2>/dev/null | grep -q ' T GetPjrtApi'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)          DEST="$2"; shift 2 ;;
    --release)       RELEASE="$2"; shift 2 ;;
    --url)           URL="$2"; shift 2 ;;
    --check)         CHECK_ONLY=1; shift ;;
    --versions-file) VERSIONS_FILE="$2"; shift 2 ;;
    --manifest)      MANIFEST="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for tool in curl sha256sum tar; do
  command -v "$tool" >/dev/null || die "$tool is required"
done

# shellcheck disable=SC1090
[[ -f "$VERSIONS_FILE" ]] && . "$VERSIONS_FILE"
RELEASE="${RELEASE:-${PLUGIN_RELEASE:-}}"

case "$(uname -s)" in
  Linux)  os=linux ;;
  Darwin) os=darwin ;;
  *)      die "unsupported OS $(uname -s); build from source with tools/build_plugin.sh" ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) die "unsupported architecture $(uname -m); build from source with tools/build_plugin.sh" ;;
esac
platform="${os}-${arch}"

SO="$DEST/libpjrt_c_api_cpu_plugin.so"
if [[ "$CHECK_ONLY" == 1 ]]; then
  [[ -f "$SO" ]] || die "no plugin at $SO"
  exports_pjrt_api "$SO" || die "$SO does not export GetPjrtApi"
  echo "get_plugin: $SO looks like a PJRT plugin"
  exit 0
fi

# ------------------------------------------------------- resolve url + digest
sha=""
if [[ -z "$URL" ]]; then
  [[ -f "$MANIFEST" ]] || die "no manifest at $MANIFEST"
  # Rows: <release-tag> <os-arch> <url> <sha256>. Comments and blanks ignored.
  line="$(grep -v '^[[:space:]]*#' "$MANIFEST" | grep -v '^[[:space:]]*$' \
          | awk -v r="$RELEASE" -v p="$platform" '$1==r && $2==p {print; exit}')" || true
  if [[ -z "$line" ]]; then
    cat >&2 <<MSG
get_plugin: no prebuilt plugin published for $platform at release '${RELEASE:-<unset>}'.

Options:
  1. Build it from the XLA fork (needs docker, ~30-60 min the first time):
       docker compose -f docker/compose.yml run --rm plugin-builder tools/build_plugin.sh
     or, with bazel on this machine:
       make plugin-source
  2. Publish the asset for this platform:
       tools/release_plugin.sh --version jax-v${JAX_VERSION:-<version>}
  3. Point the runtime at a plugin you already have:
       export PJRT_CPU_PLUGIN=/path/to/libpjrt_c_api_cpu_plugin.so
     A stock (unpatched) plugin works, but jnp.linalg.* executables will fail
     to load and inline execution is unavailable.
MSG
    exit 1
  fi
  URL="$(awk '{print $3}' <<< "$line")"
  sha="$(awk '{print $4}' <<< "$line")"
fi

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "get_plugin: downloading $URL"
curl -fL --retry 3 --retry-delay 2 -o "$tmp/plugin.tar.gz" "$URL"

if [[ -n "$sha" && "$sha" != "-" ]]; then
  echo "$sha  $tmp/plugin.tar.gz" | sha256sum -c - >/dev/null \
    || die "checksum mismatch for $URL (expected $sha)"
  echo "get_plugin: sha256 ok"
else
  echo "get_plugin: warning: no checksum in the manifest for this asset" >&2
fi

tar xzf "$tmp/plugin.tar.gz" -C "$DEST"
[[ -f "$SO" ]] || die "the archive did not contain libpjrt_c_api_cpu_plugin.so"
exports_pjrt_api "$SO" || die "$SO does not export GetPjrtApi"

echo "get_plugin: installed $SO"
[[ -f "$DEST/PLUGIN_INFO.txt" ]] && cat "$DEST/PLUGIN_INFO.txt"
exit 0
