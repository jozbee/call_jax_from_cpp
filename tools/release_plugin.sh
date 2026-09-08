#!/usr/bin/env bash
#
# Build the prebuilt PJRT CPU plugin and publish it to this repository's GitHub
# Releases.
#
# There is no official prebuilt PJRT CPU plugin anywhere: jaxlib links its CPU
# client statically and never exports GetPjrtApi, so a C++ caller has nothing to
# dlopen. This project therefore ships its own, built from the XLA fork. The
# other producer of the same assets is .github/workflows/plugin.yml, and the two
# must agree exactly on names, because tools/plugin_versions.txt,
# cmake/GetPjrtPlugin.cmake and `make plugin` all resolve assets by name:
#
#   tag    plugin-jax-v<jax version>
#   asset  pjrt_cpu_plugin-jax-v<jax version>-linux-<arch>.tar.gz
#          pjrt_cpu_plugin-jax-v<jax version>-linux-<arch>.tar.gz.sha256
#
# Usage:
#   tools/release_plugin.sh --version jax-v0.11.0 [--arch x86_64,aarch64]
#                           [--tag TAG] [--xla-ref REF] [--dry-run]
#                           [--update-manifest] [--skip-build] [--force]
#
#   --version V        the JAX version being released; "jax-v0.11.0",
#                      "v0.11.0" and "0.11.0" all mean the same thing.
#                      Default: JAX_VERSION from versions.env.
#   --arch LIST        comma separated, from x86_64 and aarch64. Default: both.
#   --tag TAG          release tag. Default: plugin-<version>.
#   --xla-ref REF      fork commit or branch to build. Default: the
#                      third_party/xla submodule pointer, else XLA_FORK_COMMIT.
#   --skip-build       reuse build/release/<arch> from an earlier run.
#   --update-manifest  rewrite this release's rows in tools/plugin_versions.txt
#                      and print the diff for a human to commit.
#   --force            replace assets that already exist on the release.
#   --dry-run          do everything except the gh writes, printing them
#                      instead. --dry-run --skip-build rehearses the whole
#                      flow in seconds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_FILE="$REPO_ROOT/versions.env"
MANIFEST="$REPO_ROOT/tools/plugin_versions.txt"
RELEASE_DIR="$REPO_ROOT/build/release"

VERSION=""
TAG=""
XLA_REF=""
ARCH_LIST=""
DRY_RUN=0
UPDATE_MANIFEST=0
SKIP_BUILD=0
FORCE=0
WARNINGS=0

die()  { echo "release_plugin: $*" >&2; exit 1; }
warn() { echo "release_plugin: warning: $*" >&2; WARNINGS=$((WARNINGS + 1)); }
say()  { echo "release_plugin: $*"; }

# The header comment is the help text, and stopping at the first non-comment
# line means there is no line range to keep in step with edits.
usage() { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; }

# Print $2 or explain which flag was left dangling; callers pass "$@".
arg() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"
  printf '%s' "$2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)         VERSION="$(arg "$@")"; shift 2 ;;
    --arch)            ARCH_LIST="$(arg "$@")"; shift 2 ;;
    --tag)             TAG="$(arg "$@")"; shift 2 ;;
    --xla-ref)         XLA_REF="$(arg "$@")"; shift 2 ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --update-manifest) UPDATE_MANIFEST=1; shift ;;
    --skip-build)      SKIP_BUILD=1; shift ;;
    --force)           FORCE=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f "$VERSIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  . "$VERSIONS_FILE"
fi

# ------------------------------------------------------------------ resolution
VERSION="${VERSION:-${JAX_VERSION:-}}"
[[ -n "$VERSION" ]] || die "--version is required (versions.env sets no JAX_VERSION)"
VERSION="${VERSION#plugin-}"
[[ "$VERSION" == jax-v* ]] || VERSION="jax-v${VERSION#v}"
JAX_VER="${VERSION#jax-v}"
TAG="${TAG:-plugin-$VERSION}"

# build_plugin.sh names the tarball from versions.env and the loader checks the
# sidecar against the plugin it finds, so a release that disagrees with the tree
# ships assets whose names contradict what they contain.
if [[ -n "${JAX_VERSION:-}" && "$JAX_VER" != "$JAX_VERSION" ]]; then
  if [[ "$FORCE" == 1 ]]; then
    warn "releasing JAX $JAX_VER from a tree pinned to $JAX_VERSION (--force)"
  else
    die "asked for JAX $JAX_VER but versions.env pins $JAX_VERSION; bump the tree first (docs/developer/bumping-jax.md)"
  fi
fi

ARCH_LIST="${ARCH_LIST:-x86_64,aarch64}"
IFS=',' read -r -a ARCHES <<< "$ARCH_LIST"
[[ ${#ARCHES[@]} -gt 0 ]] || die "--arch is empty"
for a in "${ARCHES[@]}"; do
  case "$a" in
    x86_64|aarch64) ;;
    *) die "unsupported --arch '$a' (x86_64, aarch64)" ;;
  esac
done

host_arch="$(uname -m)"
case "$host_arch" in
  amd64) host_arch=x86_64 ;;
  arm64) host_arch=aarch64 ;;
esac

# --------------------------------------------------------------- preconditions
for tool in tar sha256sum awk curl git diff; do
  command -v "$tool" >/dev/null || die "$tool is required"
done
command -v docker >/dev/null || die "docker is required"
docker buildx version >/dev/null 2>&1 \
  || die "docker buildx is required (apt-get install docker-buildx-plugin)"
command -v gh >/dev/null || die "gh (the GitHub CLI) is required"
if ! gh auth status >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == 1 ]]; then
    warn "gh is not authenticated; the release URLs below are predictions"
  else
    die "gh is not authenticated: run 'gh auth login'"
  fi
fi

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
  warn "the working tree is dirty, so these assets do not correspond to any commit"
fi

if [[ -z "$XLA_REF" ]]; then
  # A gitlink entry is "160000 commit <sha>\t<path>".
  # No commits yet, or no submodule in HEAD: fall back to versions.env.
  XLA_REF="$(git -C "$REPO_ROOT" ls-tree HEAD third_party/xla 2>/dev/null \
             | awk '$2 == "commit" { print $3 }' || true)"
fi
XLA_REF="${XLA_REF:-${XLA_FORK_COMMIT:-}}"
[[ -n "$XLA_REF" ]] || die "no fork ref: pass --xla-ref or set XLA_FORK_COMMIT in versions.env"
FORK_REPO="${XLA_FORK_REPO:-}"
[[ -n "$FORK_REPO" ]] || die "versions.env sets no XLA_FORK_REPO"

# https://github.com/owner/repo.git and git@github.com:owner/repo.git both
# reduce to owner/repo; anything that is not GitHub yields nothing.
github_slug() {
  local url="${1%.git}"
  case "$url" in
    https://github.com/*)   printf '%s' "${url#https://github.com/}" ;;
    ssh://git@github.com/*) printf '%s' "${url#ssh://git@github.com/}" ;;
    git@github.com:*)       printf '%s' "${url#git@github.com:}" ;;
  esac
}

# Is $2 fetchable by someone who is not us? This whole check exists because a
# previous release named a fork commit that had never been pushed: the assets
# could not be rebuilt by anyone, including their author.
ref_is_published() {
  local repo="$1" ref="$2" slug
  # A branch or tag name resolves directly.
  if git ls-remote --exit-code "$repo" "$ref" >/dev/null 2>&1; then
    return 0
  fi
  # A commit hash does not: ls-remote sees ref tips only, so a tip match proves
  # it is published but no match proves nothing.
  if git ls-remote "$repo" 2>/dev/null \
     | awk -v s="$ref" '$1 == s { hit = 1 } END { exit !hit }'; then
    return 0
  fi
  slug="$(github_slug "$repo")"
  [[ -n "$slug" ]] || return 1
  # Ask GitHub, which answers for any commit reachable from any ref. Through gh
  # first: a private fork returns 404 to an anonymous curl even when pushed.
  if gh api "repos/$slug/commits/$ref" --jq .sha >/dev/null 2>&1; then
    return 0
  fi
  curl -fsSL -o /dev/null --max-time 20 "https://github.com/$slug/commit/$ref"
}

say "release $TAG, JAX $JAX_VER, arches ${ARCHES[*]}"
say "fork $FORK_REPO @ $XLA_REF"
if ! ref_is_published "$FORK_REPO" "$XLA_REF"; then
  die "$XLA_REF is not reachable on $FORK_REPO:
  push the fork branch first (docs/developer/bumping-jax.md step 8)
  Building from a local-only commit produces assets nobody can reproduce."
fi

SLUG="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
if [[ -z "$SLUG" ]]; then
  SLUG="$(github_slug "$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)")"
fi
[[ -n "$SLUG" ]] || die "cannot tell which GitHub repository to release to (no gh access and no github.com origin)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# -------------------------------------------------------------------- packaging
# GetPjrtApi is the plugin's only entry point, and the object has to be for the
# architecture it is being packaged as: with no binfmt handler installed, a
# cross build can quietly produce host-architecture objects instead.
verify_so() {
  local so="$1" arch="$2" syms="" machine=""
  if syms="$(nm -D --defined-only "$so" 2>/dev/null)"; then
    grep -q ' T GetPjrtApi' <<< "$syms" || die "$so does not export GetPjrtApi"
  else
    warn "nm could not read $so; skipping the GetPjrtApi check"
  fi
  if command -v readelf >/dev/null; then
    machine="$(readelf -h "$so" 2>/dev/null | sed -n 's/^ *Machine: *//p')"
  fi
  case "$arch:$machine" in
    x86_64:*X86-64*|aarch64:*AArch64*) ;;
    *:) warn "could not read the ELF machine of $so" ;;
    *) die "$so is a '$machine' object but is being packaged as $arch;
  install the binfmt handler (docker run --privileged --rm tonistiigi/binfmt --install arm64)" ;;
  esac
}

# The container runs build_plugin.sh from /home/dev, outside any checkout, so it
# cannot read versions.env and records "unknown" where it would otherwise name
# the JAX and XLA versions. A tarball whose own metadata cannot say what it was
# built for is not worth publishing, so fill those in from versions.env.
stamp_plugin_info() {
  local info="$1" key value
  for key in jax_version xla_commit; do
    case "$key" in
      jax_version) value="$JAX_VER" ;;
      xla_commit)  value="${XLA_COMMIT:-unknown}" ;;
    esac
    if [[ "$value" != unknown ]] && grep -qE "^$key *= *unknown *\$" "$info"; then
      sed -i "s|^$key\\( *\\)= *unknown *\$|$key\\1= $value|" "$info"
      warn "PLUGIN_INFO.txt said $key = unknown; set to $value from versions.env"
    fi
  done
  # Rewrite the trailer rather than skip it when it is already there. A
  # --dry-run followed by a real run stamps twice, and the first pass may have
  # resolved XLA_REF from a submodule pointer that has since been committed --
  # which shipped an asset whose fork_ref contradicted its own fork_commit.
  # Skipping is the bug; a stale trailer is worse than none.
  if grep -q '^release_tag' "$info"; then
    sed -i '/^release_tag/,$d' "$info"
    sed -i -e :a -e '/^$/{$d;N;ba' -e '}' "$info"
  fi
  printf '\nrelease_tag      = %s\nfork_repo        = %s\nfork_ref         = %s\n' \
    "$TAG" "$FORK_REPO" "$XLA_REF" >> "$info"
}

info_field() {  # PLUGIN_INFO.txt, key
  sed -n "s|^$2 *= *||p" "$1" | head -1
}

package_arch() {
  local arch="$1" dir="$2" base so
  so="$dir/libpjrt_c_api_cpu_plugin.so"
  [[ -f "$so" ]] || die "$so is missing (the buildx export produced nothing)"
  [[ -f "$dir/PLUGIN_INFO.txt" ]] || die "$dir/PLUGIN_INFO.txt is missing"
  # The repository is Unlicense but this binary is XLA, so Apache-2.0 requires
  # its licence to travel with it.
  if [[ ! -f "$dir/LICENSE-xla" ]]; then
    if [[ -f "$REPO_ROOT/third_party/xla/LICENSE" ]]; then
      cp -f "$REPO_ROOT/third_party/xla/LICENSE" "$dir/LICENSE-xla"
    else
      die "$dir/LICENSE-xla is missing and third_party/xla/LICENSE is not available;
  the plugin binary is Apache-2.0 XLA and must ship its licence"
    fi
  fi
  verify_so "$so" "$arch"
  stamp_plugin_info "$dir/PLUGIN_INFO.txt"

  local minor
  minor="$(info_field "$dir/PLUGIN_INFO.txt" pjrt_api_minor)"
  if [[ -n "$minor" && -n "${PJRT_API_MINOR:-}" && "$minor" != "$PJRT_API_MINOR" ]]; then
    warn "$arch plugin is PJRT API 0.$minor but third_party/pjrt is 0.$PJRT_API_MINOR"
  fi

  base="pjrt_cpu_plugin-$VERSION-linux-$arch.tar.gz"
  tar czf "$RELEASE_DIR/$base" -C "$dir" \
    libpjrt_c_api_cpu_plugin.so PLUGIN_INFO.txt LICENSE-xla
  ( cd "$RELEASE_DIR" && sha256sum "$base" > "$base.sha256" )
  say "packaged $RELEASE_DIR/$base ($(du -h "$RELEASE_DIR/$base" | cut -f1))"
}

# ----------------------------------------------------------------------- build
mkdir -p "$RELEASE_DIR"
for arch in "${ARCHES[@]}"; do
  platform=linux/amd64
  [[ "$arch" != aarch64 ]] || platform=linux/arm64
  outdir="$RELEASE_DIR/$arch"

  if [[ "$SKIP_BUILD" == 1 ]]; then
    [[ -f "$outdir/libpjrt_c_api_cpu_plugin.so" ]] \
      || die "--skip-build, but $outdir/libpjrt_c_api_cpu_plugin.so does not exist"
    say "reusing $outdir"
  else
    if [[ "$arch" != "$host_arch" ]]; then
      warn "building $arch on a $host_arch host runs every compiler invocation
  under QEMU: budget hours, not minutes, and once per machine
      docker run --privileged --rm tonistiigi/binfmt --install arm64
  Building only --arch $host_arch here, the other half natively on an $arch
  machine, and uploading both to the same tag is much faster."
      if ! docker buildx inspect --bootstrap 2>/dev/null | grep -q "$platform"; then
        warn "this buildx builder does not advertise $platform; install the binfmt handler above or select a builder that has it"
      fi
    fi
    rm -rf "$outdir"
    mkdir -p "$outdir"
    say "building $arch ($platform) from $XLA_REF"
    # docs: begin release-buildx
    docker buildx build \
      --platform "$platform" \
      --target plugin-export \
      --build-arg PLUGIN_SOURCE=source \
      --build-arg XLA_REPO="$FORK_REPO" \
      --build-arg XLA_REF="$XLA_REF" \
      --output "type=local,dest=$outdir" \
      -f "$REPO_ROOT/docker/Dockerfile" \
      "$REPO_ROOT"
    # docs: end release-buildx
  fi

  package_arch "$arch" "$outdir"
done

# ----------------------------------------------------------------------- notes
NOTES="$RELEASE_DIR/NOTES.md"
{
  echo "PJRT CPU plugin for JAX $JAX_VER."
  echo
  echo "jaxlib links its CPU client statically and never exports \`GetPjrtApi\`,"
  echo "so there is no official prebuilt PJRT CPU plugin. This one is built from"
  echo "the XLA fork, which adds jaxlib's LAPACK FFI kernels (without them any"
  echo "executable lowering to \`lapack_*_ffi\` fails to load) and the CPU plugin"
  echo "create options that make inline execution reachable through the C API."
  echo
  echo '| | |'
  echo '|---|---|'
  echo "| jax / jaxlib | $JAX_VER / ${JAXLIB_VERSION:-$JAX_VER} |"
  echo "| XLA commit | \`${XLA_COMMIT:-unknown}\` |"
  echo "| fork | \`${XLA_FORK_BRANCH:-unknown}\` @ \`$XLA_REF\` |"
  echo "| fork repo | $FORK_REPO |"
  echo "| PJRT C API | ${PJRT_API_MAJOR:-0}.${PJRT_API_MINOR:-unknown} |"
  echo "| build | \`bazel -c opt\`, no \`--config=avx_*\` |"
  echo
  echo "Built baseline on purpose: the plugin only orchestrates, and the compute"
  echo "kernels are LLVM-compiled at export time into the \`.binpb\`, so the"
  echo "serialized executable is the ISA-locked half, not this."
  echo
  for arch in "${ARCHES[@]}"; do
    base="pjrt_cpu_plugin-$VERSION-linux-$arch.tar.gz"
    info="$RELEASE_DIR/$arch/PLUGIN_INFO.txt"
    echo "## linux-$arch"
    echo
    echo "- \`$base\`"
    echo "- sha256 \`$(awk '{ print $1; exit }' "$RELEASE_DIR/$base.sha256")\`"
    if [[ -f "$info" ]]; then
      glibc="$(info_field "$info" glibc_max)"
      deps="$(info_field "$info" runtime_deps)"
      echo "- requires glibc >= ${glibc#GLIBC_}"
      echo "- links against $deps"
      echo
      echo '```'
      cat "$info"
      echo '```'
    fi
    echo
  done
  echo "## Install"
  echo
  echo '```sh'
  echo "make plugin            # downloads and sha256-verifies the asset"
  echo "tools/get_plugin.sh    # the same thing, directly"
  echo '```'
  echo
  echo "The checksums above are recorded in \`tools/plugin_versions.txt\`;"
  echo "\`make plugin\` refuses an asset that does not match."
  echo
  echo "The plugin binary is Apache-2.0 (XLA), shipped as LICENSE-xla inside each"
  echo "tarball, even though this repository is released under the Unlicense."
} > "$NOTES"
say "wrote $NOTES"

# --------------------------------------------------------------------- publish
gh_run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'release_plugin: [dry-run] gh'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  gh "$@"
}

assets=()
for arch in "${ARCHES[@]}"; do
  base="pjrt_cpu_plugin-$VERSION-linux-$arch.tar.gz"
  assets+=("$RELEASE_DIR/$base" "$RELEASE_DIR/$base.sha256")
done

existing=""
if gh release view "$TAG" >/dev/null 2>&1; then
  say "release $TAG already exists; adding assets to it"
  say "its notes are left alone; to refresh them: gh release edit $TAG --notes-file $NOTES"
  existing="$(gh release view "$TAG" --json assets --jq '.assets[].name' 2>/dev/null || true)"
else
  say "creating release $TAG"
  # --latest=false: these are per-JAX-version toolchain assets, and "latest"
  # must keep meaning the latest source release.
  gh_run release create "$TAG" \
    --title "PJRT CPU plugin for JAX $JAX_VER" \
    --notes-file "$NOTES" \
    --latest=false
fi

clash=()
for a in "${assets[@]}"; do
  if grep -qxF "$(basename "$a")" <<< "$existing"; then
    clash+=("$(basename "$a")")
  fi
done
if [[ ${#clash[@]} -gt 0 && "$FORCE" != 1 ]]; then
  die "already published on $TAG: ${clash[*]}
  Asset names are load-bearing: replacing a binary under a name whose sha256 is
  already recorded in tools/plugin_versions.txt breaks every checkout that
  pinned it. Re-run with --force to clobber deliberately."
fi

upload=("$TAG" "${assets[@]}")
if [[ "$FORCE" == 1 ]]; then
  upload+=(--clobber)
fi
gh_run release upload "${upload[@]}"

# -------------------------------------------------------------------- manifest
manifest_rows() {
  local arch base sha url
  for arch in "${ARCHES[@]}"; do
    base="pjrt_cpu_plugin-$VERSION-linux-$arch.tar.gz"
    sha="$(awk '{ print $1; exit }' "$RELEASE_DIR/$base.sha256")"
    url="https://github.com/$SLUG/releases/download/$TAG/$base"
    printf '%-22s %-14s %s  %s\n' "$TAG" "linux-$arch" "$url" "$sha"
  done
}

# Existing rows for the (tag, platform) pairs being republished are dropped, so
# re-running for one architecture never orphans the other's row.
platforms=""
for arch in "${ARCHES[@]}"; do platforms+="linux-$arch "; done
NEW_MANIFEST="$TMP/plugin_versions.txt"
manifest_src="$MANIFEST"
if [[ ! -f "$MANIFEST" ]]; then
  warn "$MANIFEST does not exist; writing the first rows into a new file"
  manifest_src=/dev/null
fi
awk -v tag="$TAG" -v plats="$platforms" '
  BEGIN { n = split(plats, p, " ") }
  {
    if ($0 !~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*$/) {
      for (i = 1; i <= n; i++) if ($1 == tag && $2 == p[i]) next
    }
    print
  }' "$manifest_src" > "$NEW_MANIFEST"
manifest_rows >> "$NEW_MANIFEST"

echo
echo "=== tools/plugin_versions.txt rows for $TAG ==="
manifest_rows

if [[ "$UPDATE_MANIFEST" == 1 ]]; then
  echo
  echo "=== diff ==="
  diff -u "$manifest_src" "$NEW_MANIFEST" || true
  if [[ "$DRY_RUN" == 1 ]]; then
    say "[dry-run] tools/plugin_versions.txt not written"
  else
    cp -f "$NEW_MANIFEST" "$MANIFEST"
    say "updated tools/plugin_versions.txt; review the diff above and commit it"
  fi
fi

# ------------------------------------------------------------------ post-check
# Download what was actually published and check it the way a user will. The
# freshly computed manifest is passed explicitly because the committed one does
# not carry these rows yet (and never will without --update-manifest).
if [[ "$DRY_RUN" == 1 ]]; then
  say "[dry-run] skipping the post-check; after publishing, run"
  say "    tools/get_plugin.sh --release $TAG --dest build/release/verify"
elif printf '%s\n' "${ARCHES[@]}" | grep -qx "$host_arch"; then
  say "post-check: downloading the published linux-$host_arch asset back"
  "$REPO_ROOT/tools/get_plugin.sh" --release "$TAG" \
    --dest "$RELEASE_DIR/verify" --manifest "$NEW_MANIFEST" \
    || die "the published asset did not verify; the release is not usable yet"
else
  say "no linux-$host_arch asset in this release; skipping the post-check"
fi

echo
say "done: $TAG, $WARNINGS warning(s)"
