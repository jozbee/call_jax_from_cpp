# syntax=docker/dockerfile:1.7
#
# Build environment for call_jax_from_cpp.
#
# Stages:
#   base            toolchain: clang, cmake, uv, bazelisk, LAPACK, doxygen
#   python          the project's virtualenv at /opt/venv, keyed on uv.lock
#   plugin-none     an empty plugin directory (bootstrap: no plugin yet)
#   plugin-prebuilt the released PJRT CPU plugin, downloaded and checksummed
#   plugin-source   the PJRT CPU plugin built from the XLA fork with bazel
#   plugin          whichever of the three PLUGIN_SOURCE selects
#   plugin-export   scratch image holding just the release files, for buildx
#                   --output type=local
#   dev             the final image: venv + plugin + workspace
#
# Nothing links against the plugin; PJRT_CPU_PLUGIN tells the runtime where
# it is.

ARG PLUGIN_SOURCE=none

########################################  base
FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld g++ make cmake ninja-build pkg-config \
      git curl ca-certificates unzip zip xxd file gdb strace sudo \
      python3 python3-venv \
      doxygen graphviz shellcheck \
      util-linux procps \
      liblapack-dev libblas-dev liblapack3 libblas3 \
    && rm -rf /var/lib/apt/lists/*

# uv, pinned. Keep in step with the version mise provides on the host.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /usr/local/bin/

# bazelisk reads .bazelversion from the XLA tree, so bazel itself is not pinned
# here. TARGETARCH is amd64/arm64, which matches bazelisk's asset names.
ARG TARGETARCH
ARG BAZELISK_VERSION=v1.29.0
RUN curl -fsSL -o /usr/local/bin/bazel \
      "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${TARGETARCH}" \
    && chmod 0755 /usr/local/bin/bazel

# uid/gid 1000 matches the host user, so bind-mounted files stay writable.
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd -g 1000 dev && useradd -m -u 1000 -g 1000 -s /bin/bash dev \
    && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev \
    && mkdir -p /opt/venv /opt/pjrt && chown -R dev:dev /opt/venv /opt/pjrt

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

########################################  python
FROM base AS python
USER dev
WORKDIR /build
# uv.loc[k] copies the lock only when it exists; a fresh tree has none.
COPY --chown=dev:dev pyproject.toml uv.loc[k] /build/
COPY --chown=dev:dev python /build/python
COPY --chown=dev:dev README.md LICENSE /build/
# The system python3, so the venv's interpreter symlink stays valid in the
# stages that copy /opt/venv.
RUN if [ -f uv.lock ]; then \
      uv sync --locked --all-groups --extra docs --python /usr/bin/python3; \
    else \
      uv sync --all-groups --extra docs --python /usr/bin/python3; \
    fi

########################################  plugin variants
FROM base AS plugin-none
RUN install -d -o dev -g dev /opt/pjrt \
    && echo 'no plugin baked into this image; run `make plugin` or mount one' \
       > /opt/pjrt/README.txt

FROM base AS plugin-prebuilt
COPY tools/get_plugin.sh tools/plugin_versions.txt /tmp/tools/
COPY versions.env /tmp/
RUN /tmp/tools/get_plugin.sh --dest /opt/pjrt --versions-file /tmp/versions.env \
    --manifest /tmp/tools/plugin_versions.txt

FROM base AS plugin-source
USER dev
ARG XLA_REPO=https://github.com/jozbee/xla.git
ARG XLA_REF=cjfc/jax-v0.11.0
ARG BUILD_MODE=opt
RUN git clone --depth 1 --branch "${XLA_REF}" "${XLA_REPO}" /home/dev/xla \
    || git clone --filter=blob:none "${XLA_REPO}" /home/dev/xla \
       && git -C /home/dev/xla checkout --detach "${XLA_REF}"
COPY --chown=dev:dev tools/build_plugin.sh /home/dev/build_plugin.sh
# BuildKit creates a cache mount's missing parents as root, and bazelisk
# writes ~/.cache/bazelisk beside the mount; created as dev first, the mount
# lands inside a directory dev owns.
RUN mkdir -p /home/dev/.cache
RUN --mount=type=cache,id=bazel-${TARGETARCH},target=/home/dev/.cache/bazel,uid=1000,gid=1000,sharing=locked \
    XLA_DIR=/home/dev/xla OUT_DIR=/opt/pjrt BUILD_MODE="${BUILD_MODE}" \
    /home/dev/build_plugin.sh

FROM plugin-${PLUGIN_SOURCE} AS plugin

# Just the three release files, for `buildx --target plugin-export
# --output type=local,dest=DIR`.
FROM scratch AS plugin-export
COPY --from=plugin /opt/pjrt/ /

########################################  dev
FROM base AS dev
COPY --from=python /opt/venv /opt/venv
COPY --from=plugin /opt/pjrt /opt/pjrt
RUN chown -R dev:dev /opt/venv /opt/pjrt
# The cache-mount hazard again: compose mounts the bazel-cache volume at
# /home/dev/.cache/bazel, and docker creates a mount point's missing parents
# as root, leaving dev unable to write ~/.cache/uv or ~/.cache/bazelisk.
RUN install -d -o dev -g dev /home/dev/.cache /home/dev/.cache/bazel
ENV PJRT_CPU_PLUGIN=/opt/pjrt/libpjrt_c_api_cpu_plugin.so \
    CJFC_IN_CONTAINER=1
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh
USER dev
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
