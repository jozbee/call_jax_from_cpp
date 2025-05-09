FROM --platform=linux/arm64 ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt update -y && apt upgrade -y && apt install -y \
  build-essential \
  clang \
  clang-format \
  cmake \
  git \
  python3 \
  python3-pip \
  python3-venv \
  npm \
  # need to also need runtime and headers for clang++, so we need g++-14
  g++-14

# fail?
# RUN apt install -y \
#   emacs

RUN npm install -g @bazel/bazelisk

RUN mkdir ~/.virtualenvs
RUN python3 -m venv ~/.virtualenvs/jax
RUN echo "source ~/.virtualenvs/jax/bin/activate" >> ~/.bashrc
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  jax \
  ipython

# temp?
RUN apt install -y \
  protobuf-compiler
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  google-api-python-client
