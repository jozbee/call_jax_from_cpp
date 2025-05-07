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
  npm

RUN npm install -g @bazel/bazelisk

RUN mkdir ~/.virtualenvs
RUN python3 -m venv ~/.virtualenvs/jax
RUN echo "source ~/.virtualenvs/jax/bin/activate" >> ~/.bashrc
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  jax \
  ipython
