FROM ubuntu:24.04

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
  sudo \
  xxd \
  # needed to link jaxlib's LAPACK FFI kernels into the PJRT CPU plugin
  liblapack-dev \
  libblas-dev \
  # need to also need runtime and headers for clang++, so we need g++-14
  g++-14

# fail?
# RUN apt install -y \
#   emacs

RUN npm install -g @bazel/bazelisk

RUN useradd -m -s /bin/bash jozbee \
  && usermod -aG sudo jozbee \
  && echo "jozbee ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/jozbee \
  && chmod 0440 /etc/sudoers.d/jozbee

USER jozbee
WORKDIR /home/jozbee

RUN mkdir ~/.virtualenvs
RUN python3 -m venv ~/.virtualenvs/jax
RUN echo "source ~/.virtualenvs/jax/bin/activate" >> ~/.bashrc
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  jax \
  ipython

# temp?
USER root
RUN apt install -y \
  protobuf-compiler
USER jozbee
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  google-api-python-client

USER root
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
USER jozbee

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
