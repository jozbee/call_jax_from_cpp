# Calling JAX from C++

TLDR; we compile a [JAX](https://github.com/jax-ml/jax) function to a serialized executable, and we call this via the [PJRT](https://openxla.org/xla/pjrt) C API.

## Motivation

I feel like it is easier to develop and debug numerics in Python and NumPy.
The problem is that Python is slow, and NumPy does not support automatic differentiation, which is helpful for optimization problems.
JAX helps solve these problems.

I am also interested in robotics control, and for real-time performance and integration with [ros2_control](https://github.com/ros-controls/ros2_control), I need my algorithms to be naturally embedded into a C++ program.
For hacking a project quickly, I don't want to rewrite the JAX programs in boutique C++ for extra performance.
JAX also solves this problem, but because this use case is niche, the documentation is poor.
This project provides some guiding examples for Float64 CPU integration.

## Examples

This project provides 3 examples:

1. [`lc0`](https://github.com/LeelaChessZero/lc0) implementation: we demonstrate how to perform JIT computation of a JAX program using some C++ [PJRT wrappers](https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla).

2. [`pjrt_c_api`](https://github.com/openxla/xla/blob/f47564c12397631f240de1ca44279fdf20b66d88/xla/pjrt/c/pjrt_c_api.h): we load and execute a program compiled AOT, only using the `pjrt_c_api` provided by [`xla`](https://github.com/openxla/xla).
Note that our example does not clean up after itself.
It simply shows that we can execute an AOT compiled program in C++.

3. A simple PJRT C++ wrapper: we implement some simple wrappers around the `pjrt_c_api` that performs cleanup and makes code very easy to write.
Note that we severely underexpose the flexibility of the PJRT API.
My applications find this to be mostly sufficient.

## Usage

The compiled examples only _mildly_ depend on XLA.
(Mildly after initial setup...)
They depend on a shared library and a couple of header files: `pjrt_c_api_cpu_plugin.so` (runtime), `pjrt_c_api.h` (API), and `pjrt_c_api_cpu.h` (API struct getter).
This requires the [`bazel`](https://github.com/bazelbuild/bazel) build system.
See the script `compile_xla_runtime.sh` for hints and the correct `bazel` target.

> **NOTE.**
> Make sure that you compile the PJRT runtime to be compatible with JAX.
> This note is probably immaterial if you are running an up-to-date version of JAX, but on the safe side, I've pinned an XLA commit that is compatible with JAX v0.9.0.1.
> Cf. [`jax/third_party/xla/revision.bzl`](https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/third_party/xla/revision.bzl).

For compiling the examples, see the included `Makefile`.
These scripts depend on some byproducts produced from JAX.
The corresponding Python scripts are found in `src/examples`.
You should probably `pip install -e .` in base directory to install a couple of `jax2exec` modules to get this scripts to run.

## References

Note that I stole the name of this repository from `joaospinto`...

- https://github.com/jax-ml/jax/discussions/22184
- https://github.com/joaospinto/call_jax_from_cpp/tree/main
- https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla
- https://github.com/gomlx/gopjrt/tree/main
