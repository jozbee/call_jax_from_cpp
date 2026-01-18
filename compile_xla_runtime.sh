#!/usr/bin/env bash

# make sure to use the correct xla commit, cf.
# https://github.com/jax-ml/jax/blob/jax-v0.8.2/third_party/xla/revision.bzl
cd ./third_party/xla
./configure.py --backend=CPU --host_compiler=CLANG --lld_path=/usr/bin/ld
bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.11 //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
echo $(pwd)

# add lib prefix for compiler
cp bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so ../../artifacts/libpjrt_c_api_cpu_plugin.so

# might need to change the id of the shared library
# install_name_tool -id /Users/jozbee/work/call_jax_from_cpp/artifacts/libpjrt_c_api_cpu_plugin_darwin.dylib libpjrt_c_api_cpu_plugin_darwin.dylib
# otool -L libpjrt_c_api_cpu_plugin_darwin.dylib

# other needed headers
# cp xla/pjrt/c/pjrt_c_api.h ../../src/xla/.
# cp xla/pjrt/c/pjrt_c_api_cpu.h ../../src/xla/.
