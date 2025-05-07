#!/usr/bin/env bash

cd /workspace/third_party/xla
./configure.py --backend=CPU --host_compiler=CLANG #--lld_path=/usr/bin/ld
bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.11 //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
echo $(pwd)

# add lib prefix for compiler
cp bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so ../../artifacts/libpjrt_c_api_cpu_plugin.so

cp xla/pjrt/c/pjrt_c_api.h ../../artifacts/.
cp xla/pjrt/c/pjrt_c_api_cpu.h ../../artifacts/.
