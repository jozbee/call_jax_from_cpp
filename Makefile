# for building examples that use the PJRT C API

CXX      := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -I.
LDFLAGS  := -L./artifacts -Wl,-rpath,./artifacts
LDLIBS   := -lpjrt_c_api_cpu_plugin

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
PJRT_PLUGIN_LIB := artifacts/libpjrt_c_api_cpu_plugin.dylib
else
PJRT_PLUGIN_LIB := artifacts/libpjrt_c_api_cpu_plugin.so
endif

.PHONY: \
	default all clean \
	lc0_clean aot_clean aot_jax2exec_clean \
	pjrt_clean jax_jax2exec_clean \
	pjrt_runtime

default: aot_jax2exec_example

all: lc0_example aot_example aot_jax2exec_example

clean: lc0_clean aot_clean aot_jax2exec_clean

artifacts:
	mkdir -p artifacts

#######
# lc0 #
#######

lc0_example: src/examples/lc0_example.cpp artifacts/pjrt.o
	$(CXX) \
		-o lc0_example src/examples/lc0_example.cpp artifacts/pjrt.o \
		$(CXXFLAGS)

artifacts/pjrt.o: src/lc0/pjrt.cc | artifacts
	$(CXX) -c $(CXXFLAGS) -o artifacts/pjrt.o src/lc0/pjrt.cc

lc0_clean:
	rm -f lc0_example
	rm -f artifacts/pjrt.o

#######
# aot #
#######

aot_example: src/examples/aot_example.cpp artifacts/jax_example.binpb artifacts/jax_example_exec.binpb $(PJRT_PLUGIN_LIB)
	$(CXX) \
		-o aot_example src/examples/aot_example.cpp \
		$(CXXFLAGS) \
		$(LDFLAGS) $(LDLIBS)

artifacts/jax_example.binpb artifacts/jax_example.pb artifacts/jax_example.hlo artifacts/jax_example_comp_opt.binpb artifacts/jax_example_comp_opt.pb artifacts/jax_example_exec.binpb: src/examples/jax_bare_metal.py | artifacts
	python3 src/examples/jax_bare_metal.py

aot_clean:
	rm -f aot_example
	rm -f artifacts/jax_example.binpb
	rm -f artifacts/jax_example.pb
	rm -f artifacts/jax_example.hlo
	rm -f artifacts/jax_example_comp_opt.binpb
	rm -f artifacts/jax_example_comp_opt.pb
	rm -f artifacts/jax_example_exec.binpb

################
# aot jax2exec #
################

aot_jax2exec_example: src/examples/aot_jax2exec_example.cpp artifacts/pjrt_exec.o artifacts/jax_jax2exec.binpb artifacts/jax_jax2exec.json $(PJRT_PLUGIN_LIB)
	$(CXX) \
		-o aot_jax2exec_example src/examples/aot_jax2exec_example.cpp artifacts/pjrt_exec.o \
		$(CXXFLAGS) \
		$(LDFLAGS) $(LDLIBS)

artifacts/jax_jax2exec.binpb artifacts/jax_jax2exec.json: src/examples/jax_jax2exec.py | artifacts
	python3 src/examples/jax_jax2exec.py

aot_jax2exec_clean: pjrt_clean jax_jax2exec_clean
	rm -f aot_jax2exec_example

jax_jax2exec_clean:
	rm -f artifacts/jax_jax2exec.binpb
	rm -f artifacts/jax_jax2exec.json

#############
# pjrt_exec #
#############

artifacts/pjrt_exec.o: src/pjrt_exec/pjrt_exec.cpp | artifacts
	$(CXX) \
		-c -o artifacts/pjrt_exec.o src/pjrt_exec/pjrt_exec.cpp \
		$(CXXFLAGS) \
		-Wno-missing-field-initializers

pjrt_clean:
	rm -f artifacts/pjrt_exec.o

################
# pjrt runtime #
################

$(PJRT_PLUGIN_LIB):
	OS_NAME=$(UNAME_S) ./build_scripts/compile_xla_runtime.sh

pjrt_runtime: $(PJRT_PLUGIN_LIB)
