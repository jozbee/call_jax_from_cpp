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
	pjrt_runtime \
	bench bench_clean fixtures test_correct test_alloc

default: aot_jax2exec_example

all: lc0_example aot_example aot_jax2exec_example

clean: lc0_clean aot_clean aot_jax2exec_clean bench_clean

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

aot_jax2exec_example: src/examples/aot_jax2exec_example.cpp artifacts/pjrt_exec.o artifacts/runtime.o artifacts/jax_jax2exec.binpb artifacts/jax_jax2exec.json $(PJRT_PLUGIN_LIB)
	$(CXX) \
		-o aot_jax2exec_example src/examples/aot_jax2exec_example.cpp \
		artifacts/pjrt_exec.o artifacts/runtime.o \
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

artifacts/pjrt_exec.o: src/pjrt_exec/pjrt_exec.cpp src/pjrt_exec/pjrt_exec.hpp \
		| artifacts
	$(CXX) \
		-c -o artifacts/pjrt_exec.o src/pjrt_exec/pjrt_exec.cpp \
		$(CXXFLAGS) \
		-Wno-missing-field-initializers

artifacts/rt.o: src/pjrt_exec/rt.cpp src/pjrt_exec/rt.hpp | artifacts
	$(CXX) -c -o artifacts/rt.o src/pjrt_exec/rt.cpp $(CXXFLAGS)

artifacts/runtime.o: src/pjrt_exec/runtime.cpp src/pjrt_exec/runtime.hpp \
		src/pjrt_exec/pjrt_exec.hpp | artifacts
	$(CXX) \
		-c -o artifacts/runtime.o src/pjrt_exec/runtime.cpp \
		$(CXXFLAGS) \
		-Wno-missing-field-initializers

pjrt_clean:
	rm -f artifacts/pjrt_exec.o artifacts/runtime.o artifacts/rt.o

################
# pjrt runtime #
################

$(PJRT_PLUGIN_LIB):
	OS_NAME=$(UNAME_S) ./build_scripts/compile_xla_runtime.sh

pjrt_runtime: $(PJRT_PLUGIN_LIB)

#############
# benchmark #
#############

# The measurement spine: every runtime change is judged by `bench` output.
# `BENCH_ARGS` passes flags through, e.g.
#   make bench BENCH_ARGS="--fixture synth_solver --iterations 4000"
BENCH_ARGS ?= --fixture mpc_solver

bench: artifacts/bench
	./artifacts/bench $(BENCH_ARGS)

artifacts/bench: src/bench/bench_main.cpp src/bench/stats.hpp \
		src/bench/fixture.hpp src/bench/guard.hpp artifacts/pjrt_exec.o \
		artifacts/runtime.o artifacts/rt.o $(PJRT_PLUGIN_LIB)
	$(CXX) \
		-o artifacts/bench src/bench/bench_main.cpp \
		artifacts/pjrt_exec.o artifacts/runtime.o artifacts/rt.o \
		$(CXXFLAGS) \
		$(LDFLAGS) $(LDLIBS)

artifacts/probe: src/bench/probe_zerocopy.cpp src/bench/fixture.hpp \
		artifacts/pjrt_exec.o $(PJRT_PLUGIN_LIB)
	$(CXX) \
		-o artifacts/probe src/bench/probe_zerocopy.cpp artifacts/pjrt_exec.o \
		$(CXXFLAGS) -Wno-missing-field-initializers \
		$(LDFLAGS) $(LDLIBS)

# Exports the acceptance workload (mpc_solver) and its fixed-cost synthetic
# twin. Serialized executables embed target machine code, so this must run on
# the machine that will execute them.
fixtures:
	python3 tools/export_fixture.py --target all \
		--out-dir artifacts --assets-dir tests/assets/mpc

# Correctness only: a short run that still validates against the reference.
test_correct: artifacts/bench
	./artifacts/bench --api rt --fixture mpc_solver --iterations 20 --warmup 5
	./artifacts/bench --api rt --fixture synth_solver --iterations 20 --warmup 5
	./artifacts/bench --api legacy --fixture mpc_solver --iterations 20 --warmup 5

# Proves the steady-state call path does not allocate. The guard is preloaded,
# not linked, and the bench no-ops when it is absent.
artifacts/malloc_guard.so: tests/support/malloc_guard.c | artifacts
	$(CC) -shared -fPIC -O2 -o artifacts/malloc_guard.so \
		tests/support/malloc_guard.c -ldl

ifeq ($(UNAME_S),Darwin)
PRELOAD_VAR := DYLD_INSERT_LIBRARIES
else
PRELOAD_VAR := LD_PRELOAD
endif

test_alloc: artifacts/bench artifacts/malloc_guard.so
	$(PRELOAD_VAR)=./artifacts/malloc_guard.so \
		./artifacts/bench --api rt --fixture mpc_solver --iterations 200 --warmup 20

bench_clean:
	rm -f artifacts/bench artifacts/malloc_guard.so
