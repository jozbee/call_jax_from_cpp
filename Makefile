# Build the pjrt_exec library, the examples, the C++ tests and the benchmark.
#
# Nothing links against the PJRT plugin: it is dlopen-ed at run time, so a
# plain `make` never touches bazel or the network, and `make plugin` is a
# separate errand that only the run targets need. Every pinned version comes
# from versions.env. Every variable below is overridable from the command line:
#   make CXX=g++ BUILD_DIR=/tmp/b bench BENCH_ARGS="--fixture trajopt --iterations 500"

# `all`, not `help`: the docs, the Dockerfile and CI say `make` and mean build.
.DEFAULT_GOAL := all

# A failed recipe must not leave a half-written .o behind as up to date.
.DELETE_ON_ERROR:

# ----------------------------------------------------------------- toolchain

# `CXX ?= clang++` would be a no-op: make predefines CXX as g++, so the default
# is replaced only while it is still make's own. An explicit CXX still wins.
ifeq ($(origin CXX),default)
CXX := clang++
endif
CC ?= cc
AR ?= ar
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS += -Iinclude -Ithird_party
LDLIBS   += -ldl -lpthread

# uv comes from mise and is not on PATH in a non-interactive shell; override
# these two when that bites (`make PYTHON=python3 PYTEST=pytest test`).
PYTHON ?= uv run python
PYTEST ?= uv run pytest
RUFF   ?= uv run ruff

PYTEST_ARGS ?=
BENCH_ARGS  ?= --fixture trajopt --iterations 2000

# ------------------------------------------------------------------- layout

BUILD_DIR     ?= build
ARTIFACTS_DIR ?= artifacts
PLUGIN_DIR    ?= $(BUILD_DIR)/plugin
PLUGIN        ?= $(PLUGIN_DIR)/libpjrt_c_api_cpu_plugin.so

OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin

LIB      := $(LIB_DIR)/libpjrt_exec.a
GUARD    := $(LIB_DIR)/malloc_guard.so
REPORTS  := $(ARTIFACTS_DIR)/reports

# ----------------------------------------------------------------- versions

# `:=` so each value is read from versions.env once, not once per reference.
VERSIONS_ENV   := versions.env
version_of      = $(shell sed -n 's/^$(1)=//p' $(VERSIONS_ENV) 2>/dev/null | head -1)
JAX_VERSION    := $(call version_of,JAX_VERSION)
PLUGIN_RELEASE := $(call version_of,PLUGIN_RELEASE)

# ------------------------------------------------------------------- sources

LIB_SRCS := \
  src/pjrt_exec/runtime.cpp \
  src/pjrt_exec/rt.cpp \
  src/pjrt_exec/isa.cpp
LIB_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

EXAMPLE_BINS := example_01_basic example_02_trajopt example_03_minimal \
                example_04_realtime
TEST_BINS    := fn_info test_debug_checks test_load_errors test_latency \
                test_guard_selftest
BENCH_BIN    := bench
# A developer tool; a trimmed checkout may not carry its source.
TOOL_BINS    := $(if $(wildcard tools/plugin_probe.cpp),plugin_probe)

# One main object plus the library per binary. The link rule reads
# `main_<binary>` back through secondary expansion: one rule, not one each.
main_example_01_basic    := $(OBJ_DIR)/examples/01_basic/basic.o
main_example_02_trajopt  := $(OBJ_DIR)/examples/02_trajopt/trajopt.o
main_example_03_minimal  := $(OBJ_DIR)/examples/03_minimal/minimal.o
main_example_04_realtime := $(OBJ_DIR)/examples/04_realtime/realtime.o
main_bench               := $(OBJ_DIR)/bench/bench_main.o
main_fn_info             := $(OBJ_DIR)/tests/cpp/fn_info.o
main_test_debug_checks   := $(OBJ_DIR)/tests/cpp/test_debug_checks.o
main_test_load_errors    := $(OBJ_DIR)/tests/cpp/test_load_errors.o
main_test_latency        := $(OBJ_DIR)/tests/cpp/test_latency.o
main_test_guard_selftest := $(OBJ_DIR)/tests/cpp/test_guard_selftest.o
main_plugin_probe        := $(OBJ_DIR)/tools/plugin_probe.o

ALL_BINS     := $(EXAMPLE_BINS) $(TEST_BINS) $(BENCH_BIN) $(TOOL_BINS)
EXAMPLE_OBJS := $(foreach b,$(EXAMPLE_BINS),$(main_$(b)))
TEST_OBJS    := $(foreach b,$(TEST_BINS),$(main_$(b)))
MAIN_OBJS    := $(foreach b,$(ALL_BINS),$(main_$(b)))
ALL_OBJS     := $(LIB_OBJS) $(MAIN_OBJS)

EXAMPLE_PATHS := $(addprefix $(BIN_DIR)/,$(EXAMPLE_BINS))
TEST_PATHS    := $(addprefix $(BIN_DIR)/,$(TEST_BINS))
TOOL_PATHS    := $(addprefix $(BIN_DIR)/,$(TOOL_BINS))

# --------------------------------------------------------- compilation flags

# For `make print-config`: the plugin-path define below carries nested quotes
# that echo cannot reproduce, so it is reported on its own line.
CPPFLAGS_DISPLAY := $(CPPFLAGS)

# Absolute, so a binary run from any directory finds the plugin.
# $PJRT_CPU_PLUGIN and RuntimeOptions::plugin_path still win at run time.
CPPFLAGS += -DPJRT_EXEC_DEFAULT_PLUGIN_PATH='"$(abspath $(PLUGIN))"'

# -MP: a phony target per header, so a deleted header does not wedge the build.
DEPFLAGS := -MMD -MP

# -Iexamples for all of them: the examples include their helpers as
# "common/cli.hpp", and bench and the tests reuse those helpers rather than
# carrying a copy.
$(EXAMPLE_OBJS):            CPPFLAGS += -Iexamples
$(main_bench):              CPPFLAGS += -Ibench -Iexamples
$(TEST_OBJS):               CPPFLAGS += -Itests -Iexamples -Ibench

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
PRELOAD_VAR := DYLD_INSERT_LIBRARIES
else
PRELOAD_VAR := LD_PRELOAD
endif

# --------------------------------------------------------------------- phony

.PHONY: all lib examples tests-cpp tools guard bench export run-examples \
        test test-slow test-rt test-alloc rt-check plugin plugin-source \
        docs docs-live docs-linkcheck docs-clean format clean distclean \
        print-config help plugin-hint

all: lib examples  ## Build the library and the examples (default)

lib: $(LIB)  ## Build build/lib/libpjrt_exec.a

examples: $(EXAMPLE_PATHS)  ## Build the four example binaries

tests-cpp: $(TEST_PATHS)  ## Build the C++ test binaries

tools: $(TOOL_PATHS)  ## Build the developer tools (plugin_probe)

guard: $(GUARD)  ## Build the preloadable allocation counter

# ------------------------------------------------------------------ building

$(BIN_DIR) $(LIB_DIR) $(ARTIFACTS_DIR) $(REPORTS):
	@mkdir -p $@

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

$(LIB): $(LIB_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $(LIB_OBJS)

.SECONDEXPANSION:
$(BIN_DIR)/%: $$(main_$$*) $(LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(main_$*) $(LIB) $(LDLIBS)

# Preloaded, never linked: the binaries find its markers with dlsym and no-op
# without it.
$(GUARD): tests/support/malloc_guard.c | $(LIB_DIR)
	$(CC) -shared -fPIC -O2 -Wall -Wextra -o $@ $< -ldl

-include $(ALL_OBJS:.o=.d)

# -------------------------------------------------------------------- plugin
#
# No official prebuilt CPU plugin exists (jaxlib never exports GetPjrtApi), so
# this project publishes its own. Only running needs one.

plugin:  ## Download the prebuilt PJRT CPU plugin (sha256-verified)
	tools/get_plugin.sh --dest $(PLUGIN_DIR)

plugin-source:  ## Build the PJRT CPU plugin from the XLA fork (bazel, 30-60 min)
	tools/build_plugin.sh --out $(PLUGIN_DIR)

# Advice, not a gate: CI supplies its plugin through $PJRT_CPU_PLUGIN.
plugin-hint:
	@if [ ! -f "$(PLUGIN)" ] && [ -z "$$PJRT_CPU_PLUGIN" ]; then \
	  echo "note: no plugin at $(PLUGIN), and PJRT_CPU_PLUGIN is unset."; \
	  echo "      make plugin          download the prebuilt one ($(PLUGIN_RELEASE))"; \
	  echo "      make plugin-source   build it from the XLA fork"; \
	  echo "      export PJRT_CPU_PLUGIN=/path/to/libpjrt_c_api_cpu_plugin.so"; \
	fi

# -------------------------------------------------------------------- export
#
# Artifacts embed machine code for the exporting host, so every run target
# depends on the export rather than on a committed file. The three .binpb
# names are what the export scripts write.
BASIC_ARTIFACT   := $(ARTIFACTS_DIR)/basic.binpb
TRAJOPT_ARTIFACT := $(ARTIFACTS_DIR)/trajopt.binpb
ARM_ARTIFACT     := $(ARTIFACTS_DIR)/arm.binpb
EXPORTER_SRCS    := $(wildcard python/jax2exec/*.py)

# PYTHONPATH so an interpreter that has jax but not this package installed
# (`make export PYTHON=python3`) still finds jax2exec in the source tree.
export_env = PYTHONPATH=$(CURDIR)/python$(if $(PYTHONPATH),:$(PYTHONPATH))

$(BASIC_ARTIFACT): examples/01_basic/export.py $(EXPORTER_SRCS) | $(ARTIFACTS_DIR)
	$(export_env) $(PYTHON) examples/01_basic/export.py --out $(ARTIFACTS_DIR)

$(TRAJOPT_ARTIFACT): examples/02_trajopt/export.py $(EXPORTER_SRCS) | $(ARTIFACTS_DIR)
	$(export_env) $(PYTHON) examples/02_trajopt/export.py --out $(ARTIFACTS_DIR) --cases 4

$(ARM_ARTIFACT): examples/05_ros2_control/export.py $(EXPORTER_SRCS) | $(ARTIFACTS_DIR)
	$(export_env) $(PYTHON) examples/05_ros2_control/export.py --out $(ARTIFACTS_DIR)

export: $(BASIC_ARTIFACT) $(TRAJOPT_ARTIFACT) $(ARM_ARTIFACT)  ## Export the example artifacts with JAX

# ---------------------------------------------------------------------- runs

run-examples: examples export guard plugin-hint | $(REPORTS)  ## Run all four examples end to end
	$(BIN_DIR)/example_01_basic
	$(BIN_DIR)/example_01_basic --debug
	$(BIN_DIR)/example_02_trajopt --iterations 500 --json $(REPORTS)/trajopt.json
	$(BIN_DIR)/example_03_minimal artifacts/basic 1000 1000
	$(PRELOAD_VAR)=$(abspath $(GUARD)) \
	  $(BIN_DIR)/example_04_realtime --iterations 1000 --json $(REPORTS)/realtime.json

# The load average goes with the numbers: a concurrent build does not add
# noise to a tail measurement, it invalidates it.
bench: $(BIN_DIR)/bench export plugin-hint  ## Build and run the benchmark (see BENCH_ARGS)
	@echo "load average: $$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
	$(BIN_DIR)/bench $(BENCH_ARGS)

# --------------------------------------------------------------------- tests

test: examples tests-cpp guard plugin-hint  ## Build everything, then run the test suite
	$(PYTEST) $(PYTEST_ARGS)

test-slow: examples tests-cpp guard plugin-hint  ## The suite including the long campaigns
	$(PYTEST) --runslow $(PYTEST_ARGS)

test-rt: examples tests-cpp guard plugin-hint  ## Real-time gates (tuned, idle host only)
	@echo "load average: $$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
	$(PYTEST) -m rt $(PYTEST_ARGS)

# `--alloc-gate self`: the wrapper's own count must be zero. XLA's thunk
# runtime allocates thousands of times per call inside the plugin, out of reach.
test-alloc: guard $(BIN_DIR)/bench export plugin-hint | $(REPORTS)  ## Gate: zero wrapper allocations per call
	$(PRELOAD_VAR)=$(abspath $(GUARD)) $(BIN_DIR)/bench \
	  --iterations 200 --warmup 20 --alloc-gate self --require-guard \
	  --json $(REPORTS)/alloc_census.json

rt-check:  ## Audit this host's real-time settings (read-only)
	tools/rt_check.sh

# ---------------------------------------------------------------------- docs

docs:  ## Build the documentation (warnings are errors)
	$(MAKE) -C docs html

docs-live:  ## Serve the documentation, rebuilding as you edit
	$(MAKE) -C docs live

docs-linkcheck:  ## Verify every external link in the documentation
	$(MAKE) -C docs linkcheck

docs-clean:  ## Remove the built documentation
	$(MAKE) -C docs clean

# --------------------------------------------------------------- maintenance

FORMAT_CXX := $(wildcard include/pjrt_exec/*.hpp src/pjrt_exec/*.cpp \
                         examples/common/*.hpp examples/*/*.hpp \
                         examples/*/*.cpp bench/*.cpp bench/*.hpp \
                         tests/cpp/*.cpp tools/*.cpp \
                         tests/support/malloc_guard.c)

# No .clang-format in the tree: --fallback-style picks the house style, and
# -style=file would prefer a config file if one appeared.
format:  ## Format the C++ and Python sources in place
	@if command -v clang-format >/dev/null 2>&1; then \
	  clang-format -i -style=file --fallback-style=Google $(FORMAT_CXX); \
	else \
	  echo "clang-format not found; skipping the C++ sources"; \
	fi
	$(RUFF) format .
	$(RUFF) check --fix .

# The plugin is a download or an hour of bazel, not a build product: kept.
clean:  ## Remove objects, binaries and the library (keeps build/plugin)
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)

distclean: docs-clean  ## Remove everything generated, including the plugin and artifacts
	rm -rf $(BUILD_DIR) $(ARTIFACTS_DIR)

print-config:  ## Print the resolved configuration
	@echo "CXX             = $(CXX)"
	@echo "CXXFLAGS        = $(CXXFLAGS)"
	@echo "CPPFLAGS        = $(CPPFLAGS_DISPLAY)"
	@echo "default plugin  = $(abspath $(PLUGIN))   (-DPJRT_EXEC_DEFAULT_PLUGIN_PATH)"
	@echo "LDLIBS          = $(LDLIBS)"
	@echo "PYTHON          = $(PYTHON)"
	@echo "PYTEST          = $(PYTEST)"
	@echo "BUILD_DIR       = $(BUILD_DIR)"
	@echo "ARTIFACTS_DIR   = $(ARTIFACTS_DIR)"
	@echo "PLUGIN          = $(PLUGIN)"
	@echo "PRELOAD_VAR     = $(PRELOAD_VAR)"
	@echo "JAX_VERSION     = $(JAX_VERSION)      (versions.env)"
	@echo "PLUGIN_RELEASE  = $(PLUGIN_RELEASE)   (versions.env)"
	@echo "BENCH_ARGS      = $(BENCH_ARGS)"
	@echo "binaries        = $(ALL_BINS)"

help:  ## List the targets
	@echo "call_jax_from_cpp -- JAX $(JAX_VERSION), plugin release $(PLUGIN_RELEASE)"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /\t/' \
	  | awk -F'\t' '{printf "  %-16s %s\n", $$1, $$2}'
	@echo
	@echo "Variables: CXX CXXFLAGS BUILD_DIR ARTIFACTS_DIR PLUGIN PYTHON PYTEST BENCH_ARGS"
	@echo "           (make print-config shows the resolved values)"
