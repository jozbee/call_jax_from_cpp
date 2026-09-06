# Build the pjrt_exec library, the examples, the C++ tests and the benchmark.
#
# Two facts shape this file:
#
#   1. NOTHING links against the PJRT plugin. It is dlopen-ed at run time, so
#      the library links only -ldl and -lpthread and a plain `make` never
#      touches bazel or the network. `make plugin` is a separate errand that
#      only the RUN targets care about.
#   2. Every pinned version lives in versions.env. This file greps it rather
#      than repeating a number that would then rot.
#
# Everything below is overridable from the command line, e.g.
#   make CXX=g++ BUILD_DIR=/tmp/b bench BENCH_ARGS="--fixture trajopt --iterations 500"

# The default goal is `all` (library + examples) rather than `help`, because
# the documentation, the Dockerfile and CI all say "run make" and mean "build
# the thing". `make help` lists the rest and is one word away.
.DEFAULT_GOAL := all

# A failed recipe must not leave a half-written .o or .a behind to be picked up
# as up to date by the next run.
.DELETE_ON_ERROR:

# ----------------------------------------------------------------- toolchain

# `CXX ?= clang++` would be a no-op: make already defines CXX as g++, so the
# override has to be conditional on the value still being make's own default.
# An explicit `make CXX=g++` (or a CXX in the environment) still wins.
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
#
# One grep per variable out of versions.env, which is the single source of
# truth for every pin in this project. `:=` so the greps run once, not once per
# reference. Reported by `make print-config`.
VERSIONS_ENV   := versions.env
version_of      = $(patsubst $(1)=%,%,$(shell grep -m1 '^$(1)=' $(VERSIONS_ENV) 2>/dev/null))
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
# plugin_probe is a developer tool, not part of the deliverable; build it only
# when its source is present so a trimmed checkout still builds.
TOOL_BINS    := $(if $(wildcard tools/plugin_probe.cpp),plugin_probe)

# A binary is one main object plus the library. `main_<binary>` names that
# object; the link rule below reads it back through secondary expansion, which
# is what keeps this to a single rule instead of one per binary.
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

# Kept for `make print-config`: the plugin-path define below carries nested
# quotes that no single echo can reproduce faithfully, so it is reported on its
# own line instead.
CPPFLAGS_DISPLAY := $(CPPFLAGS)

# A binary run from any directory has to find the plugin, so the build-time
# default is absolute. $PJRT_CPU_PLUGIN and RuntimeOptions::plugin_path still
# win over it at run time.
CPPFLAGS += -DPJRT_EXEC_DEFAULT_PLUGIN_PATH='"$(abspath $(PLUGIN))"'

# -MMD -MP: one .d per object listing the headers it read, with a phony target
# for each of them so a deleted header does not wedge the build.
DEPFLAGS := -MMD -MP

# Examples include their own shared helpers as "common/cli.hpp"; bench and the
# C++ tests get the same courtesy for their local headers. bench and the tests
# also reuse examples/common (the CLI parser, the JSON report writer and the
# host-environment probe are the same code the examples show off), so they get
# -Iexamples too rather than a second copy of all three.
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
        test test-slow test-rt test-alloc check rt-check plugin plugin-source \
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

# Nothing links against the guard: it is preloaded, and the binaries resolve
# its markers with dlsym and no-op when they are absent.
$(GUARD): tests/support/malloc_guard.c | $(LIB_DIR)
	$(CC) -shared -fPIC -O2 -Wall -Wextra -o $@ $< -ldl

-include $(ALL_OBJS:.o=.d)

# -------------------------------------------------------------------- plugin
#
# There is no official prebuilt CPU PJRT C-API plugin (jaxlib links its CPU
# client statically and never exports GetPjrtApi), so this project publishes
# its own. Neither target is a prerequisite of a build: only running needs a
# plugin.

plugin:  ## Download the prebuilt PJRT CPU plugin (sha256-verified)
	tools/get_plugin.sh --dest $(PLUGIN_DIR)

plugin-source:  ## Build the PJRT CPU plugin from the XLA fork (bazel, 30-60 min)
	tools/build_plugin.sh --out $(PLUGIN_DIR)

# Advice, not a gate: a plugin supplied through $PJRT_CPU_PLUGIN is just as
# valid as one in build/plugin, and CI does exactly that.
plugin-hint:
	@if [ ! -f "$(PLUGIN)" ] && [ -z "$$PJRT_CPU_PLUGIN" ]; then \
	  echo "note: no plugin at $(PLUGIN), and PJRT_CPU_PLUGIN is unset."; \
	  echo "      make plugin          download the prebuilt one ($(PLUGIN_RELEASE))"; \
	  echo "      make plugin-source   build it from the XLA fork"; \
	  echo "      export PJRT_CPU_PLUGIN=/path/to/libpjrt_c_api_cpu_plugin.so"; \
	fi

# -------------------------------------------------------------------- export
#
# Serialized executables embed target machine code, so these must be produced
# on the machine that will run them; that is why artifacts/ is gitignored and
# why every run target depends on the export rather than on a committed file.
#
# The two names below are what the export scripts write. They are variables so
# that a rename is a one-line fix here instead of a rule that re-exports on
# every invocation.
EXPORT_BASIC   ?= basic
EXPORT_TRAJOPT ?= trajopt

BASIC_ARTIFACT   := $(ARTIFACTS_DIR)/$(EXPORT_BASIC).binpb
TRAJOPT_ARTIFACT := $(ARTIFACTS_DIR)/$(EXPORT_TRAJOPT).binpb
EXPORTER_SRCS    := $(wildcard python/jax2exec/*.py)

# PYTHONPATH so an interpreter that has jax but not this package installed
# (`make export PYTHON=python3`) still finds jax2exec in the source tree.
export_env = PYTHONPATH=$(CURDIR)/python$(if $(PYTHONPATH),:$(PYTHONPATH))

$(BASIC_ARTIFACT): examples/01_basic/export.py $(EXPORTER_SRCS) | $(ARTIFACTS_DIR)
	$(export_env) $(PYTHON) examples/01_basic/export.py --out $(ARTIFACTS_DIR)

$(TRAJOPT_ARTIFACT): examples/02_trajopt/export.py $(EXPORTER_SRCS) | $(ARTIFACTS_DIR)
	$(export_env) $(PYTHON) examples/02_trajopt/export.py --out $(ARTIFACTS_DIR) --cases 4

export: $(BASIC_ARTIFACT) $(TRAJOPT_ARTIFACT)  ## Export the example artifacts with JAX

# ---------------------------------------------------------------------- runs

run-examples: examples export guard plugin-hint | $(REPORTS)  ## Run all four examples end to end
	$(BIN_DIR)/example_01_basic
	$(BIN_DIR)/example_01_basic --debug
	$(BIN_DIR)/example_02_trajopt --iterations 500 --json $(REPORTS)/trajopt.json
	$(BIN_DIR)/example_03_minimal artifacts/basic 1000 1000
	$(PRELOAD_VAR)=$(abspath $(GUARD)) \
	  $(BIN_DIR)/example_04_realtime --iterations 1000 --json $(REPORTS)/realtime.json

# `bench` builds and runs, because a benchmark that was built but not run tells
# you nothing. The load average goes with the numbers: a concurrent build does
# not add noise to a tail measurement, it invalidates it.
bench: $(BIN_DIR)/bench export plugin-hint  ## Build and run the benchmark (see BENCH_ARGS)
	@echo "load average: $$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
	$(BIN_DIR)/bench $(BENCH_ARGS)

# --------------------------------------------------------------------- tests

test: examples tests-cpp guard plugin-hint  ## Build everything, then run the test suite
	$(PYTEST) $(PYTEST_ARGS)

check: test  ## Alias for `test`

test-slow: examples tests-cpp guard plugin-hint  ## The suite including the long campaigns
	$(PYTEST) --runslow $(PYTEST_ARGS)

# Only meaningful on a tuned, idle host; rt-check reports whether this is one.
test-rt: examples tests-cpp guard plugin-hint  ## Real-time gates (tuned, idle host only)
	@echo "load average: $$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
	$(PYTEST) -m rt $(PYTEST_ARGS)

# Proves the wrapper's own steady-state allocations are zero. The count that
# matters is `self`: XLA's thunk runtime allocates ~15,400 times per call
# inside the plugin, and that is not reachable from here.
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

# clang-format has no config file in this tree on purpose; --fallback-style is
# what picks the house style, and adding a .clang-format later changes nothing
# here because -style=file prefers it.
format:  ## Format the C++ and Python sources in place
	@if command -v clang-format >/dev/null 2>&1; then \
	  clang-format -i -style=file --fallback-style=Google $(FORMAT_CXX); \
	else \
	  echo "clang-format not found; skipping the C++ sources"; \
	fi
	$(RUFF) format .
	$(RUFF) check --fix .

# The plugin is expensive to obtain (a download, or an hour of bazel) and is
# not a build product of this tree, so `clean` leaves it alone. distclean does
# not.
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
