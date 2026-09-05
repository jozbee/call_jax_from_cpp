# docs: begin make-fragment
# pjrt_exec.mk -- drop-in fragment for a Make-based consumer.
#
# This project is vendored, not installed: you check it out (submodule, subtree
# or a plain copy), include this file, and build its three sources into your own
# tree. Nothing here links against the PJRT plugin -- it is dlopen-ed at run
# time -- so including this adds no link-time dependency beyond -ldl -lpthread.
#
#   PJRT_EXEC_DIR   ?= third_party/call_jax_from_cpp   # default: this file's dir
#   PJRT_EXEC_BUILD ?= build/pjrt_exec                 # where the .o files go
#   include $(PJRT_EXEC_DIR)/pjrt_exec.mk
#
#   controller: controller.o $(PJRT_EXEC_LIB)
#   	$(CXX) -o $@ $^ $(PJRT_EXEC_LDLIBS)
#   controller.o: controller.cpp
#   	$(CXX) $(PJRT_EXEC_CPPFLAGS) $(CXXFLAGS) -c -o $@ $<
#
# Defines PJRT_EXEC_CPPFLAGS, PJRT_EXEC_SRCS, PJRT_EXEC_OBJS, PJRT_EXEC_LIB and
# PJRT_EXEC_LDLIBS, plus the pattern rule that builds the objects. It sets no
# other variable and touches neither CPPFLAGS nor CXXFLAGS: your flags stay
# yours, and the C++ standard must be at least C++17.

# The location of this fragment, captured immediately: MAKEFILE_LIST keeps
# growing as your makefiles are read, so `$(lastword ...)` means something else
# by the time a deferred variable would expand it.
pjrt_exec_this := $(lastword $(MAKEFILE_LIST))
PJRT_EXEC_DIR   ?= $(patsubst %/,%,$(dir $(pjrt_exec_this)))
PJRT_EXEC_BUILD ?= $(PJRT_EXEC_DIR)/build

# Including a fragment must not steal your default goal: make would otherwise
# pick the library rule below, because it is the first explicit rule it reads.
# An empty .DEFAULT_GOAL means "not chosen yet", so restoring the empty value
# at the end of the file hands the choice back to the next target you define.
pjrt_exec_saved_goal := $(.DEFAULT_GOAL)

# Both include roots are needed: the public headers say #include "pjrt/..."
# and #include "nlohmann/json.hpp", which resolve under third_party/.
PJRT_EXEC_CPPFLAGS := -I$(PJRT_EXEC_DIR)/include -I$(PJRT_EXEC_DIR)/third_party

# Optional: compile in the plugin location for a binary that must find it
# without $PJRT_CPU_PLUGIN being set. Absolute, because a control binary is
# rarely started from the directory that built it.
ifneq ($(PJRT_EXEC_PLUGIN_PATH),)
PJRT_EXEC_CPPFLAGS += -DPJRT_EXEC_DEFAULT_PLUGIN_PATH='"$(abspath $(PJRT_EXEC_PLUGIN_PATH))"'
endif

PJRT_EXEC_SRCS := $(addprefix $(PJRT_EXEC_DIR)/src/pjrt_exec/,runtime.cpp rt.cpp isa.cpp)
PJRT_EXEC_OBJS := $(patsubst $(PJRT_EXEC_DIR)/%.cpp,$(PJRT_EXEC_BUILD)/%.o,$(PJRT_EXEC_SRCS))
PJRT_EXEC_LIB  := $(PJRT_EXEC_BUILD)/lib/libpjrt_exec.a

# dl for dlopen of the plugin, pthread because XLA's CPU runtime starts threads.
PJRT_EXEC_LDLIBS := -ldl -lpthread

$(PJRT_EXEC_BUILD)/%.o: $(PJRT_EXEC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PJRT_EXEC_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c -o $@ $<

$(PJRT_EXEC_LIB): $(PJRT_EXEC_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $(PJRT_EXEC_OBJS)

# Header dependencies for the fragment's own objects only; yours are your
# business.
-include $(PJRT_EXEC_OBJS:.o=.d)

.DEFAULT_GOAL := $(pjrt_exec_saved_goal)
# docs: end make-fragment
