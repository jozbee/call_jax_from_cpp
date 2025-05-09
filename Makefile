# e.g,
# clang++ -I. -Wl,-rpath,./artifacts -L/workspace/artifacts \
#  -lpjrt_c_api_cpu_plugin -o aot_example src/aot_example.cpp

.PHONY: all clean

all: lc0_example

lc0_example: src/lc0_example.cpp pjrt
	clang++ -std=c++17 -Wall -Wextra -g -I. -o lc0_example src/lc0_example.cpp artifacts/pjrt.o

pjrt: src/pjrt.cc
	clang++ -c -std=c++17 -Wall -Wextra -g -I. -o artifacts/pjrt.o src/pjrt.cc

clean:
	rm -f lc0_example
	rm -f artifacts/pjrt.o

# # Compiler and compiler flags
# CXX = clang++
# CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# # Target executable
# TARGET = aot_example

# # Source files
# SRCS = src/aot_example.cpp

# # Include directories
# INCLUDES = -I.

# # Libraries and paths
# LDFLAGS = -L./artifacts
# LIBS = -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin

# # Build rules
# all: pjrt

# $(TARGET): $(SRCS)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# # Clean rule
# clean:
# 	rm -f $(TARGET)

# .PHONY: all clean
