# e.g,
# clang++ -I. -Wl,-rpath,./artifacts -L/workspace/artifacts \
#  -lpjrt_c_api_cpu_plugin -o aot_example src/aot_example.cpp

# Compiler and compiler flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Target executable
TARGET = aot_example

# Source files
SRCS = src/aot_example.cpp

# Include directories
INCLUDES = -I.

# Libraries and paths
LDFLAGS = -L./artifacts
LIBS = -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin

# Build rules
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)

.PHONY: all clean
