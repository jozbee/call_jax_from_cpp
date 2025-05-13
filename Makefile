# for building examples that use the PJRT C API

.PHONY: \
	all clean \
	lc0_clean aot_clean aot_api_clean aot_jax2exec_clean

all: aot_jax2exec_example

clean: aot_jax2exec_clean

#######
# lc0 #
#######

lc0_example: src/examples/lc0_example.cpp pjrt
	clang++ \
		-o lc0_example src/examples/lc0_example.cpp artifacts/pjrt.o \
	 -std=c++17 -Wall -Wextra -O3 \
	 -I.

pjrt: src/lc0/pjrt.cc
	clang++ -c -std=c++17 -Wall -Wextra -O3 -I. -o artifacts/pjrt.o src/lc0/pjrt.cc

lc0_clean:
	rm -f lc0_example
	rm -f artifacts/pjrt.o

#######
# aot #
#######

aot_example: src/examples/aot_example.cpp
	clang++ \
		-o aot_example src/examples/aot_example.cpp \
		-std=c++17 -Wall -Wextra -O3 \
		-I. \
		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_darwin
#		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_linux

aot_clean:
	rm -f aot_example

################
# aot jax2exec #
################

aot_jax2exec_example: src/examples/aot_jax2exec_example.cpp pjrt_exec
	clang++ \
		-o aot_jax2exec_example src/examples/aot_jax2exec_example.cpp artifacts/pjrt_exec.o \
		-std=c++17 -Wall -Wextra -O3 \
		-I. \
		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_darwin
#		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_linux

aot_jax2exec_clean:
	rm -f aot_jax2exec_example
	rm -f artifacts/pjrt_exec.o
