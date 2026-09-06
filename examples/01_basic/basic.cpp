/**
 * @file basic.cpp
 * @brief Load the artifact `examples/01_basic/export.py` wrote, call it once,
 *        and check the answer against arithmetic done here.
 *
 * The whole call path and nothing else: a `pjrt::Runtime`, a `pjrt::Function`,
 * a signature check, the inputs, `call()`, the outputs.  The flags, the
 * `key=value` reporting and the `--debug` demonstration are in `support.hpp`.
 * `02_trajopt` and `03_realtime` add timing and hardening on top of this.
 */
#include <cstddef>
#include <cstdio>
#include <exception>
#include <random>

#include "common/cli.hpp"
#include "pjrt_exec/runtime.hpp"
#include "support.hpp"

int main(int argc, char** argv) {
  try {
    const cjfc::Cli cli(argc, argv, {"artifact", "debug", "threads"},
                        basic::kUsage);
    if (cli.help()) {
      return 0;
    }
    const basic::Options options = basic::parse_options(cli);

    // docs: begin load
    pjrt::RuntimeOptions runtime_options;
    runtime_options.worker_threads = static_cast<int>(options.threads);
    pjrt::Runtime runtime(runtime_options);  // one per process

    pjrt::FunctionOptions function_options;
    function_options.debug = options.debug;
    function_options.check_values = options.debug;
    pjrt::Function f(runtime, options.artifact, function_options);  // load once
    // docs: end load

    // docs: begin signature
    basic::print_signature(runtime, f);

    // Names resolve to indices once, here, because `find_input` is a linear
    // scan over strings.  A loop indexes; only startup looks names up.
    const std::size_t a_index = basic::require_input(f, "A");
    const std::size_t b_index = basic::require_input(f, "b");
    basic::require_shape(f, a_index, b_index);

    // The order comes from the artifact; nothing below assumes it is 4.
    const std::size_t n = f.input_numel(b_index);
    // docs: end signature

    // docs: begin call
    double* A = f.input<double>(a_index);  // the arena XLA reads, not a copy
    double* b = f.input<double>(b_index);

    // A fixed seed, so two runs print the same numbers and a difference in the
    // output is a difference in the computation.
    std::mt19937_64 rng(20240517);
    for (std::size_t i = 0; i < n * n; ++i) {
      A[i] = basic::next_uniform(rng);  // row-major, as the sidecar declares
    }
    for (std::size_t i = 0; i < n; ++i) {
      A[i * n + i] += 4.0;  // diagonally dominant, so well conditioned
      b[i] = basic::next_uniform(rng);
    }

    f.call();

    const double* x = f.output<double>(0);
    const double residual_from_jax = *f.output<double>(1);
    // docs: end call

    const double inf_norm = basic::residual_inf_norm(A, b, x, n);
    basic::print_solution(x, n, inf_norm, residual_from_jax);

    // `!(<=)` rather than `>`, so a nan fails instead of passing.
    if (!(inf_norm <= basic::kResidualTolerance)) {
      return basic::residual_failure(inf_norm);
    }

    if (options.debug) {
      basic::demonstrate_debug_checks(f, a_index);
    } else {
      std::printf("debug=0 (checks disabled; see --debug)\n");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "example_01_basic: %s\n", error.what());
    return 1;
  }
}
