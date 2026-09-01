/**
 * @file aot_jax2exec_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call jax from C++.
 *
 * The function was exported by `src/examples/jax_jax2exec.py`: it takes a 4x4
 * matrix A (flattened) and a vector b, and returns `inv(A) @ b` together with
 * a few derived quantities.
 *
 * The shape of this example is the shape a control loop should have: load
 * once, then call in a loop, writing inputs into the arenas the runtime owns
 * rather than handing it new buffers every time.
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/pjrt_exec/runtime.hpp"

int main() {
  const std::size_t num_samples = 1024;

  // --- load once ---
  pjrt::Runtime runtime;
  pjrt::Function solve(runtime, "./artifacts/jax_jax2exec");

  std::cout << "loaded: " << solve.num_inputs() << " inputs, "
            << solve.num_outputs() << " outputs\n";

  // --- call many ---
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // Random input: A (4x4, flattened row-major) and b (4,). The diagonal is
    // biased so that A stays well-conditioned, since the exported function
    // inverts it.
    double* A = solve.input(0);
    double* b = solve.input(1);
    for (std::size_t j = 0; j < solve.input_size(0); ++j) {
      A[j] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (std::size_t d = 0; d < 4; ++d) {
      A[d * 4 + d] += 4.0;
    }
    for (std::size_t j = 0; j < solve.input_size(1); ++j) {
      b[j] = static_cast<double>(rand()) / RAND_MAX;
    }

    const auto start = std::chrono::steady_clock::now();
    solve.call();
    const auto end = std::chrono::steady_clock::now();

    timings[i] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() *
        1e-3;
  }

  // --- report ---
  double mean = 0.0;
  for (double t : timings) {
    mean += t;
  }
  mean /= static_cast<double>(num_samples);

  double stddev = 0.0;
  for (double t : timings) {
    stddev += (t - mean) * (t - mean);
  }
  stddev = std::sqrt(stddev / static_cast<double>(num_samples));

  double min = timings[0];
  double max = timings[0];
  for (double t : timings) {
    min = std::min(min, t);
    max = std::max(max, t);
  }

  std::cout << "Average timing: " << mean << " microseconds\n"
            << "Stddev timing:  " << stddev << " microseconds\n"
            << "Min timing:     " << min << " microseconds\n"
            << "Max timing:     " << max << " microseconds\n";

  const double* x = solve.output(0);
  std::cout << "Output data: " << x[0] << ", " << x[1] << ", " << x[2] << ", "
            << x[3] << std::endl;
  return 0;
}
