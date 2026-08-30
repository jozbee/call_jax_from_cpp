/**
 * @file aot_jax2exec_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call jax from C++.
 */

#include <cmath>
#include <iostream>
#include <thread>

#include "src/pjrt_exec/pjrt_exec.hpp"

int main() {
  // example setup
  const std::size_t num_samples = 1024;
  const std::string base_name = "./artifacts/jax_jax2exec";
  std::vector<double> output_data = {-1.0, -1.0, -1.0, -1.0};  // sentinel

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];
  pjrt::AOTComputation aot_comp(base_name, client);

  // WARNING: sleep to avoid segfault
  // without sleeping, the BUFFER::to_device call sometimes segfaults
  // I do not now what the optimal sleep time is, but 1ms is sufficient
  std::this_thread::sleep_for(std::chrono::microseconds(1000));

  // random input timing
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // random input: A (4x4, flattened row-major) and b (4,)
    // the diagonal is biased so that A stays well-conditioned, since
    // `jax_jax2exec.py` inverts A
    std::vector<double> A_data(16);
    std::vector<double> b_data(4);

    for (auto& v : A_data) {
      v = static_cast<double>(rand()) / RAND_MAX;
    }
    for (std::size_t d = 0; d < 4; ++d) {
      A_data[d * 4 + d] += 4.0;
    }
    for (auto& v : b_data) {
      v = static_cast<double>(rand()) / RAND_MAX;
    }

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // compute
    std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
        pjrt::Buffer::to_device_blocking(A_data.data(),
                                         A_data.size(), client, device),
        pjrt::Buffer::to_device_blocking(b_data.data(),
                                         b_data.size(), client, device)};
    auto output_buffers = aot_comp.execute_blocking(input_buffers);
    output_buffers[0]->to_host_blocking(output_data.data(), output_data.size());

    // end timing
    auto end = std::chrono::high_resolution_clock::now();
    timings[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }

  // compute average timing
  double avg_timing = 0.0;
  for (std::size_t i = 0; i < num_samples; ++i) {
    avg_timing += timings[i];
  }
  avg_timing /= num_samples;
  std::cout << "Average timing: " << avg_timing << " microseconds" << std::endl;

  // compute stddev timing
  double stddev_timing = 0.0;
  for (std::size_t i = 0; i < num_samples; ++i) {
    stddev_timing += (timings[i] - avg_timing) * (timings[i] - avg_timing);
  }
  stddev_timing = std::sqrt(stddev_timing / num_samples);
  std::cout << "Stddev timing: " << stddev_timing << " microseconds"
            << std::endl;

  // compute min and max timing
  std::size_t min_index = 1;
  std::size_t max_index = 1;
  for (std::size_t i = 2; i < num_samples; ++i) {
    if (timings[i] < timings[min_index]) {
      min_index = i;
    }
    if (timings[i] > timings[max_index]) {
      max_index = i;
    }
  }
  std::cout << "Min timing: " << timings[min_index] << " microseconds\n";
  std::cout << "Max timing: " << timings[max_index] << " microseconds\n";
  std::cout << "Min timing index: " << min_index << std::endl;
  std::cout << "Max timing index: " << max_index << std::endl;

  // not setinels?
  std::cout << "Output data: " << output_data[0] << ", " << output_data[1]
            << ", " << output_data[2] << ", " << output_data[3] << std::endl;

  return 0;
}
