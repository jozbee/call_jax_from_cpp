/**
 * @file aot_jax2exec_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call jax from C++.
 */

#include <iostream>

#include "src/pjrt_exec/pjrt_exec.hpp"

int main() {
  // example setup
  const std::size_t num_samples = 1024;
  const std::string base_name = "./artifacts/jax_jax2exec";
  std::vector<double> output_data = {-1.0, -1.0};  // sentinel

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];
  pjrt::AOTComputation aot_comp(base_name, client);

  // random input timing
  std::vector<double> timings(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    // random input
    std::vector<std::vector<double>> input_data = {
        {static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX},
        {static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX,
         static_cast<double>(rand()) / RAND_MAX}};

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // compute
    std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
        pjrt::Buffer::to_device_blocking(input_data[0].data(),
                                         input_data[0].size(), client, device),
        pjrt::Buffer::to_device_blocking(input_data[1].data(),
                                         input_data[1].size(), client, device)};
    auto output_buffers = aot_comp.execute_blocking(input_buffers);
    output_buffers[0]->to_host_blocking(&output_data[0], 0);
    output_buffers[1]->to_host_blocking(&output_data[1], 0);

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
  std::size_t min_index = 0;
  std::size_t max_index = 0;
  for (std::size_t i = 1; i < num_samples; ++i) {
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
            << std::endl;

  return 0;
}
