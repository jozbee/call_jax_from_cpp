/**
 * @file aot_jax2exec_example.cpp
 * @brief Example of using `pjrt_exec` and `jax2exec` to call jax from C++.
 */

#include <iostream>

#include "src/pjrt_exec/pjrt_exec.hpp"

int main() {
  // example setup
  const std::vector<std::vector<double>> input_data = {
      {-1.0, 0.0, 1.0, 2.0},
      {-2.0, -1.0, 0.0, 1.0},
  };
  std::vector<double> output_data = {-1.0, -1.0};  // sentinel
  const std::string base_name = "./artifacts/jax_jax2exec";

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];

  // inputs
  std::vector<std::shared_ptr<pjrt::Buffer>> input_buffers = {
      pjrt::Buffer::to_device_blocking(input_data[0].data(),
                                       input_data[0].size(), client, device),
      pjrt::Buffer::to_device_blocking(input_data[1].data(),
                                       input_data[1].size(), client, device)};

  // exec
  pjrt::AOTComputation aot_comp(base_name, client);
  auto output_buffers = aot_comp.execute_blocking(input_buffers);

  // ouputs
  output_buffers[0]->to_host_blocking(&output_data[0], 0);
  output_buffers[1]->to_host_blocking(&output_data[1], 0);
  std::cout << "output data: " << output_data[0] << ", " << output_data[1]
            << "\n";

  return 0;
}
