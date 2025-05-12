/**
 * @file aot_api_example.cpp
 * @brief Example of using our `pjrt_exec` api to execute a compiled program.
 */

#include <iostream>

#include "src/pjrt_exec/pjrt_exec.h"

int main() {
  // example setup
  const double input_data = 3.0;
  double output_data = 0.0;  // sentinel
  const std::string input_binary = "./artifacts/jax_example_exec.binpb";

  // pjrt setup
  auto client = std::make_shared<pjrt::Client>();
  auto devices = client->get_devices();
  auto device = devices[0];

  // inputs
  auto input_buffer =
      pjrt::Buffer::to_device_blocking(&input_data, 0, client, device);

  // exec
  pjrt::AOTComputation aot_comp(input_binary, client);
  auto output_buffer = aot_comp.execute_blocking(input_buffer);

  // ouputs
  output_buffer->to_host_blocking(&output_data, 0);
  std::cout << "output data: " << output_data << "\n";

  return 0;
}
