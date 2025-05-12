#include <cassert>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/lc0/pjrt.h"

int main() {
  // pjrt
  lczero::Pjrt pjrt("./artifacts/libpjrt_c_api_cpu_plugin_darwin.dylib");
  std::cout << "pjrt\n";
  std::flush(std::cout);

  // client
  auto client = pjrt.CreateClient();
  std::cout << "client\n";
  std::flush(std::cout);

  // read in hlo file
  std::ifstream hlo_file("./artifacts/jax_example.binpb",
                         std::ios_base::binary);
  const std::vector<char> hlo_buffer(std::istreambuf_iterator<char>(hlo_file),
                                     {});
  std::ifstream comp_opt_file("./artifacts/jax_example_comp_opt.binpb",
                              std::ios_base::binary);
  const std::vector<char> comp_opt_buffer(
      std::istreambuf_iterator<char>(comp_opt_file), {});
  std::cout << "read hlo\n";
  std::flush(std::cout);

  // compile
  auto exec =
      client->CompileHlo({hlo_buffer.data(), hlo_buffer.size()},
                         {comp_opt_buffer.data(), comp_opt_buffer.size()});
  std::cout << "exec\n";
  std::flush(std::cout);

  // cpu device
  auto devices = client->GetDevices();
  assert(devices.size() > 1);  // == 4?
  auto* device = devices[0].get();
  std::cout << "device: " << device->ToString() << "\n";
  std::flush(std::cout);

  // input
  double input = 2.0;
  const std::string_view input_view(reinterpret_cast<char*>(&input));
  auto input_device_transfer = client->HostToDevice(
      reinterpret_cast<char*>(&input), lczero::PjrtType::F64, {}, device);
  auto input_device_buffer = input_device_transfer->AwaitAndReleaseBuffer();
  std::cout << "input_device_buffer: " << input_device_buffer->GetSize()
            << "\n";
  std::flush(std::cout);

  // execute
  auto output_buffers = exec->ExecuteBlocking({input_device_buffer.get()});
  std::cout << "output_buffers.size(): " << output_buffers.size() << "\n";
  std::flush(std::cout);

  // output
  double output = 0.0;  // sentinel
  auto get_output_event =
      output_buffers[0]->DeviceToHost(&output, sizeof(double));
  get_output_event->Await();
  std::cout << "Output: " << output << "\n";
  std::flush(std::cout);

  return 0;
}
