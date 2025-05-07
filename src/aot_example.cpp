// call a simple xla script, preferably a couple of times
// we don't want to require all of the xla dependencies, only the runtime
//  shared library

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "artifacts/pjrt_c_api.h"
#include "artifacts/pjrt_c_api_cpu.h"

PJRT_Error_Code get_error_code(const PJRT_Api* api, PJRT_Error* error) {
  PJRT_Error_GetCode_Args args;
  memset(&args, 0, sizeof(PJRT_Error_GetCode_Args));
  args.struct_size = sizeof(PJRT_Error_GetCode_Args);
  args.extension_start = nullptr;
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  return args.code;
}

void check_error(PJRT_Error* error, const PJRT_Api* api) {
  if (!error) return;
  const PJRT_Error_Code code = get_error_code(api, error);
  std::cout << "RECEIVED ERROR CODE " << code << std::endl;
}

int main() {
  // read compiled program
  const std::string input_binary = "artifacts/jax_example.binpb";
  std::ifstream input_file(input_binary, std::ios::binary);
  std::vector<char> buffer(
      // the extra parentheses are important
      (std::istreambuf_iterator<char>(input_file)),
      std::istreambuf_iterator<char>());

  std::cout << "read binary\n";
  std::flush(std::cout);

  // api
  // should be linked with the cpu plugin (.so)
  const PJRT_Api* api = GetPjrtApi();

  std::cout << "api\n";
  std::flush(std::cout);

  // initialize
  PJRT_Plugin_Initialize_Args init_args;
  memset(&init_args, 0, sizeof(PJRT_Plugin_Initialize_Args));
  init_args.struct_size = sizeof(PJRT_Plugin_Initialize_Args);
  init_args.extension_start = nullptr;
  check_error(api->PJRT_Plugin_Initialize(&init_args), api);
  std::cout << "initialize\n";
  std::flush(std::cout);

  // client
  PJRT_Client_Create_Args client_create_args;
  memset(&client_create_args, 0, sizeof(PJRT_Client_Create_Args));
  client_create_args.struct_size = sizeof(PJRT_Client_Create_Args);
  check_error(api->PJRT_Client_Create(&client_create_args), api);
  PJRT_Client* client = client_create_args.client;

  std::cout << "client\n";
  std::flush(std::cout);

  // load executable
  PJRT_Executable_DeserializeAndLoad_Args decereal_args;
  memset(&decereal_args, 0, sizeof(PJRT_Executable_DeserializeAndLoad_Args));
  decereal_args.struct_size = sizeof(PJRT_Executable_DeserializeAndLoad_Args);
  decereal_args.client = client;
  decereal_args.serialized_executable = buffer.data();
  decereal_args.serialized_executable_size = buffer.size();
  check_error(api->PJRT_Executable_DeserializeAndLoad(&decereal_args), api);
  PJRT_LoadedExecutable* loaded_executable = decereal_args.loaded_executable;

  std::cout << "load executable\n";
  std::flush(std::cout);

  // cpu device
  PJRT_Client_AddressableDevices_Args device_args;
  memset(&device_args, 0, sizeof(PJRT_Client_AddressableDevices_Args));
  device_args.struct_size = sizeof(PJRT_Client_AddressableDevices_Args);
  device_args.client = client;
  check_error(api->PJRT_Client_AddressableDevices(&device_args), api);
  PJRT_Device* cpu_device = device_args.addressable_devices[0];

  std::cout << "cpu device\n";
  std::flush(std::cout);

  // input
  double input_data = 3.0;
  int64_t dims[] = {};  // scalar?
  size_t num_dims = 0;

  PJRT_Client_BufferFromHostBuffer_Args input_buffer_args;
  memset(&input_buffer_args, 0, sizeof(PJRT_Client_BufferFromHostBuffer_Args));
  input_buffer_args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  input_buffer_args.client = client;
  input_buffer_args.data = &input_data;
  input_buffer_args.type = PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
  input_buffer_args.dims = dims;
  input_buffer_args.num_dims = num_dims;
  input_buffer_args.byte_strides = nullptr;             // dense layout
  input_buffer_args.num_byte_strides = sizeof(double);  // dense layout
  input_buffer_args.host_buffer_semantics = PJRT_HostBufferSemantics::
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  input_buffer_args.device = cpu_device;
  input_buffer_args.memory = nullptr;  // consider non-null, for less copying
  input_buffer_args.device_layout = nullptr;  // dense layout

  check_error(api->PJRT_Client_BufferFromHostBuffer(&input_buffer_args), api);

  PJRT_Buffer** input_buffer = &input_buffer_args.buffer;

  std::cout << "buffer in\n";
  std::flush(std::cout);

  // check if input is on cpu
  PJRT_Buffer_IsOnCpu_Args is_on_cpu_args;
  memset(&is_on_cpu_args, 0, sizeof(PJRT_Buffer_IsOnCpu_Args));
  is_on_cpu_args.struct_size = sizeof(PJRT_Buffer_IsOnCpu_Args);
  is_on_cpu_args.extension_start = nullptr;
  is_on_cpu_args.buffer = input_buffer_args.buffer;
  check_error(api->PJRT_Buffer_IsOnCpu(&is_on_cpu_args), api);

  if (is_on_cpu_args.is_on_cpu) {
    std::cout << "input buffer is on cpu\n";
  } else {
    std::cout << "input buffer is NOT on cpu\n";
  }
  std::flush(std::cout);

  // buffer out
  double output_data = 0.0;
  PJRT_Client_BufferFromHostBuffer_Args output_buffer_args = input_buffer_args;
  output_buffer_args.data = &output_data;
  output_buffer_args.buffer = nullptr;  // reset buffer (because copy)
  check_error(api->PJRT_Client_BufferFromHostBuffer(&output_buffer_args), api);
  PJRT_Buffer** output_buffer = &output_buffer_args.buffer;

  std::cout << "buffer out\n";
  std::flush(std::cout);

  // check if output is on cpu
  is_on_cpu_args.buffer = output_buffer_args.buffer;
  check_error(api->PJRT_Buffer_IsOnCpu(&is_on_cpu_args), api);
  if (is_on_cpu_args.is_on_cpu) {
    std::cout << "output buffer is on cpu\n";
  } else {
    std::cout << "output buffer is NOT on cpu\n";
  }
  std::flush(std::cout);

  // get executable
  PJRT_LoadedExecutable_GetExecutable_Args get_executable_args;
  memset(&get_executable_args, 0,
         sizeof(PJRT_LoadedExecutable_GetExecutable_Args));
  get_executable_args.struct_size =
      sizeof(PJRT_LoadedExecutable_GetExecutable_Args);
  get_executable_args.extension_start = nullptr;
  get_executable_args.loaded_executable = loaded_executable;
  check_error(api->PJRT_LoadedExecutable_GetExecutable(&get_executable_args),
              api);
  PJRT_Executable* executable = get_executable_args.executable;
  std::cout << "get executable\n";
  std::flush(std::cout);

  // execute
  PJRT_ExecuteContext_Create_Args execute_context_args;
  memset(&execute_context_args, 0, sizeof(PJRT_ExecuteContext_Create_Args));
  execute_context_args.struct_size = sizeof(PJRT_ExecuteContext_Create_Args);
  execute_context_args.extension_start = nullptr;
  check_error(api->PJRT_ExecuteContext_Create(&execute_context_args), api);
  PJRT_ExecuteContext* execute_context = execute_context_args.context;

  const int64_t non_donatable_input_indices[] = {0};
  PJRT_ExecuteOptions execute_options;
  memset(&execute_options, 0, sizeof(PJRT_ExecuteOptions));
  execute_options.struct_size = sizeof(PJRT_ExecuteOptions);
  execute_options.extension_start = nullptr;
  execute_options.send_callbacks = nullptr;
  execute_options.recv_callbacks = nullptr;
  execute_options.num_send_ops = 0;
  execute_options.num_recv_ops = 0;
  execute_options.launch_id = 0;  // only one device
  execute_options.non_donatable_input_indices = non_donatable_input_indices;
  execute_options.num_non_donatable_input_indices = 1;
  execute_options.context = execute_context;

  PJRT_Event* event_ptr;  // lc0
  PJRT_LoadedExecutable_Execute_Args execute_args;
  memset(&execute_args, 0, sizeof(PJRT_LoadedExecutable_Execute_Args));
  execute_args.struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args);
  execute_args.extension_start = nullptr;
  execute_args.executable = loaded_executable;
  execute_args.options = &execute_options;
  execute_args.argument_lists = &input_buffer;
  execute_args.num_devices = 1;
  execute_args.num_args = 1;
  execute_args.output_lists = &output_buffer;
  execute_args.device_complete_events = &event_ptr;
  execute_args.execute_device = nullptr;  // execute compiled device

  check_error(api->PJRT_LoadedExecutable_Execute(&execute_args), api);

  std::cout << "execute\n";
  std::flush(std::cout);

  return 0;
}
