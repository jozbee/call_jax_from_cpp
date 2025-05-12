// call a simple xla script, preferably a couple of times
// we don't want to require all of the xla dependencies, only the runtime
//  shared library

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "src/xla/pjrt_c_api.h"
#include "src/xla/pjrt_c_api_cpu.h"

PJRT_Error_Code get_error_code(const PJRT_Api* api, PJRT_Error* error) {
  PJRT_Error_GetCode_Args args;
  memset(&args, 0, sizeof(PJRT_Error_GetCode_Args));
  args.struct_size = sizeof(PJRT_Error_GetCode_Args);
  args.extension_start = nullptr;
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  return args.code;
}

const char* get_error_message(const PJRT_Api* api, PJRT_Error* error) {
  PJRT_Error_Message_Args args;
  memset(&args, 0, sizeof(PJRT_Error_Message_Args));
  args.struct_size = sizeof(PJRT_Error_Message_Args);
  args.extension_start = nullptr;
  args.error = error;
  api->PJRT_Error_Message(&args);
  return args.message;
}

void check_error(PJRT_Error* error, const PJRT_Api* api) {
  if (!error) return;
  const PJRT_Error_Code code = get_error_code(api, error);
  std::cout << "RECEIVED ERROR CODE " << code << "\n";
  const char* message = get_error_message(api, error);
  std::cout << "RECEIVED ERROR MESSAGE " << message << "\n";

  PJRT_Error_Destroy_Args args;
  memset(&args, 0, sizeof(PJRT_Error_Destroy_Args));
  args.struct_size = sizeof(PJRT_Error_Destroy_Args);
  args.extension_start = nullptr;
  args.error = error;
  api->PJRT_Error_Destroy(&args);
}

int main() {
  // read compiled program
  const std::string input_binary = "artifacts/jax_example.binpb";
  std::ifstream input_file(input_binary, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input_file), {});
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

  // cpu device
  PJRT_Client_AddressableDevices_Args device_args;
  memset(&device_args, 0, sizeof(PJRT_Client_AddressableDevices_Args));
  device_args.struct_size = sizeof(PJRT_Client_AddressableDevices_Args);
  device_args.client = client;
  check_error(api->PJRT_Client_AddressableDevices(&device_args), api);
  PJRT_Device* cpu_device = device_args.addressable_devices[0];
  std::cout << "cpu device, out of " << device_args.num_addressable_devices
            << "\n";
  std::flush(std::cout);

  // read in executable
  std::ifstream exec_file("./artifacts/jax_example_exec.binpb",
                          std::ios_base::binary);
  const std::vector<char> exec_buffer(std::istreambuf_iterator<char>(exec_file),
                                      {});
  std::cout << "read exec\n";
  std::flush(std::cout);

  // load executable
  PJRT_Executable_DeserializeAndLoad_Args load_args;
  memset(&load_args, 0, sizeof(PJRT_Executable_DeserializeAndLoad_Args));
  load_args.struct_size = sizeof(PJRT_Executable_DeserializeAndLoad_Args);
  load_args.extension_start = nullptr;
  load_args.client = client;
  load_args.serialized_executable = exec_buffer.data();
  load_args.serialized_executable_size = exec_buffer.size();
  check_error(api->PJRT_Executable_DeserializeAndLoad(&load_args), api);
  PJRT_LoadedExecutable* loaded_executable = load_args.loaded_executable;
  std::cout << "load exec\n";
  std::flush(std::cout);

  // input
  double input_data = 3.0;
  std::array<int64_t, 0> dims = {};  // scalar
  const size_t num_dims = 0;

  PJRT_Client_BufferFromHostBuffer_Args input_buffer_args;
  memset(&input_buffer_args, 0, sizeof(PJRT_Client_BufferFromHostBuffer_Args));
  input_buffer_args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  input_buffer_args.client = client;
  input_buffer_args.data = &input_data;
  input_buffer_args.type = PJRT_Buffer_Type::PJRT_Buffer_Type_F64;
  input_buffer_args.dims = dims.data();
  input_buffer_args.num_dims = num_dims;
  input_buffer_args.byte_strides = nullptr;  // dense layout
  input_buffer_args.num_byte_strides = 0;    // dense layout
  input_buffer_args.host_buffer_semantics = PJRT_HostBufferSemantics::
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  input_buffer_args.device = cpu_device;
  input_buffer_args.memory = nullptr;  // consider non-null, for less copying
  input_buffer_args.device_layout = nullptr;  // dense layout
  check_error(api->PJRT_Client_BufferFromHostBuffer(&input_buffer_args), api);
  PJRT_Event* host_event = input_buffer_args.done_with_host_buffer;
  PJRT_Event_Await_Args host_event_args;
  memset(&host_event_args, 0, sizeof(PJRT_Event_Await_Args));
  host_event_args.struct_size = sizeof(PJRT_Event_Await_Args);
  host_event_args.extension_start = nullptr;
  host_event_args.event = host_event;
  check_error(api->PJRT_Event_Await(&host_event_args), api);
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

  // output
  // remark: do not allocate ouput buffers, because that is the job of execution
  //  (if there a no donatable buffers)
  std::vector<PJRT_Buffer*> output_buffer(1);  // sized to hold one null pointer
  PJRT_Buffer** output_buffer_ptr = output_buffer.data();

  // execute
  const std::array<int64_t, 1> non_donatable_input_indices = {0};
  PJRT_ExecuteOptions execute_options;
  memset(&execute_options, 0, sizeof(PJRT_ExecuteOptions));
  execute_options.struct_size = sizeof(PJRT_ExecuteOptions);
  execute_options.extension_start = nullptr;
  execute_options.send_callbacks = nullptr;
  execute_options.recv_callbacks = nullptr;
  execute_options.num_send_ops = 0;
  execute_options.num_recv_ops = 0;
  execute_options.launch_id = 0;  // only one device
  execute_options.non_donatable_input_indices =
      non_donatable_input_indices.data();
  execute_options.num_non_donatable_input_indices = 1;
  execute_options.context = nullptr;  // nothing fancy here

  PJRT_Event* exec_event = nullptr;
  PJRT_LoadedExecutable_Execute_Args execute_args;
  memset(&execute_args, 0, sizeof(PJRT_LoadedExecutable_Execute_Args));
  execute_args.struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args);
  execute_args.extension_start = nullptr;
  execute_args.executable = loaded_executable;
  execute_args.options = &execute_options;
  execute_args.argument_lists = &input_buffer;
  execute_args.num_devices = 1;
  execute_args.num_args = 1;
  execute_args.output_lists = &output_buffer_ptr;
  execute_args.device_complete_events = &exec_event;
  execute_args.execute_device = nullptr;  // execute on _only_ compiled device

  check_error(api->PJRT_LoadedExecutable_Execute(&execute_args), api);

  std::cout << "execute\n";
  std::flush(std::cout);

  // wait for execution to finish
  PJRT_Event_Await_Args exec_event_args;
  memset(&exec_event_args, 0, sizeof(PJRT_Event_Await_Args));
  exec_event_args.struct_size = sizeof(PJRT_Event_Await_Args);
  exec_event_args.extension_start = nullptr;
  exec_event_args.event = exec_event;
  check_error(api->PJRT_Event_Await(&exec_event_args), api);
  std::cout << "execution finished\n";
  std::flush(std::cout);

  // output
  PJRT_Event_Await_Args output_event_args;
  memset(&output_event_args, 0, sizeof(PJRT_Event_Await_Args));
  output_event_args.struct_size = sizeof(PJRT_Event_Await_Args);
  output_event_args.extension_start = nullptr;

  double output_data = 0.0;  // sentinel
  PJRT_Buffer_ToHostBuffer_Args output_buffer_args;
  memset(&output_buffer_args, 0, sizeof(PJRT_Buffer_ToHostBuffer_Args));
  output_buffer_args.struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args);
  output_buffer_args.extension_start = nullptr;
  output_buffer_args.src = output_buffer[0];
  output_buffer_args.host_layout = nullptr;  // dense layout
  output_buffer_args.dst = &output_data;
  output_buffer_args.dst_size = sizeof(double);
  output_buffer_args.event = nullptr;  // output
  check_error(api->PJRT_Buffer_ToHostBuffer(&output_buffer_args), api);
  output_event_args.event = output_buffer_args.event;
  check_error(api->PJRT_Event_Await(&output_event_args), api);
  std::cout << "output buffer size: " << output_buffer_args.dst_size << "\n";
  std::cout << "output data: " << output_data << "\n";
  std::flush(std::cout);

  return 0;
}
