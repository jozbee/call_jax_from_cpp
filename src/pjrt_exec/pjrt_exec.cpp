#include "src/pjrt_exec/pjrt_exec.h"

#include <fstream>
#include <iostream>

// enforce internal linkags
// (clang-tidy told me to do it)
namespace {
size_t _get_error_code(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return 0;
  }
  PJRT_Error_GetCode_Args args = {
      .struct_size = sizeof(PJRT_Error_GetCode_Args),
      .extension_start = nullptr,
      .error = error};
  api->PJRT_Error_GetCode(&args);
  return args.code;
}

std::string _get_error_message(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return "";
  }
  PJRT_Error_Message_Args args = {
      .struct_size = sizeof(PJRT_Error_Message_Args),
      .extension_start = nullptr,
      .error = error};
  api->PJRT_Error_Message(&args);
  return {args.message, args.message_size};
}

}  // namespace

namespace pjrt {

///////////
// setup //
///////////

const PJRT_Api* get_pjrt_api_() {
  const PJRT_Api* api = GetPjrtApi();
  PJRT_Plugin_Initialize_Args args = {
      .struct_size = sizeof(PJRT_Plugin_Initialize_Args),
      .extension_start = nullptr};
  PJRT_Error* error = api->PJRT_Plugin_Initialize(&args);
  if (error != nullptr) {
    std::cerr << "Error initializing PJRT plugin: "
              << _get_error_message(api, error) << "\n";
    PJRT_Error_Destroy_Args destroy_args = {
        .struct_size = sizeof(PJRT_Error_Destroy_Args),
        .extension_start = nullptr,
        .error = error};
    api->PJRT_Error_Destroy(&destroy_args);
  }
  return api;
}

///////////
// error //
///////////

Error::Error(PJRT_Error* error)
    : std::runtime_error(_get_error_message(api_, error)),
      code_(_get_error_code(api_, error)),
      message_(_get_error_message(api_, error)) {
  PJRT_Error_Destroy_Args args = {
      .struct_size = sizeof(PJRT_Error_Destroy_Args),
      .extension_start = nullptr,
      .error = error};
  api_->PJRT_Error_Destroy(&args);
}

void check_error(PJRT_Error* error) {
  if (error != nullptr) {
    throw Error(error);
  }
}

////////////
// client //
////////////

Client::Client() {
  PJRT_Client_Create_Args args = {
      .struct_size = sizeof(PJRT_Client_Create_Args),
      .extension_start = nullptr,
      .create_options = nullptr,
      .num_options = 0,
      .kv_get_callback = nullptr,
      .kv_get_user_arg = nullptr,
      .kv_put_callback = nullptr,
      .kv_put_user_arg = nullptr,
      .client = nullptr,  // out
      .kv_try_get_callback = nullptr,
      .kv_try_get_user_arg = nullptr};
  check_error(api_->PJRT_Client_Create(&args));
  this->client_ = args.client;
}

Client::~Client() {
  PJRT_Client_Destroy_Args args = {
      .struct_size = sizeof(PJRT_Client_Destroy_Args),
      .extension_start = nullptr,
      .client = client_};
  try {
    check_error(api_->PJRT_Client_Destroy(&args));
  } catch (const Error& e) {
    // I don't know how to handle this error?
    std::cerr << "Client destructor error: " << e.what() << "\n";
  }
}

std::vector<std::shared_ptr<Device>> Client::get_devices() const {
  PJRT_Client_Devices_Args args = {
      .struct_size = sizeof(PJRT_Client_Devices_Args),
      .extension_start = nullptr,
      .client = client_,
      .devices = nullptr,  // out
      .num_devices = 0};   // out
  check_error(api_->PJRT_Client_Devices(&args));
  std::vector<std::shared_ptr<Device>> devices(args.num_devices);
  for (size_t i = 0; i < args.num_devices; ++i) {
    devices[i] = std::make_shared<Device>(args.devices[i]);
  }
  return devices;
}

////////////
// device //
////////////

Device::Device(PJRT_Device* device) : device_(device) {
  if (device == nullptr) {
    throw std::invalid_argument("Device cannot be null");
  }
}

///////////
// event //
///////////

Event::Event(PJRT_Event* event) : event_(event) {
  if (event == nullptr) {
    throw std::invalid_argument("Event cannot be null");
  }
}

Event::~Event() {
  PJRT_Event_Destroy_Args args = {
      .struct_size = sizeof(PJRT_Event_Destroy_Args),
      .extension_start = nullptr,
      .event = event_};
  api_->PJRT_Event_Destroy(&args);
}

void Event::await() {
  PJRT_Event_Await_Args args = {.struct_size = sizeof(PJRT_Event_Await_Args),
                                .extension_start = nullptr,
                                .event = event_};
  check_error(api_->PJRT_Event_Await(&args));
}

////////////
// buffer //
////////////

Buffer::Buffer(PJRT_Buffer* buffer) : buffer_(buffer) {
  if (buffer == nullptr) {
    throw std::invalid_argument("Buffer cannot be null");
  }
}

Buffer::~Buffer() {
  PJRT_Buffer_Destroy_Args args = {
      .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
      .extension_start = nullptr,
      .buffer = buffer_};
  api_->PJRT_Buffer_Destroy(&args);
}

std::tuple<std::shared_ptr<Buffer>, std::shared_ptr<Event>> Buffer::to_device(
    const double* data, size_t size, std::shared_ptr<Client> client,
    std::shared_ptr<Device> device) {
  const std::array<int64_t, 1> dims = {static_cast<int64_t>(size)};
  const size_t num_dims = size == 0 ? 0 : 1;
  PJRT_Client_BufferFromHostBuffer_Args args = {
      .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
      .client = client->client_,
      .data = data,
      .type = PJRT_Buffer_Type::PJRT_Buffer_Type_F64,  // good for science
      .dims = dims.data(),
      .num_dims = num_dims,
      .byte_strides = nullptr,  // dense layout
      .num_byte_strides = 0,    // dense layout
      // warning!
      // assumes data is not modified while buffer is alive
      // (usually fine...)
      .host_buffer_semantics =
          PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableZeroCopy,
      .device = device->device_,
      .memory = nullptr,                 // consider non-null, for less copying
      .device_layout = nullptr,          // dense layout
      .done_with_host_buffer = nullptr,  // out
      .buffer = nullptr};                // out
  check_error(api_->PJRT_Client_BufferFromHostBuffer(&args));
  return {std::make_shared<Buffer>(args.buffer),
          std::make_shared<Event>(args.done_with_host_buffer)};
}

std::shared_ptr<Buffer> Buffer::to_device_blocking(
    const double* data, size_t size, std::shared_ptr<Client> client,
    std::shared_ptr<Device> device) {
  auto [buffer, event] = Buffer::to_device(data, size, client, device);
  event->await();
  return buffer;
}

std::shared_ptr<Event> Buffer::to_host(double* data, size_t size) {
  size = size == 0 ? 1 : size;
  PJRT_Buffer_ToHostBuffer_Args args = {
      .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
      .extension_start = nullptr,
      .src = buffer_,
      .host_layout = nullptr,  // dense layout
      .dst = data,
      .dst_size = size * sizeof(double),
      .event = nullptr};  // out
  check_error(api_->PJRT_Buffer_ToHostBuffer(&args));
  return std::make_shared<Event>(args.event);
}

void Buffer::to_host_blocking(double* data, size_t size) {
  auto event = this->to_host(data, size);
  event->await();
}

/////////
// aot //
/////////

AOTComputation::AOTComputation(const std::string& file_name,
                               std::shared_ptr<Client> client) {
  this->client_ = client;

  // read file
  std::ifstream file(file_name, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + file_name);
  }
  std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  if (buffer.empty()) {
    throw std::runtime_error("File is empty: " + file_name);
  }

  // load executable
  PJRT_Executable_DeserializeAndLoad_Args args = {
      .struct_size = sizeof(PJRT_Executable_DeserializeAndLoad_Args),
      .extension_start = nullptr,
      .client = client_->client_,
      .serialized_executable = buffer.data(),
      .serialized_executable_size = buffer.size(),
      .loaded_executable = nullptr};  // out
  check_error(api_->PJRT_Executable_DeserializeAndLoad(&args));
  this->loaded_executable_ = args.loaded_executable;
}

AOTComputation::~AOTComputation() {
  PJRT_LoadedExecutable_Destroy_Args args = {
      .struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args),
      .extension_start = nullptr,
      .executable = loaded_executable_};
  try {
    check_error(api_->PJRT_LoadedExecutable_Destroy(&args));
  } catch (const Error& e) {
    std::cerr << "AOTComputation destructor error: " << e.what() << "\n";
  }
}

std::tuple<std::shared_ptr<Buffer>, std::shared_ptr<Event>>
AOTComputation::execute(std::shared_ptr<Buffer> input) {
  // massage  input
  std::vector<PJRT_Buffer**> raw_input(1);
  raw_input[0] = &input->buffer_;

  // execute options
  const std::array<int64_t, 1> non_donatable_input_indices = {0};
  PJRT_ExecuteOptions execute_options = {
      .struct_size = sizeof(PJRT_ExecuteOptions),
      .extension_start = nullptr,
      .send_callbacks = nullptr,
      .recv_callbacks = nullptr,
      .num_send_ops = 0,
      .num_recv_ops = 0,
      .launch_id = 0,  // only one device
      .non_donatable_input_indices = non_donatable_input_indices.data(),
      .num_non_donatable_input_indices = non_donatable_input_indices.size(),
      .context = nullptr};  // nothing fancy here

  // execute
  std::vector<PJRT_Buffer*> output_buffer(1);  // sized to hold one null pointer
  PJRT_Buffer** output_buffer_ptr = output_buffer.data();
  PJRT_Event* exec_event = nullptr;
  PJRT_LoadedExecutable_Execute_Args args = {
      .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
      .extension_start = nullptr,
      .executable = loaded_executable_,
      .options = &execute_options,
      .argument_lists = raw_input.data(),
      .num_devices = 1,
      .num_args = 1,
      .output_lists = &output_buffer_ptr,     // out
      .device_complete_events = &exec_event,  // out
      .execute_device = nullptr};  // execute _only_ on compiled device
  check_error(api_->PJRT_LoadedExecutable_Execute(&args));
  return {std::make_shared<Buffer>(output_buffer[0]),
          std::make_shared<Event>(exec_event)};
}

std::shared_ptr<Buffer> AOTComputation::execute_blocking(
    std::shared_ptr<Buffer> input) {
  std::flush(std::cout);
  auto [buffer, event] = this->execute(input);
  event->await();
  return buffer;
}

}  // namespace pjrt
