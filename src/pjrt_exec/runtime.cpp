#include "src/pjrt_exec/runtime.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "src/nlohmann/json.hpp"
#include "src/pjrt_exec/pjrt_exec.hpp"

namespace pjrt {
namespace {

/// XLA refuses zero-copy below `xla::cpu::MinAlign()` and prefers 64 bytes.
constexpr std::size_t kAlignment = 64;

double* alloc_arena(std::size_t doubles) {
  // Scalars are recorded as size 0 but still occupy one element.
  const std::size_t n = doubles == 0 ? 1 : doubles;
  void* p = nullptr;
  if (posix_memalign(&p, kAlignment, n * sizeof(double)) != 0 ||
      p == nullptr) {
    throw std::runtime_error("failed to allocate an aligned arena");
  }
  std::memset(p, 0, n * sizeof(double));
  return static_cast<double*>(p);
}

std::size_t executable_num_outputs(PJRT_LoadedExecutable* loaded) {
  PJRT_LoadedExecutable_GetExecutable_Args get_args = {
      .struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
      .extension_start = nullptr,
      .loaded_executable = loaded,
      .executable = nullptr};  // out
  check_error(api()->PJRT_LoadedExecutable_GetExecutable(&get_args));

  PJRT_Executable_NumOutputs_Args num_args = {
      .struct_size = sizeof(PJRT_Executable_NumOutputs_Args),
      .extension_start = nullptr,
      .executable = get_args.executable,
      .num_outputs = 0};  // out
  PJRT_Error* error = api()->PJRT_Executable_NumOutputs(&num_args);

  PJRT_Executable_Destroy_Args destroy_args = {
      .struct_size = sizeof(PJRT_Executable_Destroy_Args),
      .extension_start = nullptr,
      .executable = get_args.executable};
  api()->PJRT_Executable_Destroy(&destroy_args);

  check_error(error);
  return num_args.num_outputs;
}

}  // namespace

/////////////
// runtime //
/////////////

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  // XLA sizes both its intra-op and dispatch pools from this, and reads it
  // when the client is created, so it has to be set first.
  if (options_.worker_threads > 0) {
    setenv("PJRT_NPROC", std::to_string(options_.worker_threads).c_str(),
           /*overwrite=*/1);
  }

  std::vector<PJRT_NamedValue> create_options;
  const std::string async_name = "asynchronous";
  const std::string device_count_name = "cpu_device_count";

  PJRT_NamedValue async_option = {
      .struct_size = sizeof(PJRT_NamedValue),
      .extension_start = nullptr,
      .name = async_name.c_str(),
      .name_size = async_name.size(),
      .type = PJRT_NamedValue_kBool,
      .bool_value = !options_.synchronous,
      .value_size = 1};
  PJRT_NamedValue device_count_option = {
      .struct_size = sizeof(PJRT_NamedValue),
      .extension_start = nullptr,
      .name = device_count_name.c_str(),
      .name_size = device_count_name.size(),
      .type = PJRT_NamedValue_kInt64,
      .int64_value = options_.cpu_device_count,
      .value_size = 1};

  if (options_.synchronous) {
    create_options.push_back(async_option);
  }
  if (options_.cpu_device_count > 0) {
    create_options.push_back(device_count_option);
  }

  PJRT_Client_Create_Args args = {
      .struct_size = sizeof(PJRT_Client_Create_Args),
      .extension_start = nullptr,
      .create_options = create_options.empty() ? nullptr
                                               : create_options.data(),
      .num_options = create_options.size(),
      .kv_get_callback = nullptr,
      .kv_get_user_arg = nullptr,
      .kv_put_callback = nullptr,
      .kv_put_user_arg = nullptr,
      .client = nullptr,  // out
      .kv_try_get_callback = nullptr,
      .kv_try_get_user_arg = nullptr};
  check_error(api()->PJRT_Client_Create(&args));
  client_ = args.client;

  // The stock CPU plugin parses only `cpu_device_count` and ignores anything
  // else without complaint, so acceptance of `asynchronous` cannot be detected
  // from the create call. `PJRT_Plugin_Attributes` carries the marker that a
  // patched plugin advertises.
  synchronous_supported_ = false;
  PJRT_Plugin_Attributes_Args attr_args = {
      .struct_size = sizeof(PJRT_Plugin_Attributes_Args),
      .extension_start = nullptr,
      .attributes = nullptr,   // out
      .num_attributes = 0};    // out
  if (api()->PJRT_Plugin_Attributes(&attr_args) == nullptr) {
    for (std::size_t i = 0; i < attr_args.num_attributes; ++i) {
      const PJRT_NamedValue& attr = attr_args.attributes[i];
      if (std::string(attr.name, attr.name_size) ==
          "supports_synchronous_execution") {
        synchronous_supported_ = true;
      }
    }
  }

  PJRT_Client_Devices_Args device_args = {
      .struct_size = sizeof(PJRT_Client_Devices_Args),
      .extension_start = nullptr,
      .client = client_,
      .devices = nullptr,  // out
      .num_devices = 0};   // out
  check_error(api()->PJRT_Client_Devices(&device_args));
  if (device_args.num_devices == 0) {
    throw std::runtime_error("PJRT client reported no devices");
  }
  device_ = device_args.devices[0];
}

Runtime::~Runtime() {
  if (client_ == nullptr) {
    return;
  }
  PJRT_Client_Destroy_Args args = {
      .struct_size = sizeof(PJRT_Client_Destroy_Args),
      .extension_start = nullptr,
      .client = client_};
  PJRT_Error* error = api()->PJRT_Client_Destroy(&args);
  if (error != nullptr) {
    PJRT_Error_Destroy_Args destroy = {
        .struct_size = sizeof(PJRT_Error_Destroy_Args),
        .extension_start = nullptr,
        .error = error};
    api()->PJRT_Error_Destroy(&destroy);
  }
}

//////////////
// function //
//////////////

Function::Function(Runtime& runtime, const std::string& base_name,
                   const FunctionOptions& options)
    : runtime_(runtime) {
  load_metadata(base_name);
  load_executable(base_name);

  if (options.check_metadata) {
    const std::size_t actual = executable_num_outputs(executable_);
    if (actual != output_sizes_.size()) {
      throw std::runtime_error(
          "Stale metadata: " + base_name + ".json declares " +
          std::to_string(output_sizes_.size()) + " outputs but " + base_name +
          ".binpb produces " + std::to_string(actual));
    }
  }

  allocate_arenas();
  wrap_inputs();

  output_buffers_.assign(output_sizes_.size(), nullptr);
  execute_options_ = PJRT_ExecuteOptions{
      .struct_size = sizeof(PJRT_ExecuteOptions),
      .extension_start = nullptr,
      .send_callbacks = nullptr,
      .recv_callbacks = nullptr,
      .num_send_ops = 0,
      .num_recv_ops = 0,
      .launch_id = 0,
      .non_donatable_input_indices = nullptr,
      .num_non_donatable_input_indices = 0,
      .context = nullptr};

  // Warm up on zeroed arenas: faults in every page, resolves lazy bindings,
  // and lets the allocator reach the steady state the caller will time.
  for (int i = 0; i < options.warmup_calls; ++i) {
    call();
  }
}

Function::~Function() {
  for (PJRT_Buffer* buffer : input_buffers_) {
    if (buffer == nullptr) {
      continue;
    }
    PJRT_Buffer_Destroy_Args args = {
        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
        .extension_start = nullptr,
        .buffer = buffer};
    api()->PJRT_Buffer_Destroy(&args);
  }
  if (executable_ != nullptr) {
    PJRT_LoadedExecutable_Destroy_Args args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Destroy_Args),
        .extension_start = nullptr,
        .executable = executable_};
    PJRT_Error* error = api()->PJRT_LoadedExecutable_Destroy(&args);
    if (error != nullptr) {
      PJRT_Error_Destroy_Args destroy = {
          .struct_size = sizeof(PJRT_Error_Destroy_Args),
          .extension_start = nullptr,
          .error = error};
      api()->PJRT_Error_Destroy(&destroy);
    }
  }
  for (double* p : input_arenas_) {
    free(p);
  }
  for (double* p : output_arenas_) {
    free(p);
  }
}

void Function::load_metadata(const std::string& base_name) {
  const std::string path = base_name + ".json";
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  nlohmann::json meta;
  file >> meta;
  if (!meta.contains("args_info") || !meta["args_info"].contains("sizes")) {
    throw std::runtime_error("File does not contain args_info/sizes: " + path);
  }
  if (!meta.contains("out_info") || !meta["out_info"].contains("sizes")) {
    throw std::runtime_error("File does not contain out_info/sizes: " + path);
  }
  input_sizes_ = meta["args_info"]["sizes"].get<std::vector<std::size_t>>();
  output_sizes_ = meta["out_info"]["sizes"].get<std::vector<std::size_t>>();
}

void Function::load_executable(const std::string& base_name) {
  const std::string path = base_name + ".binpb";
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  const std::vector<char> blob(std::istreambuf_iterator<char>(file), {});
  if (blob.empty()) {
    throw std::runtime_error("File is empty: " + path);
  }

  PJRT_Executable_DeserializeAndLoad_Args args = {
      .struct_size = sizeof(PJRT_Executable_DeserializeAndLoad_Args),
      .extension_start = nullptr,
      .client = runtime_.client(),
      .serialized_executable = blob.data(),
      .serialized_executable_size = blob.size(),
      .loaded_executable = nullptr};  // out
  check_error(api()->PJRT_Executable_DeserializeAndLoad(&args));
  executable_ = args.loaded_executable;
}

void Function::allocate_arenas() {
  input_arenas_.reserve(input_sizes_.size());
  for (std::size_t n : input_sizes_) {
    input_arenas_.push_back(alloc_arena(n));
  }
  output_arenas_.reserve(output_sizes_.size());
  for (std::size_t n : output_sizes_) {
    output_arenas_.push_back(alloc_arena(n));
  }
}

void Function::wrap_inputs() {
  input_buffers_.reserve(input_sizes_.size());
  for (std::size_t i = 0; i < input_sizes_.size(); ++i) {
    const std::size_t size = input_sizes_[i];
    const int64_t dim = static_cast<int64_t>(size);
    PJRT_Client_BufferFromHostBuffer_Args args = {
        .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
        .client = runtime_.client(),
        .data = input_arenas_[i],
        .type = PJRT_Buffer_Type::PJRT_Buffer_Type_F64,
        .dims = &dim,
        .num_dims = size == 0 ? 0u : 1u,
        .byte_strides = nullptr,  // dense layout
        .num_byte_strides = 0,
        // The buffer aliases the arena for its whole lifetime, so a call costs
        // no host-to-device copy. The runtime promises not to write it; we
        // write it only between calls, when nothing is in flight.
        .host_buffer_semantics = PJRT_HostBufferSemantics::
            PJRT_HostBufferSemantics_kImmutableZeroCopy,
        .device = runtime_.device(),
        .memory = nullptr,
        .device_layout = nullptr,          // dense layout
        .done_with_host_buffer = nullptr,  // out
        .buffer = nullptr};                // out
    check_error(api()->PJRT_Client_BufferFromHostBuffer(&args));

    // Zero copy still produces a "done with host buffer" event, which fires
    // when the buffer is destroyed. Nothing waits on it, so release it now.
    if (args.done_with_host_buffer != nullptr) {
      PJRT_Event_Destroy_Args destroy = {
          .struct_size = sizeof(PJRT_Event_Destroy_Args),
          .extension_start = nullptr,
          .event = args.done_with_host_buffer};
      api()->PJRT_Event_Destroy(&destroy);
    }
    input_buffers_.push_back(args.buffer);
  }
}

void Function::call() {
  PJRT_Buffer** input_list = input_buffers_.data();
  PJRT_Buffer** output_list = output_buffers_.data();
  PJRT_Event* complete = nullptr;

  PJRT_LoadedExecutable_Execute_Args args = {
      .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
      .extension_start = nullptr,
      .executable = executable_,
      .options = &execute_options_,
      .argument_lists = &input_list,
      .num_devices = 1,
      .num_args = input_buffers_.size(),
      .output_lists = &output_list,           // out
      .device_complete_events = &complete,    // out
      .execute_device = nullptr};
  check_error(api()->PJRT_LoadedExecutable_Execute(&args));

  if (complete != nullptr) {
    PJRT_Event_Await_Args await_args = {
        .struct_size = sizeof(PJRT_Event_Await_Args),
        .extension_start = nullptr,
        .event = complete};
    PJRT_Error* error = api()->PJRT_Event_Await(&await_args);

    PJRT_Event_Destroy_Args destroy_args = {
        .struct_size = sizeof(PJRT_Event_Destroy_Args),
        .extension_start = nullptr,
        .event = complete};
    api()->PJRT_Event_Destroy(&destroy_args);
    check_error(error);
  }

  // Outputs live in device memory, which on CPU is ordinary memory, so they
  // are copied straight out rather than round-tripped through
  // `PJRT_Buffer_ToHostBuffer` and its event.
  for (std::size_t i = 0; i < output_buffers_.size(); ++i) {
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args ptr_args = {
        .struct_size =
            sizeof(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args),
        .extension_start = nullptr,
        .buffer = output_buffers_[i],
        .device_memory_ptr = nullptr};  // out
    check_error(
        api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&ptr_args));

    const std::size_t n = output_sizes_[i] == 0 ? 1 : output_sizes_[i];
    std::memcpy(output_arenas_[i], ptr_args.device_memory_ptr,
                n * sizeof(double));

    PJRT_Buffer_Destroy_Args destroy_args = {
        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
        .extension_start = nullptr,
        .buffer = output_buffers_[i]};
    api()->PJRT_Buffer_Destroy(&destroy_args);
    output_buffers_[i] = nullptr;
  }
}

}  // namespace pjrt
