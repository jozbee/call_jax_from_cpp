#include "pjrt_exec/runtime.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "isa.hpp"
#include "nlohmann/json.hpp"

// The build defines this to wherever `make plugin` puts the shared object.  An
// empty default is deliberate: a build that did not fetch a plugin should say
// so through `LoadError`, not through a dlopen of "".
#ifndef PJRT_EXEC_DEFAULT_PLUGIN_PATH
#define PJRT_EXEC_DEFAULT_PLUGIN_PATH ""
#endif

namespace pjrt {
namespace {

/// `xla::cpu::Align()`.  Anything below `xla::cpu::MinAlign()` makes XLA copy
/// the host buffer instead of aliasing it, silently -- the zero-copy path
/// disappears with no error and no log line, and only the latency changes.
constexpr std::size_t kArenaAlignment = 64;

/// The oldest PJRT C API minor version this code has ever been run against.
/// Older plugins are refused rather than probed: the `Args` structs this file
/// fills in have grown fields since, and a plugin that predates them reads the
/// ones it knows and ignores the rest, which is exactly the sort of half-right
/// behaviour that shows up later as a wrong answer.
constexpr int kOldestSupportedApiMinor = 90;

/// Default serialized `CompileOptionsProto` for the `.mlirbc` fallback.
///
/// Never pass empty compile options: `ExecutableBuildOptionsFromProto` copies
/// `num_replicas` and `num_partitions` straight out of the proto, including
/// when they are absent and therefore zero, and a build with zero replicas
/// fails deep inside the compiler.
///
/// The bytes encode `{executable_build_options {device_ordinal: -1,
/// num_replicas: 1, num_partitions: 1, use_shardy_partitioner: true}}`:
///
///   1a 12                          field 3 (executable_build_options), wire
///                                  type 2 (length-delimited), 18 bytes
///     08 ff ff ff ff ff ff ff ff ff 01   field 1 (device_ordinal), varint,
///                                  -1 sign-extended to 64 bits
///     20 01                        field 4 (num_replicas) = 1
///     28 01                        field 5 (num_partitions) = 1
///     98 01 01                     field 19 (use_shardy_partitioner) = true
///                                  (key 152 = (19 << 3) | 0, itself a varint)
///
/// To regenerate: build the CompileOptionsProto in Python and print
/// `SerializeToString()`.  Field numbers come from
/// `xla/pjrt/proto/compile_options.proto`.
constexpr unsigned char kDefaultCompileOptions[] = {
    0x1a, 0x12, 0x08, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x01, 0x20, 0x01, 0x28,
    0x01, 0x98, 0x01, 0x01};

// Create-option names.  String literals so the `const char*` a PJRT_NamedValue
// holds stays valid however the option vector is shuffled during the
// drop-and-retry loop.
constexpr const char* kOptionAsynchronous = "asynchronous";
constexpr const char* kOptionCpuDeviceCount = "cpu_device_count";
constexpr const char* kOptionMaxInflight = "max_inflight_computations";

constexpr const char* kAttrSynchronous = "supports_synchronous_execution";
constexpr const char* kAttrMaxInflight = "supports_max_inflight_computations";

/// The element types `DType` covers, named the way an error message should
/// name them.  One string rather than eleven so the two messages that list
/// them cannot drift apart.
constexpr const char* kSupportedTypes =
    "bool, int8..int64, uint8..uint64, float32, float64";

//////////////////////
// error management //
//////////////////////

/// Destroy a `PJRT_Error`, tolerating null.  Every path that inspects an error
/// ends here; the previous version of this file did not, and leaked the error
/// object `PJRT_Plugin_Attributes` returns on a plugin that does not implement
/// it.
void destroy_error(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return;
  }
  PJRT_Error_Destroy_Args args{};
  args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  args.error = error;
  api->PJRT_Error_Destroy(&args);
}

/// Copy out an error's message without consuming it.  The string PJRT hands
/// back has the lifetime of the error, so it has to be copied before the error
/// is destroyed.
std::string error_message(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return "no error";
  }
  PJRT_Error_Message_Args args{};
  args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  args.error = error;
  api->PJRT_Error_Message(&args);
  if (args.message == nullptr) {
    return "(the plugin returned an error with no message)";
  }
  return std::string(args.message, args.message_size);
}

/// Read an error's status code, without consuming it.  A failure to read the
/// code is itself reported as an error object, which is destroyed here.
PJRT_Error_Code error_code(const PJRT_Api* api, PJRT_Error* error) {
  if (error == nullptr) {
    return PJRT_Error_Code_OK;
  }
  PJRT_Error_GetCode_Args args{};
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.error = error;
  args.code = PJRT_Error_Code_UNKNOWN;
  destroy_error(api, api->PJRT_Error_GetCode(&args));
  return args.code;
}

/**
 * @brief Owns a `PJRT_Error*` and destroys it.
 *
 * `pjrt::Error` consumes an error and is therefore only usable where the error
 * is about to be thrown.  Everywhere else -- a call whose failure is a
 * fallback rather than a fault, a destructor that has nowhere to throw to --
 * the error still has to be freed, and this is what frees it.
 *
 * `release()` hands ownership back for the one case that does throw:
 * `throw Error(api, holder.release())`.
 */
class ErrorHolder {
 public:
  ErrorHolder(const PJRT_Api* api, PJRT_Error* error)
      : api_(api), error_(error) {}
  ~ErrorHolder() { destroy_error(api_, error_); }

  ErrorHolder(const ErrorHolder&) = delete;
  ErrorHolder& operator=(const ErrorHolder&) = delete;

  explicit operator bool() const { return error_ != nullptr; }

  std::string message() const { return error_message(api_, error_); }
  PJRT_Error_Code code() const { return error_code(api_, error_); }

  /// Give up ownership; the caller must destroy the error or hand it to
  /// `pjrt::Error`, which does.
  PJRT_Error* release() {
    PJRT_Error* error = error_;
    error_ = nullptr;
    return error;
  }

 private:
  const PJRT_Api* api_;
  PJRT_Error* error_;
};

////////////////////////
// api function table //
////////////////////////

/// One entry of the table of PJRT functions this file calls.
struct ApiFunction {
  std::size_t offset;  ///< Byte offset of the pointer within `PJRT_Api`.
  const char* name;    ///< Its spelling, for the error message.
};

#define PJRT_EXEC_API_FN(field) \
  ApiFunction { offsetof(PJRT_Api, field), #field }

/// Every PJRT entry point this translation unit dereferences.  A plugin that
/// leaves one of them null is refused at load with the name of the first one
/// missing, because the alternative is a null call somewhere later with no
/// indication of which function was absent.
///
/// The optional ones are deliberately absent from this list:
/// `PJRT_Executable_Fingerprint` and `PJRT_LoadedExecutable_Fingerprint` are
/// documented as "may not be implemented by all platforms", and their absence
/// costs a diagnostic string, not correctness.
constexpr ApiFunction kRequiredApiFunctions[] = {
    PJRT_EXEC_API_FN(PJRT_Error_Destroy),
    PJRT_EXEC_API_FN(PJRT_Error_Message),
    PJRT_EXEC_API_FN(PJRT_Error_GetCode),
    PJRT_EXEC_API_FN(PJRT_Plugin_Initialize),
    PJRT_EXEC_API_FN(PJRT_Plugin_Attributes),
    PJRT_EXEC_API_FN(PJRT_Event_Destroy),
    PJRT_EXEC_API_FN(PJRT_Event_Await),
    PJRT_EXEC_API_FN(PJRT_Client_Create),
    PJRT_EXEC_API_FN(PJRT_Client_Destroy),
    PJRT_EXEC_API_FN(PJRT_Client_PlatformName),
    PJRT_EXEC_API_FN(PJRT_Client_PlatformVersion),
    PJRT_EXEC_API_FN(PJRT_Client_Devices),
    PJRT_EXEC_API_FN(PJRT_Client_Compile),
    PJRT_EXEC_API_FN(PJRT_Client_BufferFromHostBuffer),
    PJRT_EXEC_API_FN(PJRT_Executable_Destroy),
    PJRT_EXEC_API_FN(PJRT_Executable_NumOutputs),
    PJRT_EXEC_API_FN(PJRT_Executable_OutputElementTypes),
    PJRT_EXEC_API_FN(PJRT_Executable_OutputDimensions),
    PJRT_EXEC_API_FN(PJRT_Executable_DeserializeAndLoad),
    PJRT_EXEC_API_FN(PJRT_LoadedExecutable_Destroy),
    PJRT_EXEC_API_FN(PJRT_LoadedExecutable_GetExecutable),
    PJRT_EXEC_API_FN(PJRT_LoadedExecutable_Execute),
    PJRT_EXEC_API_FN(PJRT_Buffer_Destroy),
    PJRT_EXEC_API_FN(PJRT_Buffer_OpaqueDeviceMemoryDataPointer),
};

#undef PJRT_EXEC_API_FN

static_assert(sizeof(void*) == sizeof(void (*)()),
              "this file reads PJRT_Api function pointers through a void*");

/// Whether the pointer at `offset` in `api` is present and non-null.
///
/// `struct_size` is the forwards-compatibility mechanism the PJRT C API is
/// built on: a plugin compiled against an older header publishes a shorter
/// table, and everything past its `struct_size` is memory that belongs to
/// somebody else.  A field beyond the end therefore counts as absent rather
/// than as whatever byte pattern happens to be there.
bool api_fn_present(const PJRT_Api* api, std::size_t offset) {
  if (offset + sizeof(void*) > api->struct_size) {
    return false;
  }
  const void* fn = nullptr;
  std::memcpy(&fn, reinterpret_cast<const char*>(api) + offset, sizeof(fn));
  return fn != nullptr;
}

///////////////////
// file plumbing //
///////////////////

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Read a whole file as bytes.  Returns false when it cannot be opened, which
/// the caller reports with more context than an errno string would carry.
bool read_file(const std::string& path, std::string* out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>());
  return !out->empty();
}

bool file_exists(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return static_cast<bool>(file);
}

/// Where a `Function`'s files live, derived from the `base_path` a caller
/// passed to the constructor.
struct ArtifactPaths {
  std::string sidecar;    ///< `.../trajopt.json`
  std::string directory;  ///< `.../`, with its separator, or empty
  std::string stem;       ///< `trajopt`
};

/// Accept both spellings of `base_path`: a bare `artifacts/trajopt` and a
/// fully written `artifacts/trajopt.json`.  The second exists because a shell
/// completes file names, not stems, and the first artifact anybody types is
/// the sidecar.
ArtifactPaths resolve_paths(const std::string& base_path) {
  ArtifactPaths paths;
  paths.sidecar =
      ends_with(base_path, ".json") ? base_path : base_path + ".json";

  const std::size_t slash = paths.sidecar.find_last_of('/');
  paths.directory = slash == std::string::npos
                        ? std::string()
                        : paths.sidecar.substr(0, slash + 1);

  const std::string file = paths.sidecar.substr(paths.directory.size());
  paths.stem = file.substr(0, file.size() - std::strlen(".json"));
  return paths;
}

/// Resolve an artifact name from the sidecar against the sidecar's own
/// directory, so a bundle of artifacts can be moved or copied as a unit.
std::string resolve_relative(const std::string& directory,
                             const std::string& name) {
  if (name.empty() || name.front() == '/') {
    return name;
  }
  return directory + name;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw LoadError("cannot open sidecar " + path + ": " +
                    std::strerror(errno));
  }
  try {
    nlohmann::json parsed;
    file >> parsed;
    return parsed;
  } catch (const nlohmann::json::exception& error) {
    throw LoadError(path + " is not valid JSON: " + error.what());
  }
}

////////////////////////
// protobuf envelope  //
////////////////////////

/// Read one base-128 varint at `pos`, advancing it.  False on a truncated or
/// over-long varint, which is the scanner's cue to give up rather than guess.
bool read_varint(const std::string& data, std::size_t* pos,
                 std::uint64_t* value) {
  *value = 0;
  int shift = 0;
  while (*pos < data.size()) {
    const std::uint8_t byte = static_cast<std::uint8_t>(data[(*pos)++]);
    if (shift > 63) {
      return false;
    }
    *value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) {
      return true;
    }
    shift += 7;
  }
  return false;
}

/**
 * @brief Pull the compile options back out of a serialized executable.
 *
 * A `.binpb` is an `ExecutableAndOptionsProto`: field 1 is the serialized
 * executable, field 2 the `CompileOptionsProto` it was built with.  When the
 * ISA guard sends us to the `.mlirbc` instead, compiling with those same
 * options is the closest this can get to the program the exporter tested.
 *
 * A wire-format scan rather than a protobuf dependency: two field numbers and
 * four wire types is the whole grammar needed, and linking protobuf into a
 * header-only-plus-dlopen project to read one field is a poor trade.  Wire
 * types 3 and 4 (the deprecated groups) end the scan -- they nest, this does
 * not, and a wrong answer here would be a program compiled with somebody
 * else's options.
 */
bool compile_options_from_envelope(const std::string& envelope,
                                   std::string* out) {
  std::size_t pos = 0;
  while (pos < envelope.size()) {
    std::uint64_t key = 0;
    if (!read_varint(envelope, &pos, &key)) {
      return false;
    }
    const std::uint64_t field = key >> 3;
    switch (key & 0x7) {
      case 0: {  // varint
        std::uint64_t ignored = 0;
        if (!read_varint(envelope, &pos, &ignored)) {
          return false;
        }
        break;
      }
      case 1:  // 64-bit
        if (envelope.size() - pos < 8) {
          return false;
        }
        pos += 8;
        break;
      case 2: {  // length-delimited
        std::uint64_t length = 0;
        if (!read_varint(envelope, &pos, &length) ||
            length > static_cast<std::uint64_t>(envelope.size() - pos)) {
          return false;
        }
        if (field == 2) {
          out->assign(envelope, pos, static_cast<std::size_t>(length));
          return !out->empty();
        }
        pos += static_cast<std::size_t>(length);
        break;
      }
      case 5:  // 32-bit
        if (envelope.size() - pos < 4) {
          return false;
        }
        pos += 4;
        break;
      default:
        return false;
    }
  }
  return false;
}

////////////////////
// value checking //
////////////////////

/// `dtype[d0,d1]`, or `dtype[]` for a scalar -- the spelling the metadata
/// mismatch messages compare in.
std::string shape_string(DType dtype, const std::vector<std::int64_t>& dims) {
  std::string text = dtype_name(dtype);
  text += '[';
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      text += ',';
    }
    text += std::to_string(dims[i]);
  }
  text += ']';
  return text;
}

const char* nonfinite_name(double value) {
  if (std::isnan(value)) {
    return "nan";
  }
  return value < 0 ? "-inf" : "inf";
}

[[noreturn]] void throw_nonfinite(const char* role, std::size_t index,
                                  const std::string& name, std::size_t element,
                                  double value) {
  throw std::domain_error(std::string(role) + " " + std::to_string(index) +
                          " ('" + name + "') element " +
                          std::to_string(element) + " is " +
                          nonfinite_name(value));
}

[[noreturn]] void throw_bad_bool(const char* role, std::size_t index,
                                 const std::string& name, std::size_t element,
                                 unsigned value) {
  throw std::domain_error(std::string(role) + " " + std::to_string(index) +
                          " ('" + name + "') element " +
                          std::to_string(element) + " is " +
                          std::to_string(value) +
                          ", bool arenas must hold 0 or 1");
}

[[noreturn]] void throw_reentered(const std::string& name) {
  throw std::logic_error("Function '" + name + "'::call() re-entered");
}

/**
 * @brief Audit one arena: no nan or inf in a float, nothing but 0 or 1 in a
 *        bool.
 *
 * The bool case is not pedantry.  XLA does not normalize a `PRED` byte, so a
 * stray 2 can make one predicate read true and another read false within a
 * single computation, and the result is a wrong answer with no failure
 * anywhere to attach a bug report to.
 *
 * Integer arenas hold no invalid bit patterns and are skipped.
 */
void scan_arena(const char* role, std::size_t index, const ArraySpec& spec,
                const void* arena) {
  switch (spec.dtype) {
    case DType::Float64: {
      const double* values = static_cast<const double*>(arena);
      for (std::size_t i = 0; i < spec.numel; ++i) {
        if (!std::isfinite(values[i])) {
          throw_nonfinite(role, index, spec.name, i, values[i]);
        }
      }
      break;
    }
    case DType::Float32: {
      const float* values = static_cast<const float*>(arena);
      for (std::size_t i = 0; i < spec.numel; ++i) {
        if (!std::isfinite(values[i])) {
          throw_nonfinite(role, index, spec.name, i,
                          static_cast<double>(values[i]));
        }
      }
      break;
    }
    case DType::Bool: {
      const unsigned char* values = static_cast<const unsigned char*>(arena);
      for (std::size_t i = 0; i < spec.numel; ++i) {
        if (values[i] > 1) {
          throw_bad_bool(role, index, spec.name, i, values[i]);
        }
      }
      break;
    }
    default:
      break;
  }
}

/////////////
// arenas  //
/////////////

/// One 64-byte-aligned, zeroed arena of at least `nbytes`.
///
/// Rounded up so that the tail of the last cache line belongs to us: XLA's
/// vectorized epilogues read whole vectors, and a read that runs past the end
/// of an exactly sized allocation is a valgrind report at best.  A zero-byte
/// array still gets a line, because a null arena would turn every accessor
/// into a special case.
void* alloc_arena(std::size_t nbytes) {
  const std::size_t wanted = nbytes == 0 ? 1 : nbytes;
  const std::size_t rounded =
      (wanted + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
  void* memory = nullptr;
  if (posix_memalign(&memory, kArenaAlignment, rounded) != 0 ||
      memory == nullptr) {
    throw LoadError("could not allocate a " + std::to_string(rounded) +
                    "-byte aligned arena");
  }
  std::memset(memory, 0, rounded);
  return memory;
}

/// Free everything a `Function` owns.  Shared by the destructor and by the
/// constructor's failure path, which the destructor will never run for.
void release_resources(const PJRT_Api* api,
                       std::vector<PJRT_Buffer*>* input_buffers,
                       std::vector<PJRT_Buffer*>* output_buffers,
                       PJRT_LoadedExecutable** executable,
                       std::vector<void*>* input_arenas,
                       std::vector<void*>* output_arenas) {
  for (std::vector<PJRT_Buffer*>* buffers : {input_buffers, output_buffers}) {
    for (PJRT_Buffer*& buffer : *buffers) {
      if (buffer == nullptr) {
        continue;
      }
      PJRT_Buffer_Destroy_Args args{};
      args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      args.buffer = buffer;
      destroy_error(api, api->PJRT_Buffer_Destroy(&args));
      buffer = nullptr;
    }
  }

  if (*executable != nullptr) {
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = *executable;
    destroy_error(api, api->PJRT_LoadedExecutable_Destroy(&args));
    *executable = nullptr;
  }

  for (std::vector<void*>* arenas : {input_arenas, output_arenas}) {
    for (void*& arena : *arenas) {
      std::free(arena);
      arena = nullptr;
    }
  }
}

}  // namespace

////////////
// errors //
////////////

Error::Error(const PJRT_Api* api, PJRT_Error* error)
    : std::runtime_error(error_message(api, error)),
      code_(error_code(api, error)) {
  destroy_error(api, error);
}

void check_error(const PJRT_Api* api, PJRT_Error* error) {
  if (error != nullptr) {
    throw Error(api, error);
  }
}

/////////////
// runtime //
/////////////

namespace {

/// Where to look for the plugin, in the order a caller expects: what they
/// asked for, then what the environment says, then what this build was
/// configured with.
std::string resolve_plugin_path(const RuntimeOptions& options) {
  if (!options.plugin_path.empty()) {
    return options.plugin_path;
  }
  const char* from_env = std::getenv("PJRT_CPU_PLUGIN");
  if (from_env != nullptr && from_env[0] != '\0') {
    return from_env;
  }
  const std::string built_in = PJRT_EXEC_DEFAULT_PLUGIN_PATH;
  if (!built_in.empty()) {
    return built_in;
  }
  throw LoadError(
      "no PJRT CPU plugin: set RuntimeOptions::plugin_path or "
      "$PJRT_CPU_PLUGIN, or run make plugin");
}

/// Render a plugin attribute for `describe()`.  Attributes are the only
/// self-description a plugin offers, so all of them are kept, whatever their
/// type.
std::string named_value_to_string(const PJRT_NamedValue& value) {
  switch (value.type) {
    case PJRT_NamedValue_kString:
      return std::string(value.string_value, value.value_size);
    case PJRT_NamedValue_kInt64:
      return std::to_string(value.int64_value);
    case PJRT_NamedValue_kInt64List: {
      std::string text;
      for (std::size_t i = 0; i < value.value_size; ++i) {
        if (i != 0) {
          text += ',';
        }
        text += std::to_string(value.int64_array_value[i]);
      }
      return text;
    }
    case PJRT_NamedValue_kFloat:
      return std::to_string(value.float_value);
    case PJRT_NamedValue_kBool:
      return value.bool_value ? "true" : "false";
  }
  return "?";
}

/// Pull the option name out of `Unexpected option name passed to
/// PJRT_Client_Create: <name>`.
///
/// The last colon rather than a fixed prefix: the plugin's message may arrive
/// with a status prefix in front of it, and an option name never contains a
/// colon.  An empty return means the message did not name anything this code
/// can act on, which stops the retry loop rather than dropping a guess.
std::string offending_option_name(const std::string& message) {
  const std::size_t colon = message.rfind(':');
  if (colon == std::string::npos) {
    return {};
  }
  std::string name = message.substr(colon + 1);
  const std::string junk = " \t\r\n.\"'";
  const std::size_t first = name.find_first_not_of(junk);
  if (first == std::string::npos) {
    return {};
  }
  return name.substr(first, name.find_last_not_of(junk) - first + 1);
}

/// A `PJRT_NamedValue` naming a string literal, filled field by field because
/// this struct's trailing fields have moved before.
PJRT_NamedValue bool_option(const char* name, bool value) {
  PJRT_NamedValue option{};
  option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  option.name = name;
  option.name_size = std::strlen(name);
  option.type = PJRT_NamedValue_kBool;
  option.bool_value = value;
  option.value_size = 1;
  return option;
}

PJRT_NamedValue int64_option(const char* name, std::int64_t value) {
  PJRT_NamedValue option{};
  option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  option.name = name;
  option.name_size = std::strlen(name);
  option.type = PJRT_NamedValue_kInt64;
  option.int64_value = value;
  option.value_size = 1;
  return option;
}

}  // namespace

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  load_plugin(resolve_plugin_path(options_));
  // Before the client, not after: the attributes decide which create options
  // are safe to send, and sending an unknown one is now fatal.
  query_attributes();

  // A constructor that throws gets no destructor, so anything already acquired
  // has to be released here. Creating a client starts XLA's thread pools, and
  // `create_client()` can still throw after it succeeds -- on a device count
  // of zero, for one -- so without this a caller probing plugin paths or
  // option combinations and catching the failure accumulates two threads and
  // several megabytes per attempt. The plugin handle is deliberately not
  // closed: XLA keeps process-lifetime statics behind it.
  try {
    create_client();
  } catch (...) {
    destroy_client();
    throw;
  }
}

void Runtime::destroy_client() noexcept {
  if (client_ == nullptr) {
    return;
  }
  PJRT_Client_Destroy_Args args{};
  args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  args.client = client_;
  destroy_error(api_, api_->PJRT_Client_Destroy(&args));
  client_ = nullptr;
  device_ = nullptr;
}

Runtime::~Runtime() {
  destroy_client();

  // `dl_handle_` is deliberately never closed.  XLA leaves statics behind the
  // plugin boundary that live as long as the process -- LLVM's target
  // registry, the CPU feature tables, thread-local allocator state -- and
  // unmapping the code they point into ends the process on the next atexit
  // handler rather than here where it could be diagnosed.
}

void Runtime::load_plugin(const std::string& path) {
  // RTLD_NOW so an unresolved symbol is a failure here, at startup, instead of
  // a crash on the first call that reaches it.  RTLD_LOCAL so the plugin's own
  // copy of LLVM does not join the process's global symbol namespace and get
  // bound to by something else.
  dl_handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (dl_handle_ == nullptr) {
    const char* why = dlerror();
    throw LoadError("dlopen(" + path +
                    "): " + (why == nullptr ? "unknown error" : why));
  }

  dlerror();  // Clear any stale message so the check below tests this dlsym.
  void* symbol = dlsym(dl_handle_, "GetPjrtApi");
  const char* dlsym_error = dlerror();
  if (symbol == nullptr || dlsym_error != nullptr) {
    throw LoadError(
        path + ": no GetPjrtApi symbol (" +
        (dlsym_error == nullptr ? "symbol is null" : dlsym_error) +
        "); a PJRT plugin exports exactly this one function");
  }

  // A function pointer is not an object pointer; copying the bytes is the
  // portable spelling of the cast POSIX guarantees works.
  const PJRT_Api* (*get_api)() = nullptr;
  std::memcpy(&get_api, &symbol, sizeof(symbol));
  api_ = get_api();
  if (api_ == nullptr) {
    throw LoadError(path + ": GetPjrtApi returned null");
  }

  plugin_.path = path;
  plugin_.api_major = api_->pjrt_api_version.major_version;
  plugin_.api_minor = api_->pjrt_api_version.minor_version;

  // The major version gates ABI compatibility outright: a different major
  // means fields have moved, and every struct in this file would be filled in
  // at the wrong offsets.
  if (plugin_.api_major != PJRT_API_MAJOR) {
    throw LoadError(path + ": PJRT C API major version " +
                    std::to_string(plugin_.api_major) +
                    " but this build needs " +
                    std::to_string(PJRT_API_MAJOR));
  }
  if (plugin_.api_minor < kOldestSupportedApiMinor) {
    throw LoadError(path + ": PJRT C API 0." +
                    std::to_string(plugin_.api_minor) +
                    " is older than the oldest version this code has been "
                    "tested against (0." +
                    std::to_string(kOldestSupportedApiMinor) + ")");
  }

  // Checked before `PJRT_Plugin_Initialize`, which is itself in the table: a
  // null function pointer found by calling it is a segfault with no name
  // attached, and the whole point of the table is to name the first thing
  // missing.
  for (const ApiFunction& fn : kRequiredApiFunctions) {
    if (!api_fn_present(api_, fn.offset)) {
      throw LoadError(path + ": PJRT plugin does not implement " +
                      std::string(fn.name) +
                      ", which pjrt_exec calls (plugin reports PJRT C API 0." +
                      std::to_string(plugin_.api_minor) + ")");
    }
  }

  PJRT_Plugin_Initialize_Args args{};
  args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  check_error(api_, api_->PJRT_Plugin_Initialize(&args));
}

void Runtime::query_attributes() {
  PJRT_Plugin_Attributes_Args args{};
  args.struct_size = PJRT_Plugin_Attributes_Args_STRUCT_SIZE;

  // A plugin that cannot describe itself is not a fatal problem, it is a
  // plugin that advertises nothing -- which is exactly the conservative state
  // the members already hold.  The error still has to be freed, and that is
  // what the holder is for.
  ErrorHolder error(api_, api_->PJRT_Plugin_Attributes(&args));
  if (error) {
    return;
  }

  plugin_.attributes.reserve(args.num_attributes);
  for (std::size_t i = 0; i < args.num_attributes; ++i) {
    const PJRT_NamedValue& attribute = args.attributes[i];
    const std::string name(attribute.name, attribute.name_size);
    plugin_.attributes.emplace_back(name, named_value_to_string(attribute));
    if (name == kAttrSynchronous) {
      plugin_.advertises_synchronous_execution = true;
    } else if (name == kAttrMaxInflight) {
      plugin_.advertises_max_inflight = true;
    }
  }
}

void Runtime::create_client() {
  // `DefaultThreadPoolSize()` reads this environment variable, and it reads it
  // while the client is being created, so it has to be set first -- there is
  // no create option and no later hook that reaches the same knob.
  if (options_.worker_threads > 0) {
    setenv("PJRT_NPROC", std::to_string(options_.worker_threads).c_str(),
           /*overwrite=*/1);
  }

  std::vector<PJRT_NamedValue> create_options;
  create_options.push_back(
      int64_option(kOptionCpuDeviceCount, options_.cpu_device_count));
  if (options_.synchronous) {
    // The option is spelled in the negative: `asynchronous = false` is what
    // asks for inline execution.
    create_options.push_back(bool_option(kOptionAsynchronous, false));
  }
  // Withheld unless advertised.  As of XLA dcf304bc the CPU plugin returns
  // InvalidArgument for a create option it does not recognize, where it used
  // to ignore one silently, so sending this speculatively would cost the whole
  // client rather than just the option.
  if (options_.max_inflight_computations > 0 &&
      plugin_.advertises_max_inflight) {
    create_options.push_back(
        int64_option(kOptionMaxInflight, options_.max_inflight_computations));
  }

  bool asynchronous_rejected = false;
  for (;;) {
    PJRT_Client_Create_Args args{};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    args.create_options =
        create_options.empty() ? nullptr : create_options.data();
    args.num_options = create_options.size();
    ErrorHolder error(api_, api_->PJRT_Client_Create(&args));
    if (!error) {
      client_ = args.client;
      break;
    }

    const std::string message = error.message();
    const bool is_unknown_option =
        error.code() == PJRT_Error_Code_INVALID_ARGUMENT &&
        message.find("Unexpected option name") != std::string::npos;
    if (!options_.allow_async_fallback || !is_unknown_option) {
      throw Error(api_, error.release());
    }

    const std::string offender = offending_option_name(message);
    const auto victim = std::find_if(
        create_options.begin(), create_options.end(),
        [&offender](const PJRT_NamedValue& option) {
          return offender == std::string(option.name, option.name_size);
        });
    // Nothing to drop means the plugin is objecting to an option this code did
    // not send, or naming it in a way this code cannot parse.  Either way the
    // retry would send the same thing again, so stop and report.
    if (victim == create_options.end()) {
      throw Error(api_, error.release());
    }
    if (offender == kOptionAsynchronous) {
      asynchronous_rejected = true;
    }
    // Erasing is also what bounds the loop: an option can only be dropped
    // once, so a plugin that keeps naming the same one runs out of matches.
    create_options.erase(victim);
  }

  if (!options_.synchronous) {
    sync_mode_ = SyncMode::Async;
  } else if (asynchronous_rejected) {
    sync_mode_ = SyncMode::Rejected;
  } else if (plugin_.advertises_synchronous_execution) {
    sync_mode_ = SyncMode::Inline;
  } else {
    // The option survived creation, but this plugin does not carry the marker
    // attribute, so "it did not complain" is the strongest claim available.
    sync_mode_ = SyncMode::Accepted;
  }

  PJRT_Client_Devices_Args device_args{};
  device_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  device_args.client = client_;
  check_error(api_, api_->PJRT_Client_Devices(&device_args));
  if (device_args.num_devices == 0) {
    throw LoadError("PJRT client reported no devices");
  }
  device_ = device_args.devices[0];

  PJRT_Client_PlatformName_Args name_args{};
  name_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  name_args.client = client_;
  check_error(api_, api_->PJRT_Client_PlatformName(&name_args));
  plugin_.platform_name =
      std::string(name_args.platform_name, name_args.platform_name_size);

  PJRT_Client_PlatformVersion_Args version_args{};
  version_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  version_args.client = client_;
  check_error(api_, api_->PJRT_Client_PlatformVersion(&version_args));
  plugin_.platform_version = std::string(version_args.platform_version,
                                         version_args.platform_version_size);
}

std::string Runtime::describe() const {
  std::ostringstream out;
  out << "pjrt_exec " << version() << " on " << plugin_.platform_name;
  if (!plugin_.platform_version.empty()) {
    out << " (" << plugin_.platform_version << ")";
  }
  out << " through " << plugin_.path << ", PJRT C API " << plugin_.api_major
      << "." << plugin_.api_minor;
  if (plugin_.api_minor != vendored_pjrt_api_minor()) {
    out << " against a vendored header for 0." << vendored_pjrt_api_minor()
        << ", so the two may disagree about the trailing fields of the Args"
           " structs";
  }
  out << "; ";

  switch (sync_mode_) {
    case SyncMode::Inline:
      out << "execution is inline on the calling thread (the plugin "
             "advertises " << kAttrSynchronous << ")";
      break;
    case SyncMode::Accepted:
      out << "execution is inline as far as can be told: the plugin took the `"
          << kOptionAsynchronous << "` option but does not advertise "
          << kAttrSynchronous;
      break;
    case SyncMode::Rejected:
      out << "execution is asynchronous: the plugin rejected the `"
          << kOptionAsynchronous
          << "` create option, which was dropped and the client created "
             "without it, leaving the dispatch hand-off in the call path";
      break;
    case SyncMode::Async:
      out << "execution is asynchronous by request";
      break;
  }

  out << "; " << options_.cpu_device_count << " CPU device"
      << (options_.cpu_device_count == 1 ? "" : "s") << ", ";
  if (options_.worker_threads > 0) {
    out << "PJRT_NPROC=" << options_.worker_threads;
  } else {
    out << "XLA's default thread pool size (one thread per core)";
  }

  if (options_.max_inflight_computations > 0 &&
      !plugin_.advertises_max_inflight) {
    out << "; `" << kOptionMaxInflight
        << "` was withheld because the plugin does not advertise "
        << kAttrMaxInflight;
  }
  out << ".";
  return out.str();
}

//////////////
// function //
//////////////

namespace {

std::string format_ms(double milliseconds) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(1);
  out << milliseconds;
  return out.str();
}

/// Parse one `inputs`/`outputs` entry of a schema 2 sidecar.
///
/// Everything here is cross-checked rather than trusted, because the sidecar
/// is the only description of the *inputs* that exists -- the PJRT C API has
/// no parameter-shape query -- and a sidecar that has drifted from its
/// executable turns into a write past the end of an arena, discovered as
/// corrupted output several thousand calls later.
ArraySpec parse_array(const std::string& sidecar_path,
                      const nlohmann::json& entry, const char* role,
                      std::size_t position, bool wants_donation) {
  const std::string where =
      sidecar_path + ": " + role + " " + std::to_string(position);
  if (!entry.is_object()) {
    throw LoadError(where + " is not an object");
  }

  ArraySpec spec;
  spec.name = entry.value("name", std::string(role) + std::to_string(position));

  if (entry.contains("index") &&
      entry["index"].get<std::int64_t>() !=
          static_cast<std::int64_t>(position)) {
    throw LoadError(where + " ('" + spec.name + "') declares index " +
                    std::to_string(entry["index"].get<std::int64_t>()) +
                    ", but entries must be listed in executable order");
  }

  const std::string dtype_text = entry.value("dtype", std::string());
  const std::optional<DType> dtype = parse_dtype(dtype_text);
  if (!dtype) {
    throw LoadError(where + " ('" + spec.name + "') has dtype '" + dtype_text +
                    "', which pjrt_exec does not support (supported: " +
                    kSupportedTypes + ")");
  }
  spec.dtype = *dtype;

  std::size_t numel = 1;
  if (entry.contains("shape")) {
    for (const nlohmann::json& dim : entry["shape"]) {
      const std::int64_t extent = dim.get<std::int64_t>();
      if (extent < 0) {
        throw LoadError(where + " ('" + spec.name + "') has dimension " +
                        std::to_string(extent) +
                        "; pjrt_exec loads static shapes only");
      }
      spec.shape.push_back(extent);
      numel *= static_cast<std::size_t>(extent);
    }
  }
  spec.numel = numel;
  spec.nbytes = numel * itemsize(spec.dtype);

  // `numel` and `nbytes` are redundant with the shape and the dtype on
  // purpose: they are what the loader allocates against, so a sidecar whose
  // arithmetic disagrees with its own shape is rejected rather than
  // reinterpreted.
  if (entry.contains("numel") &&
      entry["numel"].get<std::size_t>() != spec.numel) {
    throw LoadError(where + " ('" + spec.name + "') declares numel " +
                    std::to_string(entry["numel"].get<std::size_t>()) +
                    " but its shape " + shape_string(spec.dtype, spec.shape) +
                    " holds " + std::to_string(spec.numel));
  }
  if (entry.contains("nbytes") &&
      entry["nbytes"].get<std::size_t>() != spec.nbytes) {
    throw LoadError(where + " ('" + spec.name + "') declares nbytes " +
                    std::to_string(entry["nbytes"].get<std::size_t>()) +
                    " but " + std::to_string(spec.numel) + " " +
                    dtype_name(spec.dtype) + " elements are " +
                    std::to_string(spec.nbytes) + " bytes");
  }

  if (wants_donation) {
    spec.donated = entry.value("donated", false);
  }
  return spec;
}

/// Widen a schema 1 `args_info`/`out_info` block into the same view the rest
/// of the loader takes.  v1 recorded one dtype for the whole function and a
/// flat list of sizes, where a size of 0 meant a scalar rather than an empty
/// array -- an ambiguity schema 2 removed by writing the shape out.
std::vector<ArraySpec> parse_v1_arrays(const std::string& sidecar_path,
                                       const nlohmann::json& info,
                                       const char* prefix) {
  const std::string dtype_text = info.value("dtype", std::string("float64"));
  const std::optional<DType> dtype = parse_dtype(dtype_text);
  if (!dtype) {
    throw LoadError(sidecar_path + ": schema 1 dtype '" + dtype_text +
                    "' is not one of " + kSupportedTypes);
  }

  std::vector<ArraySpec> specs;
  if (!info.contains("sizes")) {
    throw LoadError(sidecar_path + ": schema 1 sidecar has no " +
                    std::string(prefix) + " sizes");
  }
  std::size_t position = 0;
  for (const nlohmann::json& size : info["sizes"]) {
    const std::int64_t extent = size.get<std::int64_t>();
    if (extent < 0) {
      throw LoadError(sidecar_path + ": schema 1 size " +
                      std::to_string(extent) + " is negative");
    }
    ArraySpec spec;
    spec.name = std::string(prefix) + std::to_string(position);
    spec.dtype = *dtype;
    if (extent == 0) {
      spec.numel = 1;  // A scalar, not an empty array.
    } else {
      spec.shape.push_back(extent);
      spec.numel = static_cast<std::size_t>(extent);
    }
    spec.nbytes = spec.numel * itemsize(spec.dtype);
    specs.push_back(std::move(spec));
    ++position;
  }
  return specs;
}

/// `PJRT_Executable_Fingerprint` where the plugin has it, the deprecated
/// `PJRT_LoadedExecutable_Fingerprint` where it does not, and an empty string
/// where neither is implemented -- which is allowed, and costs a log line
/// rather than a load.
std::string read_fingerprint(const PJRT_Api* api,
                             PJRT_LoadedExecutable* loaded) {
  if (api_fn_present(api, offsetof(PJRT_Api, PJRT_Executable_Fingerprint))) {
    PJRT_LoadedExecutable_GetExecutable_Args get{};
    get.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get.loaded_executable = loaded;
    ErrorHolder get_error(api, api->PJRT_LoadedExecutable_GetExecutable(&get));
    if (!get_error) {
      PJRT_Executable_Fingerprint_Args args{};
      args.struct_size = PJRT_Executable_Fingerprint_Args_STRUCT_SIZE;
      args.executable = get.executable;
      ErrorHolder error(api, api->PJRT_Executable_Fingerprint(&args));

      // The string has the lifetime of the executable handle, so it is copied
      // before that handle is destroyed, not after.
      std::string fingerprint;
      if (!error && args.executable_fingerprint != nullptr) {
        fingerprint.assign(args.executable_fingerprint,
                           args.executable_fingerprint_size);
      }

      PJRT_Executable_Destroy_Args destroy{};
      destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
      destroy.executable = get.executable;
      destroy_error(api, api->PJRT_Executable_Destroy(&destroy));

      if (!fingerprint.empty()) {
        return fingerprint;
      }
    }
  }

  if (api_fn_present(api,
                     offsetof(PJRT_Api, PJRT_LoadedExecutable_Fingerprint))) {
    PJRT_LoadedExecutable_Fingerprint_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Fingerprint_Args_STRUCT_SIZE;
    args.executable = loaded;
    ErrorHolder error(api, api->PJRT_LoadedExecutable_Fingerprint(&args));
    if (!error && args.executable_fingerprint != nullptr) {
      return std::string(args.executable_fingerprint,
                         args.executable_fingerprint_size);
    }
  }
  return {};
}

}  // namespace

Function::Function(Runtime& runtime, const std::string& base_path,
                   const FunctionOptions& options)
    : runtime_(runtime),
      options_(options),
      debug_(options.debug),
      check_values_(options.check_values) {
  // A constructor that throws gets no destructor, and by this point there may
  // be an executable, a dozen PJRT buffers and two dozen arenas to give back.
  try {
    load_sidecar(base_path);
    load_executable(base_path);
    if (options_.check_metadata) {
      validate_metadata();
    }
    allocate_arenas();
    wrap_inputs();
    output_buffers_.assign(outputs_.size(), nullptr);

    // EVERY input is pinned for the life of the `Function`, including any the
    // export marked as donated.
    //
    // Donation and this design are incompatible, not merely unimplemented.
    // The input buffers are created once in `wrap_inputs()` and reused by
    // every call; a donated buffer is consumed by the execution it is passed
    // to, so the second call would hand XLA a buffer that has already been
    // taken and fail with "Buffer has been deleted or donated" -- after the
    // first call has already succeeded, which makes it look like corruption
    // rather than a configuration error.
    //
    // So `ArraySpec::donated` is reporting only: it says what the export
    // asked for, and this says what the runtime does about it. Exploiting
    // donation would mean recreating input buffers per call, which is the
    // per-call allocation this whole path exists to avoid.
    non_donatable_.reserve(inputs_.size());
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      non_donatable_.push_back(static_cast<std::int64_t>(i));
    }
    execute_options_.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    execute_options_.non_donatable_input_indices =
        non_donatable_.empty() ? nullptr : non_donatable_.data();
    execute_options_.num_non_donatable_input_indices = non_donatable_.size();

    // Warm up on zeroed arenas: faults in every page, resolves the runtime's
    // lazy state, and lets the allocator reach the steady state the caller is
    // about to measure.
    //
    // Value checking is suspended for the duration. The warm-up runs on zeros
    // the caller never chose and throws its results away, and plenty of honest
    // functions produce a non-finite result from an all-zero input -- solving
    // against a zero matrix, dividing by a zero parameter. Checking here would
    // report the runtime's own scratch data as the caller's bad input.
    const bool check_values_during_steady_state = check_values_;
    check_values_ = false;
    for (int attempt = 1; attempt <= options_.warmup_calls; ++attempt) {
      try {
        call();
      } catch (const Error& error) {
        throw LoadError(
            "warm-up call " + std::to_string(attempt) + " of " +
            std::to_string(options_.warmup_calls) + " failed: " +
            error.what() +
            " (an input count/shape/dtype mismatch between the sidecar and "
            "the executable shows up here; the PJRT C API has no "
            "parameter-shape query)");
      }
    }
    check_values_ = check_values_during_steady_state;
  } catch (...) {
    release_resources(runtime_.api(), &input_buffers_, &output_buffers_,
                      &executable_, &input_arenas_, &output_arenas_);
    throw;
  }
}

Function::~Function() {
  release_resources(runtime_.api(), &input_buffers_, &output_buffers_,
                    &executable_, &input_arenas_, &output_arenas_);
}

std::optional<std::size_t> Function::find_input(std::string_view name) const {
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].name == name) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> Function::find_output(std::string_view name) const {
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].name == name) {
      return i;
    }
  }
  return std::nullopt;
}

void Function::load_sidecar(const std::string& base_path) {
  const ArtifactPaths paths = resolve_paths(base_path);
  const nlohmann::json sidecar = read_json(paths.sidecar);

  // Any type error inside the sidecar arrives as a json exception; it is a
  // malformed artifact either way, and the caller should see one exception
  // type for "this artifact cannot be loaded".
  try {
    const int schema = sidecar.value("schema", 1);
    if (schema > 2) {
      throw LoadError("sidecar schema " + std::to_string(schema) +
                      " is newer than this loader (supports 1-2)");
    }

    if (schema >= 2) {
      name_ = sidecar.value("name", paths.stem);
      if (!sidecar.contains("inputs") || !sidecar.contains("outputs")) {
        throw LoadError(paths.sidecar +
                        ": a schema 2 sidecar must have both \"inputs\" and "
                        "\"outputs\"");
      }
      if (!sidecar["inputs"].is_array() || !sidecar["outputs"].is_array()) {
        throw LoadError(paths.sidecar +
                        ": \"inputs\" and \"outputs\" must be arrays, in the "
                        "order the executable takes and returns them");
      }
      std::size_t position = 0;
      for (const nlohmann::json& entry : sidecar["inputs"]) {
        inputs_.push_back(parse_array(paths.sidecar, entry, "input", position++,
                                      /*wants_donation=*/true));
      }
      position = 0;
      for (const nlohmann::json& entry : sidecar["outputs"]) {
        outputs_.push_back(parse_array(paths.sidecar, entry, "output",
                                       position++,
                                       /*wants_donation=*/false));
      }

      // `donation.donate_argnums` is deliberately NOT merged into the
      // per-input flags. It indexes the positional arguments handed to
      // `jax.jit`, while `inputs_` indexes flattened pytree leaves, so one
      // container argument before a donated one shifts every index and the
      // merge would mark the wrong input. The per-input `donated` flag is
      // already in leaf space, which is this loader's space, so it is the only
      // one read here. `donate_argnums` is kept in the sidecar for the reader.
    } else {
      name_ = paths.stem;
      if (!sidecar.contains("args_info") || !sidecar.contains("out_info")) {
        throw LoadError(paths.sidecar +
                        ": a schema 1 sidecar must have both \"args_info\" and "
                        "\"out_info\"");
      }
      inputs_ = parse_v1_arrays(paths.sidecar, sidecar["args_info"], "arg");
      outputs_ = parse_v1_arrays(paths.sidecar, sidecar["out_info"], "out");
    }
  } catch (const nlohmann::json::exception& error) {
    throw LoadError(paths.sidecar + " is not a sidecar this loader can read: " +
                    error.what());
  }

  if (name_.empty()) {
    name_ = paths.stem;
  }
}

void Function::load_executable(const std::string& base_path) {
  const ArtifactPaths paths = resolve_paths(base_path);
  // The sidecar is read a second time rather than kept on the object: the
  // artifact file names and the exporting host's ISA level are the loader's
  // business and nothing on the call path ever looks at them, so they do not
  // earn a place in an object whose members a control loop walks.
  const nlohmann::json sidecar = read_json(paths.sidecar);

  std::string binary_path = paths.directory + paths.stem + ".binpb";
  std::string mlir_path = paths.directory + paths.stem + ".mlirbc";
  std::string exported_isa;
  try {
    if (sidecar.contains("artifacts")) {
      const nlohmann::json& artifacts = sidecar["artifacts"];
      if (artifacts.contains("executable")) {
        binary_path = resolve_relative(
            paths.directory, artifacts["executable"].get<std::string>());
      }
      if (artifacts.contains("mlir")) {
        mlir_path = resolve_relative(paths.directory,
                                     artifacts["mlir"].get<std::string>());
      }
    }
    if (sidecar.contains("export") && sidecar["export"].contains("host") &&
        sidecar["export"]["host"].contains("isa_level")) {
      exported_isa = sidecar["export"]["host"]["isa_level"].get<std::string>();
    }
  } catch (const nlohmann::json::exception& error) {
    throw LoadError(paths.sidecar +
                    ": cannot read the artifact names: " + error.what());
  }

  const PJRT_Api* api = runtime_.api();

  auto deserialize = [&]() {
    std::string blob;
    if (!read_file(binary_path, &blob)) {
      throw LoadError("cannot read " + binary_path +
                      " (missing or empty); export it, or load with "
                      "LoadPolicy::CompileOnly");
    }
    PJRT_Executable_DeserializeAndLoad_Args args{};
    args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
    args.client = runtime_.client();
    args.serialized_executable = blob.data();
    args.serialized_executable_size = blob.size();
    check_error(api, api->PJRT_Executable_DeserializeAndLoad(&args));
    executable_ = args.loaded_executable;
  };

  // Options for the fallback, in descending order of how much they are known
  // to match what the exporter built: what the caller supplied, then what the
  // `.binpb` was built with, then a minimal proto that at least does not ask
  // for zero replicas.
  auto fallback_compile_options = [&]() {
    if (!options_.compile_options.empty()) {
      return options_.compile_options;
    }
    std::string envelope;
    std::string recovered;
    if (read_file(binary_path, &envelope) &&
        compile_options_from_envelope(envelope, &recovered)) {
      return recovered;
    }
    return std::string(reinterpret_cast<const char*>(kDefaultCompileOptions),
                       sizeof(kDefaultCompileOptions));
  };

  auto compile = [&]() {
    std::string bytecode;
    if (!read_file(mlir_path, &bytecode)) {
      throw LoadError("cannot read " + mlir_path +
                      " (missing or empty); export it, or load with "
                      "LoadPolicy::BinaryOnly");
    }
    const std::string compile_options = fallback_compile_options();

    PJRT_Program program{};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = bytecode.data();
    program.code_size = bytecode.size();
    program.format = "mlir";
    program.format_size = std::strlen("mlir");

    PJRT_Client_Compile_Args args{};
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.client = runtime_.client();
    args.program = &program;
    args.compile_options = compile_options.data();
    args.compile_options_size = compile_options.size();

    const auto started = std::chrono::steady_clock::now();
    check_error(api, api->PJRT_Client_Compile(&args));
    const auto finished = std::chrono::steady_clock::now();
    executable_ = args.executable;
    return std::chrono::duration<double, std::milli>(finished - started)
        .count();
  };

  switch (options_.load_policy) {
    case LoadPolicy::BinaryOnly:
      deserialize();
      load_kind_ = LoadKind::Deserialized;
      load_detail_ =
          "deserialized " + binary_path + " (LoadPolicy::BinaryOnly)";
      break;

    case LoadPolicy::CompileOnly: {
      const double milliseconds = compile();
      load_kind_ = LoadKind::Compiled;
      load_detail_ = "compiled " + mlir_path + " in " +
                     format_ms(milliseconds) + " ms (LoadPolicy::CompileOnly)";
      break;
    }

    case LoadPolicy::Auto: {
      // Why the `.binpb` was passed over, in a form fit to be logged.  Empty
      // means it was not passed over.
      std::string skipped;
      if (!file_exists(binary_path)) {
        skipped = binary_path + " is not present";
      } else if (options_.isa_guard && !exported_isa.empty()) {
        const std::string host = internal::host_isa_level();
        if (!internal::isa_at_least(host, exported_isa)) {
          // Deserializing relinks machine code; it never checks whether this
          // CPU can execute it.  Loading it anyway would be a SIGILL from
          // inside the executable, with nothing in the backtrace pointing at
          // the artifact.
          skipped = binary_path + " was exported for " + exported_isa +
                    " and this host is " + host;
        }
      }

      bool deserialized = false;
      if (skipped.empty()) {
        try {
          deserialize();
          deserialized = true;
        } catch (const std::runtime_error& error) {
          // A `.binpb` is locked to the JAX and XLA build that wrote it as
          // well as to the machine, so a refusal here is routine after an
          // upgrade and is worth falling back from rather than dying on.
          skipped = binary_path + " could not be deserialized: " +
                    error.what();
        }
      }
      if (deserialized) {
        load_kind_ = LoadKind::Deserialized;
        load_detail_ = "deserialized " + binary_path;
        break;
      }

      double milliseconds = 0;
      try {
        milliseconds = compile();
      } catch (const std::runtime_error& error) {
        throw LoadError("could not load " + binary_path + " (" + skipped +
                        ") and could not compile " + mlir_path + " (" +
                        error.what() + ")");
      }
      load_kind_ = LoadKind::Compiled;
      load_detail_ = "compiled " + mlir_path + " in " +
                     format_ms(milliseconds) + " ms because " + skipped;
      break;
    }
  }

  fingerprint_ = read_fingerprint(api, executable_);
}

void Function::validate_metadata() {
  const PJRT_Api* api = runtime_.api();

  // Which file the mismatch should be blamed on.  Only one of the two was
  // loaded, and naming the other would send a reader to the wrong artifact.
  const std::string artifact =
      name_ + (load_kind_ == LoadKind::Deserialized ? ".binpb" : ".mlirbc");
  const std::string sidecar = name_ + ".json";

  PJRT_LoadedExecutable_GetExecutable_Args get{};
  get.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get.loaded_executable = executable_;
  check_error(api, api->PJRT_LoadedExecutable_GetExecutable(&get));

  try {
    PJRT_Executable_NumOutputs_Args count{};
    count.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    count.executable = get.executable;
    check_error(api, api->PJRT_Executable_NumOutputs(&count));
    if (count.num_outputs != outputs_.size()) {
      throw LoadError(sidecar + " declares " +
                      std::to_string(outputs_.size()) + " outputs but " +
                      artifact + " produces " +
                      std::to_string(count.num_outputs));
    }

    PJRT_Executable_OutputElementTypes_Args types{};
    types.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
    types.executable = get.executable;
    check_error(api, api->PJRT_Executable_OutputElementTypes(&types));
    if (types.num_output_types != outputs_.size()) {
      throw LoadError(artifact + " reports " +
                      std::to_string(types.num_output_types) +
                      " output element types for " +
                      std::to_string(outputs_.size()) + " outputs");
    }

    PJRT_Executable_OutputDimensions_Args dimensions{};
    dimensions.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
    dimensions.executable = get.executable;
    // `num_outputs` carries no `// out` marker in the header and plugins have
    // been seen to write it; setting it to the count already established makes
    // the call correct whichever way the plugin reads it.
    dimensions.num_outputs = outputs_.size();
    check_error(api, api->PJRT_Executable_OutputDimensions(&dimensions));
    if (dimensions.num_outputs != outputs_.size()) {
      throw LoadError(artifact + " reports dimensions for " +
                      std::to_string(dimensions.num_outputs) +
                      " outputs but " + std::to_string(outputs_.size()) +
                      " were expected");
    }
    if (!outputs_.empty() &&
        (dimensions.dims == nullptr || dimensions.dim_sizes == nullptr)) {
      throw LoadError(artifact + " reported no output dimensions");
    }

    // One flat list of dimensions for every output, so walking it means
    // keeping a cursor rather than indexing.
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      const std::optional<DType> produced = from_pjrt(types.output_types[i]);
      if (!produced) {
        throw LoadError("output " + std::to_string(i) +
                        " has element type " +
                        pjrt_type_name(types.output_types[i]) +
                        ", which pjrt_exec does not support (supported: " +
                        kSupportedTypes + ")");
      }

      const std::size_t rank = dimensions.dim_sizes[i];
      const std::vector<std::int64_t> produced_shape(
          dimensions.dims + cursor, dimensions.dims + cursor + rank);
      cursor += rank;

      const std::string declared =
          shape_string(outputs_[i].dtype, outputs_[i].shape);
      const std::string actual = shape_string(*produced, produced_shape);
      if (declared != actual) {
        throw LoadError(sidecar + " declares output " + std::to_string(i) +
                        " as " + declared + " but " + artifact +
                        " produces " + actual);
      }
    }
  } catch (...) {
    // The handle is ours whatever happens next; leaking it would keep the
    // whole compiled program alive behind a failed load.
    PJRT_Executable_Destroy_Args destroy{};
    destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    destroy.executable = get.executable;
    destroy_error(api, api->PJRT_Executable_Destroy(&destroy));
    throw;
  }

  PJRT_Executable_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  destroy.executable = get.executable;
  destroy_error(api, api->PJRT_Executable_Destroy(&destroy));
}

void Function::allocate_arenas() {
  input_arenas_.reserve(inputs_.size());
  for (const ArraySpec& spec : inputs_) {
    input_arenas_.push_back(alloc_arena(spec.nbytes));
  }
  output_arenas_.reserve(outputs_.size());
  for (const ArraySpec& spec : outputs_) {
    output_arenas_.push_back(alloc_arena(spec.nbytes));
  }
}

void Function::wrap_inputs() {
  const PJRT_Api* api = runtime_.api();
  input_buffers_.reserve(inputs_.size());
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    const ArraySpec& spec = inputs_[i];

    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = runtime_.client();
    args.data = input_arenas_[i];
    args.type = to_pjrt(spec.dtype);
    args.dims = spec.shape.data();
    args.num_dims = spec.shape.size();
    // Null strides mean a dense major-to-minor layout, which is C order for
    // every dtype and every rank, and is what the exporter wrote.
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    // The buffer aliases the arena for its whole lifetime, so a call transfers
    // nothing.  PJRT's contract asks that the host buffer not be mutated while
    // a transfer is outstanding; this bends it by writing between calls, when
    // nothing is in flight -- measured, deliberate, and the property the whole
    // design rests on.
    args.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableZeroCopy;
    args.device = runtime_.device();
    check_error(api, api->PJRT_Client_BufferFromHostBuffer(&args));

    // Zero copy still produces a "done with host buffer" event, which fires
    // when the buffer is destroyed.  Nothing waits on it, and holding it would
    // just leak an event per input.
    if (args.done_with_host_buffer != nullptr) {
      PJRT_Event_Destroy_Args destroy{};
      destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      destroy.event = args.done_with_host_buffer;
      destroy_error(api, api->PJRT_Event_Destroy(&destroy));
    }
    input_buffers_.push_back(args.buffer);
  }
}

void Function::check_values_before() {
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    scan_arena("input", i, inputs_[i], input_arenas_[i]);
  }
}

void Function::check_values_after() {
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    scan_arena("output", i, outputs_[i], output_arenas_[i]);
  }
}

// docs: begin call_path
void Function::call() {
  if (debug_ && in_call_) {
    throw_reentered(name_);
  }
  CallGuard guard(in_call_);

  if (check_values_) {
    check_values_before();
  }

  const PJRT_Api* api = runtime_.api();
  PJRT_Buffer** input_list = input_buffers_.data();
  PJRT_Buffer** output_list = output_buffers_.data();
  PJRT_Event* complete = nullptr;

  PJRT_LoadedExecutable_Execute_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.executable = executable_;
  args.options = &execute_options_;
  args.argument_lists = &input_list;
  args.num_devices = 1;
  args.num_args = input_buffers_.size();
  args.output_lists = &output_list;
  args.device_complete_events = &complete;
  check_error(api, api->PJRT_LoadedExecutable_Execute(&args));

  // With inline execution the event is already ready and the await is a load
  // and a branch; with a dispatch pool it is where the hand-off is paid for.
  if (complete != nullptr) {
    PJRT_Event_Await_Args await{};
    await.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await.event = complete;
    PJRT_Error* error = api->PJRT_Event_Await(&await);

    PJRT_Event_Destroy_Args destroy{};
    destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy.event = complete;
    destroy_error(api, api->PJRT_Event_Destroy(&destroy));
    check_error(api, error);
  }

  for (std::size_t i = 0; i < output_buffers_.size(); ++i) {
    // On CPU, device memory is ordinary memory, so the results are copied
    // straight out of it.  `PJRT_Buffer_ToHostBuffer` would do the same copy
    // and charge an event, an await and a second allocation for it.
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args pointer{};
    pointer.struct_size =
        PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
    pointer.buffer = output_buffers_[i];
    check_error(api, api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&pointer));
    std::memcpy(output_arenas_[i], pointer.device_memory_ptr,
                outputs_[i].nbytes);

    PJRT_Buffer_Destroy_Args destroy{};
    destroy.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroy.buffer = output_buffers_[i];
    destroy_error(api, api->PJRT_Buffer_Destroy(&destroy));
    output_buffers_[i] = nullptr;
  }

  if (check_values_) {
    check_values_after();
  }
}
// docs: end call_path

void Function::check_input_index(std::size_t i) const {
  if (i >= inputs_.size()) {
    throw std::out_of_range("input index " + std::to_string(i) +
                            " is out of range: function '" + name_ + "' has " +
                            std::to_string(inputs_.size()) + " inputs");
  }
}

void Function::check_output_index(std::size_t i) const {
  if (i >= outputs_.size()) {
    throw std::out_of_range("output index " + std::to_string(i) +
                            " is out of range: function '" + name_ + "' has " +
                            std::to_string(outputs_.size()) + " outputs");
  }
}

void Function::check_input_access(std::size_t i, DType accessed_as) const {
  check_input_index(i);
  const ArraySpec& spec = inputs_[i];
  if (spec.dtype != accessed_as) {
    throw std::invalid_argument(
        "input " + std::to_string(i) + " ('" + spec.name + "') has dtype " +
        dtype_name(spec.dtype) + " but was accessed as " +
        dtype_name(accessed_as));
  }
}

void Function::check_output_access(std::size_t i, DType accessed_as) const {
  check_output_index(i);
  const ArraySpec& spec = outputs_[i];
  if (spec.dtype != accessed_as) {
    throw std::invalid_argument(
        "output " + std::to_string(i) + " ('" + spec.name + "') has dtype " +
        dtype_name(spec.dtype) + " but was accessed as " +
        dtype_name(accessed_as));
  }
}

//////////////////////
// build-time facts //
//////////////////////

const char* default_plugin_path() { return PJRT_EXEC_DEFAULT_PLUGIN_PATH; }

const char* version() { return "0.2.0"; }

int vendored_pjrt_api_minor() { return PJRT_API_MINOR; }

}  // namespace pjrt
