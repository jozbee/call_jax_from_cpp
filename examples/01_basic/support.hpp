/**
 * @file support.hpp
 * @brief Everything `basic.cpp` needs that is not the call path: the flags, the
 *        `key=value` reporting, the signature checks and the debug
 *        demonstration.
 *
 * Split out so that the example itself is the six steps a caller actually
 * performs -- create a runtime, load a function, check the signature, write the
 * inputs, call, read the outputs -- and nothing else.  Everything here runs at
 * startup or at exit; nothing in it belongs in a loop.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/names.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace basic {

/// Tolerance on `max_i |(A x - b)_i|` for a well-conditioned 4x4 system in
/// float64.  Generous by a wide margin: the answer is either correct to about
/// 1e-16 or wrong by an amount no tolerance would let through.
constexpr double kResidualTolerance = 1e-9;

constexpr char kUsage[] =
    "usage: example_01_basic [--artifact artifacts/basic] [--debug] "
    "[--threads N]\n"
    "\n"
    "Load an exported JAX function that solves A x = b, call it once, and\n"
    "check the solution against a residual recomputed in C++.\n"
    "\n"
    "  --artifact P  base path of the artifact set, without an extension\n"
    "                (default: artifacts/basic)\n"
    "  --debug       turn on the per-call bounds, dtype and value checks,\n"
    "                and demonstrate what each of them catches\n"
    "  --threads N   XLA worker threads (default: 1)\n";

/// Everything the command line can say, resolved once at startup.
struct Options {
  std::string artifact = "artifacts/basic";
  bool debug = false;
  long threads = 1;
};

/// @brief Read the command line into `Options`.
inline Options parse_options(const cjfc::Cli& cli) {
  Options options;
  options.artifact = cli.get("artifact", options.artifact);
  options.debug = cli.flag("debug");
  options.threads = cli.get_long("threads", options.threads);
  return options;
}

/// A shape as `[4,4]`, or `[]` for a scalar.  No spaces: this line is parsed.
inline std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      text += ',';
    }
    text += std::to_string(shape[i]);
  }
  text += ']';
  return text;
}

/// One `input[i]:` or `output[i]:` line, in the order the sidecar declares.
inline void print_spec(const char* role, std::size_t index,
                       const pjrt::ArraySpec& spec) {
  std::printf("%s[%zu]: dtype=%s shape=%s numel=%zu nbytes=%zu\n", role, index,
              pjrt::dtype_name(spec.dtype), shape_text(spec.shape).c_str(),
              spec.numel, spec.nbytes);
}

/**
 * @brief Print what was loaded and what it takes, one fact per line.
 *
 * `load_kind`, `synchronous_supported` and `sync_mode` answer the two questions
 * every later latency number depends on: did the `.binpb` load or did the
 * `.mlirbc` fallback compile it, and is execution actually inline.  The test
 * suite parses these lines, so they are a machine interface rather than
 * decoration.
 */
inline void print_signature(const pjrt::Runtime& runtime,
                            const pjrt::Function& f) {
  std::printf("load_kind=%s\n", cjfc::load_kind_name(f.load_kind()));
  std::printf("synchronous_supported=%d\n",
              runtime.synchronous_supported() ? 1 : 0);
  std::printf("sync_mode=%s\n",
              cjfc::sync_mode_name(runtime.synchronous_mode()));
  std::printf("num_inputs=%zu num_outputs=%zu\n", f.num_inputs(),
              f.num_outputs());
  for (std::size_t i = 0; i < f.num_inputs(); ++i) {
    print_spec("input", i, f.input_spec(i));
  }
  for (std::size_t i = 0; i < f.num_outputs(); ++i) {
    print_spec("output", i, f.output_spec(i));
  }
}

/// @brief The index of the input called @p name.
/// @throws std::runtime_error naming the artifact, because the remedy is to
///         re-export it rather than to edit this file.
inline std::size_t require_input(const pjrt::Function& f, const char* name) {
  const std::optional<std::size_t> index = f.find_input(name);
  if (!index.has_value()) {
    throw std::runtime_error("'" + f.name() + "' declares no input named '" +
                             name + "'; re-export it with " +
                             "examples/01_basic/export.py");
  }
  return *index;
}

/**
 * @brief Refuse an artifact that is not the one this example was written
 *        against.
 *
 * The typed accessors check the dtype only under `FunctionOptions::debug`, so
 * in a release build an artifact re-exported as float32 would hand back a
 * `double*` into a four-byte-per-element arena and say nothing -- the writes in
 * the call region would then run off the end of it.  Checking the signature
 * once, at startup, is what makes the hot accessor safe to leave unchecked.
 *
 * The order `n` is read out of the artifact rather than assumed, so re-
 * exporting with a larger matrix needs no change here.
 */
inline void require_shape(const pjrt::Function& f, std::size_t a,
                          std::size_t b) {
  const bool ok =
      f.num_outputs() == 2 && f.input_dtype(a) == pjrt::DType::Float64 &&
      f.input_rank(a) == 2 && f.input_shape(a)[0] == f.input_shape(a)[1] &&
      f.input_dtype(b) == pjrt::DType::Float64 && f.input_rank(b) == 1 &&
      f.input_shape(b)[0] == f.input_shape(a)[0] &&
      f.output_dtype(0) == pjrt::DType::Float64 &&
      f.output_numel(0) == f.input_numel(b) &&
      f.output_dtype(1) == pjrt::DType::Float64 && f.output_numel(1) == 1;
  if (!ok) {
    throw std::runtime_error(
        "'" + f.name() +
        "' is not the artifact this example expects: it wants float64 A[n,n] "
        "and b[n] in, and float64 x[n] and a scalar residual out");
  }
}

/**
 * @brief A uniform double in [-1, 1) drawn from @p rng.
 *
 * `std::uniform_real_distribution` is not specified to produce the same
 * sequence in two standard libraries, and this example is supposed to print
 * the same numbers under clang and gcc so that a difference in the output is a
 * difference in the computation.  `std::mt19937_64` *is* specified exactly, so
 * the mapping from its bits to a double is written out here rather than
 * delegated.
 */
inline double next_uniform(std::mt19937_64& rng) {
  // 53 bits is the whole mantissa; 0x1p-53 scales them into [0, 1).
  const double unit = static_cast<double>(rng() >> 11) * 0x1p-53;
  return 2.0 * unit - 1.0;
}

/**
 * @brief `max_i |(A x - b)_i|`, recomputed here from the arenas.
 *
 * The check that matters, and the reason it is not simply read off the output:
 * a layout or stride mistake on the C++ side produces a believable `x` and a
 * residual that is not small, while the residual the executable reported would
 * still look fine.
 */
inline double residual_inf_norm(const double* A, const double* b,
                                const double* x, std::size_t n) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double row = -b[i];
    for (std::size_t j = 0; j < n; ++j) {
      row += A[i * n + j] * x[j];
    }
    worst = std::fmax(worst, std::fabs(row));
  }
  return worst;
}

/// @brief Print the solution and both residuals.
///
/// Two different norms on purpose: the inf norm is what C++ can compute
/// cheaply, and `r` is the 2-norm JAX computed.  Both are near zero on a
/// correct run, and neither is derived from the other.
inline void print_solution(const double* x, std::size_t n, double inf_norm,
                           double from_jax) {
  std::printf("x=[");
  for (std::size_t i = 0; i < n; ++i) {
    std::printf("%s%.12g", i == 0 ? "" : ",", x[i]);
  }
  std::printf("]\n");
  std::printf("residual_inf_norm=%.6e\n", inf_norm);
  std::printf("residual_from_jax=%.6e\n", from_jax);
}

/// @brief Report a residual the tolerance refuses, and the exit code to give.
inline int residual_failure(double inf_norm) {
  std::fprintf(stderr,
               "residual %.6e exceeds %.1e: the solution this artifact "
               "produced does not solve the system that was written into the "
               "arenas\n",
               inf_norm, kResidualTolerance);
  return 1;
}

/**
 * @brief Print what the debug checks catch, by making each mistake on purpose.
 *
 * Every one of these is a silent bug with `FunctionOptions::debug` off: an
 * out-of-range index reads past the end of a vector, a `float*` into a float64
 * arena corrupts half the matrix on the first write, and a nan in an input is
 * simply computed with.  Turned on, each is an exception naming the array by
 * index *and* by the name the exporter gave it.
 *
 * The cost of leaving them on is one branch on a member that is always in
 * cache; the cost of the last one is a walk over every arena before and after
 * every call, which is why `check_values` is separate and belongs in a test
 * rather than in a loop.
 */
inline void demonstrate_debug_checks(pjrt::Function& f, std::size_t a_index) {
  try {
    (void)f.input<double>(99);
  } catch (const std::out_of_range& error) {
    std::printf("debug_check[out_of_range]: %s\n", error.what());
  }

  try {
    (void)f.input<float>(a_index);
  } catch (const std::invalid_argument& error) {
    std::printf("debug_check[dtype_mismatch]: %s\n", error.what());
  }

  // `check_values` audits the arenas on the way in, so this throws before the
  // executable runs and the outputs from the good call above are untouched.
  f.input<double>(a_index)[0] = std::numeric_limits<double>::quiet_NaN();
  try {
    f.call();
  } catch (const std::domain_error& error) {
    std::printf("debug_check[non_finite]: %s\n", error.what());
  }
}

}  // namespace basic
