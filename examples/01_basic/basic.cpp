/**
 * @file basic.cpp
 * @brief Load the artifact `examples/01_basic/export.py` wrote, call it once,
 *        and check the answer against arithmetic done here.
 *
 * The whole path in one file and nothing else: create a `pjrt::Runtime`, load
 * a `pjrt::Function` from a base path, write the inputs, `call()`, read the
 * outputs.  No timing, no real-time hardening, no reference cases --
 * `02_trajopt` and `03_realtime` add those on top of exactly this.
 *
 * Everything it prints before the results is `key=value`, one fact per line.
 * The test suite reads this output, and so does anyone diagnosing a load: the
 * three lines `load_kind`, `sync_mode` and `synchronous_supported` answer
 * "did the `.binpb` load or did it fall back to compiling the `.mlirbc`" and
 * "is execution actually inline", which are the two questions every later
 * latency number depends on.
 *
 * The function is `fun(A, b) -> (x, r)` with `x` solving `A x = b`.  It is
 * checked twice: against a residual recomputed here from the arenas, and
 * against `r`, which XLA produced inside the same executable.  A C++ side that
 * misread the row-major layout of `A` would still print a plausible `x`, and
 * only the recomputed residual would notice.
 *
 * @see examples/01_basic/export.py for what the artifact contains and why
 *      `jnp.linalg.inv` is in it.
 */
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

/// Tolerance on `max_i |(A x - b)_i|` for a well-conditioned 4x4 system in
/// float64.  Generous by a wide margin: the answer is either correct to about
/// 1e-16 or wrong by an amount no tolerance would let through.
constexpr double kResidualTolerance = 1e-9;

/// `SyncMode` as the reports and the tests spell it.  Same vocabulary as
/// `examples/common/report.hpp`, which the measuring examples use; two
/// spellings of "inline" in one project is one too many.
const char* sync_mode_name(pjrt::SyncMode mode) {
  switch (mode) {
    case pjrt::SyncMode::Inline:
      return "inline";
    case pjrt::SyncMode::Accepted:
      return "accepted";
    case pjrt::SyncMode::Rejected:
      return "rejected";
    case pjrt::SyncMode::Async:
      return "async";
  }
  return "unknown";
}

/// `LoadKind` as the reports and the tests spell it.
const char* load_kind_name(pjrt::LoadKind kind) {
  return kind == pjrt::LoadKind::Deserialized ? "deserialized" : "compiled";
}

/// A shape as `[4,4]`, or `[]` for a scalar.  No spaces: this line is parsed.
std::string shape_text(const std::vector<std::int64_t>& shape) {
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
void print_spec(const char* role, std::size_t index,
                const pjrt::ArraySpec& spec) {
  std::printf("%s[%zu]: dtype=%s shape=%s numel=%zu nbytes=%zu\n", role, index,
              pjrt::dtype_name(spec.dtype), shape_text(spec.shape).c_str(),
              spec.numel, spec.nbytes);
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
double next_uniform(std::mt19937_64& rng) {
  // 53 bits is the whole mantissa; 0x1p-53 scales them into [0, 1).
  const double unit = static_cast<double>(rng() >> 11) * 0x1p-53;
  return 2.0 * unit - 1.0;
}

/**
 * @brief Whether the loaded artifact is the one this file was written against.
 *
 * The typed accessors check the dtype only under `FunctionOptions::debug`, so
 * in a release build an artifact re-exported as float32 would hand back a
 * `double*` into a four-byte-per-element arena and say nothing -- the writes
 * below would then run off the end of it.  Checking the signature once, at
 * startup, is what makes the hot accessor safe to leave unchecked.
 *
 * The order `n` is read out of the artifact rather than assumed, so re-
 * exporting with a larger matrix needs no change here.
 */
bool signature_matches(const pjrt::Function& f, std::size_t a, std::size_t b) {
  return f.num_outputs() == 2 && f.input_dtype(a) == pjrt::DType::Float64 &&
         f.input_rank(a) == 2 && f.input_shape(a)[0] == f.input_shape(a)[1] &&
         f.input_dtype(b) == pjrt::DType::Float64 && f.input_rank(b) == 1 &&
         f.input_shape(b)[0] == f.input_shape(a)[0] &&
         f.output_dtype(0) == pjrt::DType::Float64 &&
         f.output_numel(0) == f.input_numel(b) &&
         f.output_dtype(1) == pjrt::DType::Float64 && f.output_numel(1) == 1;
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
void demonstrate_debug_checks(pjrt::Function& f, std::size_t a_index) {
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

}  // namespace

int main(int argc, char** argv) {
  try {
    cjfc::Cli cli(argc, argv, {"artifact", "debug", "threads"},
                  "usage: example_01_basic [--artifact artifacts/basic] "
                  "[--debug] [--threads N]\n"
                  "\n"
                  "Load an exported JAX function that solves A x = b, call it "
                  "once, and\n"
                  "check the solution against a residual recomputed in C++.\n"
                  "\n"
                  "  --artifact P  base path of the artifact set, without an "
                  "extension\n"
                  "                (default: artifacts/basic)\n"
                  "  --debug       turn on the per-call bounds, dtype and "
                  "value checks,\n"
                  "                and demonstrate what each of them catches\n"
                  "  --threads N   XLA worker threads (default: 1)\n");
    if (cli.help()) {
      return 0;
    }

    const std::string base = cli.get("artifact", "artifacts/basic");
    const bool debug = cli.flag("debug");

    // docs: begin load
    pjrt::RuntimeOptions runtime_options;
    runtime_options.worker_threads =
        static_cast<int>(cli.get_long("threads", 1));
    pjrt::Runtime runtime(runtime_options);  // one per process

    pjrt::FunctionOptions function_options;
    function_options.debug = debug;
    function_options.check_values = debug;
    pjrt::Function f(runtime, base, function_options);  // load once
    // docs: end load

    std::printf("load_kind=%s\n", load_kind_name(f.load_kind()));
    std::printf("synchronous_supported=%d\n",
                runtime.synchronous_supported() ? 1 : 0);
    std::printf("sync_mode=%s\n", sync_mode_name(runtime.synchronous_mode()));
    std::printf("num_inputs=%zu num_outputs=%zu\n", f.num_inputs(),
                f.num_outputs());
    for (std::size_t i = 0; i < f.num_inputs(); ++i) {
      print_spec("input", i, f.input_spec(i));
    }
    for (std::size_t i = 0; i < f.num_outputs(); ++i) {
      print_spec("output", i, f.output_spec(i));
    }

    // Names resolve to indices once, here, because `find_input` is a linear
    // scan over strings.  A loop indexes; only startup looks names up.
    const std::optional<std::size_t> a_index = f.find_input("A");
    const std::optional<std::size_t> b_index = f.find_input("b");
    if (!a_index.has_value() || !b_index.has_value()) {
      std::fprintf(stderr,
                   "%s.json declares no inputs named 'A' and 'b'; re-export "
                   "with examples/01_basic/export.py\n",
                   base.c_str());
      return 1;
    }
    if (!signature_matches(f, *a_index, *b_index)) {
      std::fprintf(stderr,
                   "%s is not the artifact this example expects: it wants "
                   "float64 A[n,n] and b[n] in, and float64 x[n] and a scalar "
                   "residual out\n",
                   base.c_str());
      return 1;
    }
    // The order comes from the artifact; nothing below assumes it is 4.
    const std::size_t n = f.input_numel(*b_index);

    // docs: begin call
    double* A = f.input<double>(*a_index);  // the arena XLA reads, not a copy
    double* b = f.input<double>(*b_index);

    // A fixed seed, so two runs of this example print the same numbers and a
    // difference in the output is a difference in the computation.
    std::mt19937_64 rng(20240517);
    for (std::size_t i = 0; i < n * n; ++i) {
      A[i] = next_uniform(rng);  // row-major, the layout the sidecar declares
    }
    // Adding 4 to the diagonal makes A diagonally dominant, so it is well
    // conditioned by construction and the residual below is a statement about
    // this code rather than about the matrix that happened to be drawn.
    for (std::size_t i = 0; i < n; ++i) {
      A[i * n + i] += 4.0;
      b[i] = next_uniform(rng);
    }

    f.call();

    const double* x = f.output<double>(0);
    const double residual_from_jax = *f.output<double>(1);
    // docs: end call

    // The check that matters: recompute A x - b here, from the same arenas,
    // rather than trusting the residual the executable reported.  A layout or
    // stride mistake on this side produces a believable `x` and a residual
    // that is not small.
    double residual_inf_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double row = -b[i];
      for (std::size_t j = 0; j < n; ++j) {
        row += A[i * n + j] * x[j];
      }
      residual_inf_norm = std::fmax(residual_inf_norm, std::fabs(row));
    }

    std::printf("x=[");
    for (std::size_t i = 0; i < n; ++i) {
      std::printf("%s%.12g", i == 0 ? "" : ",", x[i]);
    }
    std::printf("]\n");
    // Two different norms on purpose: the inf norm is what this file can
    // compute cheaply, and `r` is the 2-norm JAX computed.  Both are near
    // zero on a correct run, and neither is derived from the other.
    std::printf("residual_inf_norm=%.6e\n", residual_inf_norm);
    std::printf("residual_from_jax=%.6e\n", residual_from_jax);

    // `!(<=)` rather than `>`, so a nan fails instead of passing.
    if (!(residual_inf_norm <= kResidualTolerance)) {
      std::fprintf(stderr,
                   "residual %.6e exceeds %.1e: the solution this artifact "
                   "produced does not solve the system that was written into "
                   "the arenas\n",
                   residual_inf_norm, kResidualTolerance);
      return 1;
    }

    if (debug) {
      demonstrate_debug_checks(f, *a_index);
    } else {
      std::printf("debug=0 (checks disabled; see --debug)\n");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "example_01_basic: %s\n", error.what());
    return 1;
  }
}
