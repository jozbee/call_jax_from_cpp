/**
 * @file fixture.hpp
 * @brief Reference cases: what JAX computed, and what the C++ path has to
 *        reproduce.
 *
 * A C++ call path that runs is not the same as a C++ call path that is right,
 * and a benchmark that reports a fast wrong answer is worse than no benchmark.
 * These files are the ground truth: `jax2exec.reference.write_reference_cases`
 * runs the exported function under JAX and freezes, per case, every input
 * followed by every output as raw bytes.
 *
 * The layout is the dumbest thing that works -- no header, no padding, no
 * length prefixes, C order, native width, little-endian, a scalar occupying
 * exactly one element -- because the reader already knows every shape and
 * dtype from the manifest, and anything cleverer is one more thing that can
 * disagree between the two languages.
 *
 * The manifest is schema 2:
 *
 * @code{.json}
 *   { "schema": 2, "name": "trajopt",
 *     "inputs":  [{"name": "x0",    "dtype": "float64", "shape": [48]}],
 *     "outputs": [{"name": "u_opt", "dtype": "float64", "shape": [50, 6]}],
 *     "cases": ["trajopt_case0.bin"],
 *     "tolerance": {"float64": 1e-6, "float32": 1e-4} }
 * @endcode
 *
 * Two properties are worth stating outright, because both are places where a
 * check that looks like a check silently is not one:
 *
 *   - **A NaN is a mismatch, counted separately.**  Every comparison against
 *     NaN is false, so a running maximum of the relative error absorbs one
 *     without ever exceeding a tolerance.  The exporter refuses to freeze a
 *     non-finite reference for the same reason; this is the other half of that
 *     rule, on the side that reads the file.
 *   - **A stale fixture is an error, not a buffer overrun.**  Every arena is
 *     filled through `load_inputs`, which re-checks the dtype and the byte
 *     count against the loaded `Function` before it copies anything.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace bench {

/// @brief One input or output, exactly as the manifest describes it.
struct ArraySpec {
  /// The JAX argument or result name, matching the sidecar's.
  std::string name;

  /// NumPy dtype name -- "float64", "int32", "bool" -- kept as the manifest
  /// spelled it, so an error message can quote the file rather than a
  /// re-rendering of it.
  std::string dtype;

  /// The exact JAX shape, row-major.  Empty for a scalar.
  std::vector<std::size_t> shape;

  /// Product of `shape`; 1 for a scalar, never 0 unless a dimension is.
  std::size_t numel = 0;

  /// `numel * itemsize(dtype)`: the length of every `memcpy` and every
  /// comparison this fixture performs.
  std::size_t nbytes = 0;
};

/// @brief One frozen case: the inputs to feed, and the outputs to expect.
///
/// Bytes rather than values.  The reader is handed one arena per array and
/// copies into or compares against it, and the element type is a property of
/// the manifest rather than of the storage.
struct Case {
  std::vector<std::vector<std::byte>> inputs;
  std::vector<std::vector<std::byte>> outputs;
};

/**
 * @brief A manifest and its cases, loaded once and read from the call loop.
 *
 * Construction does all the I/O and all the parsing.  After that, `compare`
 * and `load_inputs` allocate nothing: `load_inputs` in particular runs inside
 * the timed region, because a control loop writes fresh inputs every step and
 * a benchmark that skips that copy is measuring something no caller does.
 */
class Fixture {
 public:
  /// The one manifest schema this reader understands.
  static constexpr int kSchema = 2;

  /**
   * @brief Read `<dir>/<name>_cases.json` and every `.bin` it names.
   *
   * @throws std::runtime_error when a file is missing or unreadable, when the
   *         schema is not @ref kSchema, when a dtype is one this project has
   *         no arena for, or when a case file's length disagrees with the
   *         manifest.
   */
  Fixture(const std::string& dir, const std::string& name)
      : manifest_path_(dir + "/" + name + "_cases.json") {
    const nlohmann::json meta = read_json(manifest_path_);

    // Read the schema before anything else: on a manifest from an older
    // exporter every other field below would fail with a message about a
    // missing key, which sends the reader looking in the wrong place.
    const int schema = meta.value("schema", 0);
    if (schema != kSchema) {
      throw std::runtime_error(
          manifest_path_ + " declares schema " + std::to_string(schema) +
          " (0 means no schema field at all, which is the old float64-only "
          "layout) but this benchmark reads schema " +
          std::to_string(kSchema) +
          " only; regenerate the fixture with examples/02_trajopt/export.py");
    }

    name_ = meta.value("name", name);
    inputs_ = read_specs(meta, "inputs", "input", input_types_);
    outputs_ = read_specs(meta, "outputs", "output", output_types_);
    read_tolerance(meta);

    const auto files = meta.at("cases").get<std::vector<std::string>>();
    cases_.reserve(files.size());
    for (const std::string& file : files) {
      cases_.push_back(read_case(dir + "/" + file));
    }
    if (cases_.empty()) {
      throw std::runtime_error("no cases listed in " + manifest_path_);
    }
  }

  /// @brief The function these cases belong to, from the manifest.
  const std::string& name() const { return name_; }

  /// @brief The manifest that was read, for error messages that need to name
  ///        the file the reader should go and look at.
  const std::string& manifest_path() const { return manifest_path_; }

  /// @brief How many reference cases were frozen.
  std::size_t num_cases() const { return cases_.size(); }

  /// @brief The inputs, in call order.
  const std::vector<ArraySpec>& inputs() const { return inputs_; }

  /// @brief The outputs, in call order.
  const std::vector<ArraySpec>& outputs() const { return outputs_; }

  /// @brief The result of checking one call's outputs against a frozen case.
  ///
  /// Three numbers rather than one because they mean different things: a
  /// relative error just over tolerance is a reassociation difference, an
  /// exact mismatch in an integer output is a logic error, and a NaN is
  /// neither -- it is the failure mode that a max-of-relative-errors check
  /// cannot see at all.
  struct Comparison {
    /// Largest relative error over every floating element of every output.
    double max_rel_err = 0.0;

    /// Elements of integer and bool outputs that differ bit for bit.
    std::size_t exact_mismatches = 0;

    /// Floating elements where either side is NaN.  Always a failure.
    std::size_t nan_count = 0;

    /// Whether this run agrees with the reference: no NaN, no exact mismatch,
    /// and every floating element within its own dtype's tolerance.
    bool ok = false;
  };

  /**
   * @brief Compare one call's outputs against case @p case_index.
   *
   * @param actual One pointer per output, in call order -- the arenas
   *               `Function::output_raw` hands back.
   *
   * Floating outputs are judged against the manifest's per-dtype tolerance
   * with an absolute floor in the denominator (1e-12 for float64, 1e-6 for
   * float32), so a reference value of exactly zero does not turn a
   * one-ULP difference into an infinite relative error.  The floor is a
   * denominator floor and not a free pass: garbage where the reference has a
   * zero still fails.  Integer and bool outputs are compared exactly, since
   * there is no such thing as a rounding difference in an index.
   *
   * @throws std::runtime_error when @p case_index is out of range or
   *         @p actual does not have one pointer per output.
   */
  Comparison compare(std::size_t case_index,
                     const std::vector<const void*>& actual) const {
    const Case& reference = case_at(case_index);
    if (actual.size() != outputs_.size()) {
      throw std::runtime_error("compare() was given " +
                               std::to_string(actual.size()) +
                               " output pointers but " + manifest_path_ +
                               " describes " + std::to_string(outputs_.size()));
    }

    Comparison result;
    result.ok = true;
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      const ArraySpec& spec = outputs_[i];
      const std::vector<std::byte>& want = reference.outputs[i];
      const void* got = actual[i];
      if (got == nullptr) {
        throw std::runtime_error("output " + std::to_string(i) + " ('" +
                                 spec.name + "') pointer is null");
      }
      switch (output_types_[i]) {
        case pjrt::DType::Float64:
          compare_floating<double>(want, got, spec.numel, tolerance_f64_,
                                   kFloorF64, result);
          break;
        case pjrt::DType::Float32:
          compare_floating<float>(want, got, spec.numel, tolerance_f32_,
                                  kFloorF32, result);
          break;
        default:
          compare_exact(want, got, spec, output_types_[i], result);
          break;
      }
    }
    return result;
  }

  /**
   * @brief Copy case @p case_index's inputs into @p function's arenas.
   *
   * Called from inside the timed region on purpose.  Every check it makes is
   * an integer comparison against values resolved at load: the dtype and byte
   * count of each array, re-checked every call because getting them wrong is a
   * write past the end of an input arena or a read past the end of an output
   * one, and a fixture regenerated for a different signature is exactly how
   * that happens.
   *
   * @throws std::runtime_error when the function and the manifest disagree, or
   *         when @p case_index is out of range.
   */
  void load_inputs(std::size_t case_index, pjrt::Function& function) const {
    const Case& reference = case_at(case_index);
    if (function.num_inputs() != inputs_.size()) {
      mismatched_arity(function);
    }
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      if (function.input_dtype(i) != input_types_[i] ||
          function.input_nbytes(i) != inputs_[i].nbytes) {
        mismatched_input(i, function);
      }
      std::memcpy(function.input_raw(i), reference.inputs[i].data(),
                  inputs_[i].nbytes);
    }

    // Outputs are checked here too, even though nothing is written to them,
    // because `compare()` reads `numel` elements straight out of the pointers
    // the caller hands it. Those come from the `Function`'s arenas, sized from
    // the artifact; if the manifest declares a larger array than the artifact
    // produces, the comparison reads past the end. The two are separate
    // command-line options (`--assets-dir` and `--artifacts-dir`), so one
    // wrong flag is enough to pair a manifest with a different export.
    if (function.num_outputs() != outputs_.size()) {
      mismatched_arity(function);
    }
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      if (function.output_dtype(i) != output_types_[i] ||
          function.output_nbytes(i) != outputs_[i].nbytes) {
        mismatched_output(i, function);
      }
    }
  }

 private:
  /// Denominator floor for float64, small enough that it only takes effect
  /// against a reference value that is itself essentially zero.
  static constexpr double kFloorF64 = 1e-12;
  /// The same for float32, scaled to that type's much coarser epsilon.
  static constexpr double kFloorF32 = 1e-6;

  /// Read a whole JSON file, naming the file when it is missing or malformed.
  /// nlohmann's own messages say what is wrong but not where, and "parse error
  /// at line 3" is unactionable without the path.
  static nlohmann::json read_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
      throw std::runtime_error(
          "cannot open " + path +
          "; export the fixture with examples/02_trajopt/export.py");
    }
    try {
      nlohmann::json meta;
      file >> meta;
      return meta;
    } catch (const nlohmann::json::exception& error) {
      throw std::runtime_error(path + " is not valid JSON: " + error.what());
    }
  }

  /// Turn one `inputs`/`outputs` array into specs, resolving each dtype to the
  /// enum the `Function` will be checked against.
  std::vector<ArraySpec> read_specs(const nlohmann::json& meta, const char* key,
                                    const char* kind,
                                    std::vector<pjrt::DType>& types) const {
    std::vector<ArraySpec> specs;
    const nlohmann::json& described = meta.at(key);
    specs.reserve(described.size());
    types.reserve(described.size());

    std::size_t index = 0;
    for (const nlohmann::json& entry : described) {
      ArraySpec spec;
      spec.name = entry.at("name").get<std::string>();
      spec.dtype = entry.at("dtype").get<std::string>();

      const std::optional<pjrt::DType> dtype = pjrt::parse_dtype(spec.dtype);
      if (!dtype) {
        throw std::runtime_error(
            manifest_path_ + " describes " + kind + " " +
            std::to_string(index) + " ('" + spec.name + "') as dtype '" +
            spec.dtype +
            "', which is not one of the eleven pjrt_exec "
            "supports (bool, int8..int64, uint8..uint64, float32, float64)");
      }

      spec.shape = entry.at("shape").get<std::vector<std::size_t>>();
      spec.numel = 1;
      for (const std::size_t dimension : spec.shape) {
        spec.numel *= dimension;
      }
      spec.nbytes = spec.numel * pjrt::itemsize(*dtype);

      types.push_back(*dtype);
      specs.push_back(std::move(spec));
      ++index;
    }
    return specs;
  }

  /// The manifest's tolerances, defaulting to `jax2exec.reference`'s own.
  /// float32 gets more room because XLA is free to fuse and reassociate, and
  /// the reference path and the executable do not have to do it the same way.
  void read_tolerance(const nlohmann::json& meta) {
    const auto tolerance = meta.find("tolerance");
    if (tolerance == meta.end() || !tolerance->is_object()) {
      return;
    }
    tolerance_f64_ = tolerance->value("float64", tolerance_f64_);
    tolerance_f32_ = tolerance->value("float32", tolerance_f32_);
  }

  /// Read one case file, whole, after checking its length against what the
  /// manifest says it should be.  Checking the total up front turns "short
  /// read" into an error that names both byte counts, which is the difference
  /// between "the fixture is stale" and a mystery.
  Case read_case(const std::string& path) const {
    const std::size_t input_bytes = total_bytes(inputs_);
    const std::size_t output_bytes = total_bytes(outputs_);
    const std::size_t expected = input_bytes + output_bytes;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
      throw std::runtime_error("cannot open " + path);
    }
    const std::streamoff size = file.tellg();
    if (size < 0) {
      throw std::runtime_error("cannot measure " + path);
    }
    if (static_cast<std::size_t>(size) != expected) {
      throw std::runtime_error(
          path + " holds " + std::to_string(static_cast<long long>(size)) +
          " bytes but " + manifest_path_ + " describes " +
          std::to_string(expected) + " (inputs " + std::to_string(input_bytes) +
          ", outputs " + std::to_string(output_bytes) +
          "); regenerate the fixture with examples/02_trajopt/export.py");
    }
    file.seekg(0, std::ios::beg);

    Case reference;
    read_group(file, path, inputs_, "input", reference.inputs);
    read_group(file, path, outputs_, "output", reference.outputs);
    return reference;
  }

  /// Sum of the arena sizes of one group, which is also its span in a case
  /// file.
  static std::size_t total_bytes(const std::vector<ArraySpec>& specs) {
    std::size_t total = 0;
    for (const ArraySpec& spec : specs) {
      total += spec.nbytes;
    }
    return total;
  }

  /// Read one group of arrays out of an open case file, in call order.
  static void read_group(std::ifstream& file, const std::string& path,
                         const std::vector<ArraySpec>& specs, const char* kind,
                         std::vector<std::vector<std::byte>>& into) {
    into.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      std::vector<std::byte> bytes(specs[i].nbytes);
      if (specs[i].nbytes != 0) {
        file.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(specs[i].nbytes));
        if (!file ||
            static_cast<std::size_t>(file.gcount()) != specs[i].nbytes) {
          throw std::runtime_error(
              "short read in " + path + ": " + kind + " " + std::to_string(i) +
              " ('" + specs[i].name + "') wants " +
              std::to_string(specs[i].nbytes) + " bytes, got " +
              std::to_string(static_cast<long long>(file.gcount())));
        }
      }
      into.push_back(std::move(bytes));
    }
  }

  /// Bounds-checked case lookup.  On the `load_inputs` path this is one
  /// comparison per call; the message is built only when it fails.
  const Case& case_at(std::size_t case_index) const {
    if (case_index >= cases_.size()) {
      out_of_range(case_index);
    }
    return cases_[case_index];
  }

  /**
   * Compare one floating output element by element.
   *
   * Equality is tested first, which is both the common answer and the only way
   * to give a pair of matching infinities the right verdict: their difference
   * is a NaN, and this is a comparison where a NaN must never pass.
   */
  template <class T>
  static void compare_floating(const std::vector<std::byte>& want_bytes,
                               const void* got_raw, std::size_t numel,
                               double tolerance, double floor,
                               Comparison& result) {
    const T* want = reinterpret_cast<const T*>(want_bytes.data());
    const T* got = static_cast<const T*>(got_raw);
    for (std::size_t j = 0; j < numel; ++j) {
      const double reference = static_cast<double>(want[j]);
      const double actual = static_cast<double>(got[j]);
      if (reference == actual) {
        continue;
      }
      if (std::isnan(reference) || std::isnan(actual)) {
        ++result.nan_count;
        result.ok = false;
        continue;
      }
      const double error =
          std::abs(actual - reference) / std::max(std::abs(reference), floor);
      if (error > result.max_rel_err) {
        result.max_rel_err = error;
      }
      if (error > tolerance) {
        result.ok = false;
      }
    }
  }

  /// Compare one integer or bool output exactly.  The whole array goes through
  /// one `memcmp`; only when that fails is the per-element loop entered, to
  /// say how much of the output is wrong -- one bad element and a completely
  /// wrong array are different bugs.
  static void compare_exact(const std::vector<std::byte>& want, const void* got,
                            const ArraySpec& spec, pjrt::DType dtype,
                            Comparison& result) {
    if (spec.nbytes == 0 || std::memcmp(want.data(), got, spec.nbytes) == 0) {
      return;
    }
    const std::size_t item = pjrt::itemsize(dtype);
    const auto* got_bytes = static_cast<const std::byte*>(got);
    for (std::size_t j = 0; j < spec.numel; ++j) {
      if (std::memcmp(want.data() + j * item, got_bytes + j * item, item) !=
          0) {
        ++result.exact_mismatches;
      }
    }
    result.ok = false;
  }

  // Out of line and [[noreturn]] on purpose: these build strings, and
  // load_inputs runs inside the timed loop.  Keeping the message off the hot
  // path is the difference between two integer comparisons per input and a
  // stack frame per input.
  [[noreturn]] void out_of_range(std::size_t case_index) const {
    throw std::runtime_error("case " + std::to_string(case_index) + ": " +
                             manifest_path_ + " has " +
                             std::to_string(cases_.size()) + " cases");
  }

  [[noreturn]] void mismatched_arity(const pjrt::Function& function) const {
    throw std::runtime_error(
        manifest_path_ + " describes " + std::to_string(inputs_.size()) +
        " inputs but function '" + function.name() + "' takes " +
        std::to_string(function.num_inputs()) +
        "; the fixture and the artifact came from different exports");
  }

  [[noreturn]] void mismatched_input(std::size_t i,
                                     const pjrt::Function& function) const {
    const ArraySpec& spec = inputs_[i];
    throw std::runtime_error(
        manifest_path_ + " describes input " + std::to_string(i) + " ('" +
        spec.name + "') as " + spec.dtype + " of " +
        std::to_string(spec.nbytes) + " bytes but function '" +
        function.name() + "' declares " +
        pjrt::dtype_name(function.input_dtype(i)) + " of " +
        std::to_string(function.input_nbytes(i)) +
        " bytes; the fixture and the artifact came from different exports");
  }

  /// The same for an output, whose mismatch is a read past the end of an arena
  /// rather than a write past it, and so is quieter and worth naming exactly.
  [[noreturn]] void mismatched_output(std::size_t i,
                                      const pjrt::Function& function) const {
    const ArraySpec& spec = outputs_[i];
    throw std::runtime_error(
        manifest_path_ + " describes output " + std::to_string(i) + " ('" +
        spec.name + "') as " + spec.dtype + " of " +
        std::to_string(spec.nbytes) + " bytes but function '" +
        function.name() + "' declares " +
        pjrt::dtype_name(function.output_dtype(i)) + " of " +
        std::to_string(function.output_nbytes(i)) +
        " bytes; the fixture and the artifact came from different exports");
  }

  std::string manifest_path_;
  std::string name_;
  std::vector<ArraySpec> inputs_;
  std::vector<ArraySpec> outputs_;

  // The dtypes again, resolved once, because the per-call check in
  // load_inputs has to be an enum comparison rather than a string one.
  std::vector<pjrt::DType> input_types_;
  std::vector<pjrt::DType> output_types_;

  double tolerance_f64_ = 1e-6;
  double tolerance_f32_ = 1e-4;

  std::vector<Case> cases_;
};

}  // namespace bench
