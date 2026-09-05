/**
 * @file trajopt_signature.hpp
 * @brief What the `02_trajopt` artifact looks like, and how to drive it in a
 *        loop.
 *
 * Two examples call the same exported function: `02_trajopt` measures it, and
 * `03_realtime` runs it on a period.  Both need the same three things -- the
 * argument order, a starting set of inputs, and the rule for feeding one call's
 * outputs into the next call's inputs -- so all three live here rather than
 * being written twice and drifting apart.
 *
 * The signature, in call order, is:
 *
 * | | input | dtype | shape |
 * |---|---|---|---|
 * | 0 | `x0`           | float64 | `[nx]`      |
 * | 1 | `x_ref`        | float64 | `[h, nx]`   |
 * | 2 | `params`       | float64 | `[np]`      |
 * | 3 | `u_warm`       | float64 | `[h, nu]`   |
 * | 4 | `weights`      | float32 | `[4]`       |
 * | 5 | `use_terminal` | bool    | scalar      |
 * | 6 | `step`         | int32   | scalar      |
 *
 * | | output | dtype | shape |
 * |---|---|---|---|
 * | 0 | `u_opt`           | float64 | `[h, nu]`     |
 * | 1 | `x_pred`          | float64 | `[h, nx]`     |
 * | 2 | `cost`            | float32 | scalar        |
 * | 3 | `grad_norm`       | float64 | scalar        |
 * | 4 | `cost_history`    | float64 | `[n_iters]`   |
 * | 5 | `iterations_used` | int32   | scalar        |
 * | 6 | `backtracks_used` | int32   | scalar        |
 * | 7 | `step_next`       | int32   | scalar        |
 *
 * `step_next` is exactly `step + 1`.  It carries no numerical meaning: it is
 * there so that a loop can check, in integers, that the call it just made is
 * the call it asked for.  A float comparison against a tolerance cannot
 * distinguish "the solver converged somewhere else" from "the executable
 * ignored an input", and an inputs-are-being-read check has to be exact.
 *
 * `check_signature` verifies dtypes and ranks but reads the sizes out of the
 * artifact, so retuning the kernel -- a longer horizon, more states -- does not
 * require touching the C++.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace cjfc {

/// Input indices, in JAX's argument order.
enum In {
  kInX0 = 0,        ///< float64 `[nx]`: the measured state.
  kInXRef,          ///< float64 `[h, nx]`: the reference trajectory.
  kInParams,        ///< float64 `[np]`: the plant parameters.
  kInUWarm,         ///< float64 `[h, nu]`: the warm-start control sequence.
  kInWeights,       ///< float32 `[4]`: the cost weights.
  kInUseTerminal,   ///< bool: whether the terminal cost is active.
  kInStep,          ///< int32: the loop counter handed to the kernel.
  kNumIn,           ///< Number of inputs.
};

/// Output indices, in JAX's result order.
enum Out {
  kOutUOpt = 0,        ///< float64 `[h, nu]`: the optimized controls.
  kOutXPred,           ///< float64 `[h, nx]`: the predicted state trajectory.
  kOutCost,            ///< float32: the final objective value.
  kOutGradNorm,        ///< float64: the gradient norm at the last iterate.
  kOutCostHistory,     ///< float64 `[n_iters]`: the objective per iteration.
  kOutIterationsUsed,  ///< int32: iterations actually run.
  kOutBacktracksUsed,  ///< int32: line-search backtracks taken.
  kOutStepNext,        ///< int32: `step + 1`, the exactness check.
  kNumOut,             ///< Number of outputs.
};

/// Plant parameters: mass, k_lin, k_cub, damp, grav, k_couple, u_gain,
/// dt_scale.  The values the kernel was exported around; a different set is a
/// different plant, not a different tuning.
inline constexpr double kNominalParams[8] = {1.0, 4.0, 1.0, 0.5,
                                             2.0, 0.5, 1.0, 1.0};

/// Cost weights: tracking, effort, terminal, smoothing.  float32 because the
/// kernel takes them that way -- the weights are a knob, not a quantity the
/// answer's accuracy depends on.
inline constexpr float kWeights[4] = {1.0f, 0.01f, 5.0f, 0.1f};

/// The sizes read out of the artifact, rather than hard-coded here.
struct Dims {
  std::size_t nx = 0;       ///< State dimension.
  std::size_t nu = 0;       ///< Control dimension.
  std::size_t h = 0;        ///< Horizon length, in rows.
  std::size_t np = 0;       ///< Plant parameters the kernel takes.
  std::size_t n_iters = 0;  ///< Length of `cost_history`.
};

namespace detail {

/// One slot of the expected signature: what it is called here, and the two
/// things that must match for the C++ types to be right.
struct SlotSpec {
  const char* name;
  pjrt::DType dtype;
  std::size_t rank;
};

inline constexpr SlotSpec kInputTable[] = {
    {"x0", pjrt::DType::Float64, 1},
    {"x_ref", pjrt::DType::Float64, 2},
    {"params", pjrt::DType::Float64, 1},
    {"u_warm", pjrt::DType::Float64, 2},
    {"weights", pjrt::DType::Float32, 1},
    {"use_terminal", pjrt::DType::Bool, 0},
    {"step", pjrt::DType::Int32, 0},
};

inline constexpr SlotSpec kOutputTable[] = {
    {"u_opt", pjrt::DType::Float64, 2},
    {"x_pred", pjrt::DType::Float64, 2},
    {"cost", pjrt::DType::Float32, 0},
    {"grad_norm", pjrt::DType::Float64, 0},
    {"cost_history", pjrt::DType::Float64, 1},
    {"iterations_used", pjrt::DType::Int32, 0},
    {"backtracks_used", pjrt::DType::Int32, 0},
    {"step_next", pjrt::DType::Int32, 0},
};

/// Every mismatch ends the same way, because every mismatch has the same fix.
[[noreturn]] inline void signature_mismatch(const std::string& what) {
  throw std::runtime_error(
      "trajopt artifact signature mismatch: " + what +
      "; re-export with examples/02_trajopt/export.py");
}

}  // namespace detail

/**
 * @brief Check the loaded function against the table above and read its sizes.
 *
 * Counts, dtypes and ranks are checked because the C++ that follows depends on
 * them: a float32 arena read through a `double*` is silent corruption, and a
 * rank-1 `x_ref` would be indexed as if it were rank 2.  Sizes are *derived*,
 * not checked, so a re-tuned kernel with a longer horizon still runs.
 *
 * @return The dimensions this artifact was exported with.
 * @throws std::runtime_error naming the first slot that disagrees.
 */
inline Dims check_signature(const pjrt::Function& function) {
  if (function.num_inputs() != static_cast<std::size_t>(kNumIn)) {
    detail::signature_mismatch(
        "expected " + std::to_string(static_cast<int>(kNumIn)) +
        " inputs, artifact has " + std::to_string(function.num_inputs()));
  }
  if (function.num_outputs() != static_cast<std::size_t>(kNumOut)) {
    detail::signature_mismatch(
        "expected " + std::to_string(static_cast<int>(kNumOut)) +
        " outputs, artifact has " + std::to_string(function.num_outputs()));
  }

  const auto check_slot = [](const char* kind, std::size_t index,
                             const detail::SlotSpec& want,
                             const pjrt::ArraySpec& got) {
    const std::string where = std::string(kind) + " " + std::to_string(index) +
                              " (" + want.name + ")";
    if (got.dtype != want.dtype) {
      detail::signature_mismatch(where + " has dtype " +
                                 pjrt::dtype_name(got.dtype) + ", expected " +
                                 pjrt::dtype_name(want.dtype));
    }
    if (got.shape.size() != want.rank) {
      detail::signature_mismatch(where + " has rank " +
                                 std::to_string(got.shape.size()) +
                                 ", expected rank " +
                                 std::to_string(want.rank));
    }
  };

  for (std::size_t i = 0; i < static_cast<std::size_t>(kNumIn); ++i) {
    check_slot("input", i, detail::kInputTable[i], function.input_spec(i));
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(kNumOut); ++i) {
    check_slot("output", i, detail::kOutputTable[i], function.output_spec(i));
  }

  const auto dim = [](const pjrt::ArraySpec& spec, std::size_t axis) {
    return static_cast<std::size_t>(spec.shape[axis]);
  };

  Dims dims;
  dims.nx = dim(function.input_spec(kInX0), 0);
  dims.h = dim(function.input_spec(kInXRef), 0);
  dims.np = dim(function.input_spec(kInParams), 0);
  dims.nu = dim(function.input_spec(kInUWarm), 1);
  dims.n_iters = dim(function.output_spec(kOutCostHistory), 0);

  const auto require = [](bool condition, const std::string& what) {
    if (!condition) {
      detail::signature_mismatch(what);
    }
  };

  require(dims.nx > 0 && dims.nu > 0 && dims.n_iters > 0,
          "a dimension is zero (nx=" + std::to_string(dims.nx) + ", nu=" +
              std::to_string(dims.nu) + ", n_iters=" +
              std::to_string(dims.n_iters) + ")");
  // feedback() shifts the control sequence up one row and reads x_pred's second
  // row, neither of which exists on a one-step horizon.
  require(dims.h >= 2,
          "horizon is " + std::to_string(dims.h) +
              ", the recirculating loop needs at least 2 rows");
  require(dims.np <= sizeof kNominalParams / sizeof kNominalParams[0],
          "params takes " + std::to_string(dims.np) +
              " values but only 8 nominal ones are known here");
  require(function.input_numel(kInWeights) ==
              sizeof kWeights / sizeof kWeights[0],
          "weights has " + std::to_string(function.input_numel(kInWeights)) +
              " elements, expected 4");

  require(dim(function.input_spec(kInXRef), 1) == dims.nx,
          "x_ref is [" + std::to_string(dims.h) + ", " +
              std::to_string(dim(function.input_spec(kInXRef), 1)) +
              "] but x0 has " + std::to_string(dims.nx) + " elements");
  require(dim(function.input_spec(kInUWarm), 0) == dims.h,
          "u_warm has " +
              std::to_string(dim(function.input_spec(kInUWarm), 0)) +
              " rows but x_ref has " + std::to_string(dims.h));
  require(dim(function.output_spec(kOutUOpt), 0) == dims.h &&
              dim(function.output_spec(kOutUOpt), 1) == dims.nu,
          "u_opt is not the same shape as u_warm");
  require(dim(function.output_spec(kOutXPred), 0) == dims.h &&
              dim(function.output_spec(kOutXPred), 1) == dims.nx,
          "x_pred is not [h, nx]");

  return dims;
}

/**
 * @brief Fill every input arena with a sane starting point.
 *
 * The arenas come from `posix_memalign` and are *not* zeroed, so this is not
 * decoration: an unwritten arena holds whatever the allocator last left there,
 * and a stray nan in `x_ref` propagates through the first solve into every
 * later one via the warm start.  `x_ref` is zeroed here as well, even though a
 * caller is expected to overwrite it with `write_reference` before the first
 * call.
 */
inline void init_inputs(pjrt::Function& function, const Dims& dims) {
  double* x0 = function.input<double>(kInX0);
  for (std::size_t i = 0; i < dims.nx; ++i) {
    x0[i] = 0.0;
  }

  double* x_ref = function.input<double>(kInXRef);
  for (std::size_t i = 0; i < dims.h * dims.nx; ++i) {
    x_ref[i] = 0.0;
  }

  double* params = function.input<double>(kInParams);
  for (std::size_t i = 0; i < dims.np; ++i) {
    params[i] = kNominalParams[i];
  }

  double* u_warm = function.input<double>(kInUWarm);
  for (std::size_t i = 0; i < dims.h * dims.nu; ++i) {
    u_warm[i] = 0.0;
  }

  float* weights = function.input<float>(kInWeights);
  for (std::size_t i = 0; i < sizeof kWeights / sizeof kWeights[0]; ++i) {
    weights[i] = kWeights[i];
  }

  *function.input<bool>(kInUseTerminal) = true;
  *function.input<std::int32_t>(kInStep) = 0;
}

/**
 * @brief Write the reference trajectory for cycle @p k into @p x_ref.
 *
 * A slow sine wave sweeping along the position coordinates, with the velocities
 * left at zero: enough for the solver to have something to track and for
 * successive cycles to differ, which is what makes the warm start do real work.
 * The horizon steps ahead of the current cycle, so `x_ref` at cycle `k` and
 * cycle `k + 1` overlap in all but one row -- the same thing a real reference
 * generator produces.
 *
 * Plain stores into the caller's arena: no allocation, no branch on `k`, and
 * safe to call from the loop.
 */
inline void write_reference(double* x_ref, const Dims& dims, std::int64_t k) {
  const std::size_t positions = dims.nx / 2;
  for (std::size_t t = 0; t < dims.h; ++t) {
    double* row = x_ref + t * dims.nx;
    const double phase =
        0.05 * static_cast<double>(k + static_cast<std::int64_t>(t));
    for (std::size_t j = 0; j < positions; ++j) {
      row[j] = 0.3 * std::sin(phase + 0.2 * static_cast<double>(j));
    }
    for (std::size_t j = positions; j < dims.nx; ++j) {
      row[j] = 0.0;
    }
  }
}

// docs: begin recirculate
/**
 * @brief Feed cycle @p k's outputs back into the inputs for cycle `k + 1`.
 *
 * This is the whole point of the persistent-arena design: the state estimate,
 * the warm start and the counters are copied from output arenas into input
 * arenas, in place, with no device transfer and no allocation.  The next
 * `call()` then reads exactly the memory that was just written.
 *
 *   - `x0` becomes the second row of `x_pred` -- where the plant is predicted
 *     to be one step from now, which is where the next solve starts.
 *   - `u_warm` becomes `u_opt` shifted up one row, with the last row repeated:
 *     the standard receding-horizon warm start.
 *   - `use_terminal` flips every 250 cycles, so the run exercises both branches
 *     of the kernel rather than one of them 100,000 times.
 *   - `step` becomes `step_next`.
 *
 * @return Whether `step_next` was exactly `k + 1`.  False means the executable
 *         did not read the `step` this loop wrote -- an input aliasing or
 *         staleness bug, and the one failure mode a tolerance-based check on
 *         the float outputs would not catch.
 */
inline bool feedback(pjrt::Function& function, const Dims& dims,
                     std::int64_t k) {
  const double* x_pred = function.output<double>(kOutXPred);
  const double* u_opt = function.output<double>(kOutUOpt);
  const std::int32_t step_next = *function.output<std::int32_t>(kOutStepNext);

  std::memcpy(function.input<double>(kInX0), x_pred + dims.nx,
              dims.nx * sizeof(double));

  double* u_warm = function.input<double>(kInUWarm);
  std::memcpy(u_warm, u_opt + dims.nu, (dims.h - 1) * dims.nu * sizeof(double));
  std::memcpy(u_warm + (dims.h - 1) * dims.nu,
              u_opt + (dims.h - 1) * dims.nu, dims.nu * sizeof(double));

  *function.input<bool>(kInUseTerminal) = (k / 250) % 2 == 0;
  *function.input<std::int32_t>(kInStep) = step_next;

  return step_next == static_cast<std::int32_t>(k + 1);
}
// docs: end recirculate

}  // namespace cjfc
