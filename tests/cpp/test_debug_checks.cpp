/**
 * @file test_debug_checks.cpp
 * @brief Make one specific mistake on purpose and report whether anything
 *        noticed.
 *
 * `FunctionOptions::debug` and `FunctionOptions::check_values` are the two
 * settings that turn a silent corruption into an exception, and the argument
 * for them is not that they catch things -- it is what happens when they are
 * off.  So each scenario here runs twice, once with `--debug` and once
 * without, and prints what came back either way.  The pair is the test:
 *
 *     THREW: input 0 ('A') element 0 is nan          (with --debug)
 *     NO-THROW nan_in_output=1                       (without)
 *
 * The second line is the one worth having.  Nothing failed, nothing was
 * logged, and the answer the control loop went on to use was nan.
 *
 *     test_debug_checks <base_path> <scenario> [--debug]
 *
 *   | scenario         | the mistake                       | artifact |
 *   |------------------|-----------------------------------|----------|
 *   | oob-input        | `input<double>(99)`               | any      |
 *   | oob-output       | `output<double>(99)`              | any      |
 *   | dtype-mismatch   | a float64 input read as `float`   | any      |
 *   | nonfinite-input  | a nan written into an input arena | any      |
 *   | nonfinite-output | a singular system, solved         | basic    |
 *   | bool-value       | the byte 2 in a bool arena        | trajopt  |
 *   | reentrant        | `call()` from two threads at once | any      |
 *
 * "any" is not quite any: `dtype-mismatch` needs an input 0 that is not
 * already float32, `nonfinite-input` needs a floating-point input, and
 * `bool-value` needs a bool one, which is why the last two name an artifact
 * that has them.  An artifact that cannot carry its scenario is exit 1 saying
 * so, never a quiet pass.
 *
 * Output is one line: `THREW: <what()>` or `NO-THROW`.  The scenarios whose
 * damage lands in an output arena rather than in an exception append
 * ` nan_in_output=0|1`, which is how the caller sees the corruption that went
 * unreported; it is appended to the `THREW:` line as well, because "the check
 * fired *and* the outputs are clean" is the claim being made there.  `nan` in
 * that name is loose: the flag is set by any non-finite value, inf included,
 * in any floating-point output arena.
 *
 * `reentrant` appends ` attempts=<n>` instead.  Two threads colliding inside
 * one `call()` is not something this program can force, so it reports how many
 * times it tried and lets pytest decide what to make of it.
 *
 * Exit 0 whether the mistake was caught or not -- both are results.  Exit 1 is
 * for a command line this program cannot act on, an artifact that cannot carry
 * the scenario, and a failure that is not the one being provoked.
 */
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include "common/workload.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

constexpr const char* kUsage =
    "usage: test_debug_checks <base_path> <scenario> [--debug]\n"
    "       test_debug_checks --artifact <base_path> <scenario> [--debug]\n"
    "scenarios: oob-input oob-output dtype-mismatch nonfinite-input "
    "nonfinite-output bool-value reentrant\n";

/// Comfortably past the end of any signature this project exports, and the
/// same index `example_01_basic --debug` uses, so the two messages match.
constexpr std::size_t kWildIndex = 99;

/// Re-entrancy is a race, and losing it is normal: the intruder can arrive
/// between two of the holder's calls instead of during one.  32 tries is
/// enough for the guard to fire many times over on any artifact this project
/// exports, and small enough that a build where it never fires still ends.
constexpr int kMaxAttempts = 32;

/// Wall-clock ceiling on the whole race, so a wedged holder thread ends the
/// test instead of the test run.
constexpr std::chrono::seconds kRaceDeadline{5};

/**
 * @brief A failure that is not the one being provoked.
 *
 * Kept apart from the scenario's own exception because the two mean opposite
 * things: one is the result being measured, the other says the measurement did
 * not happen.  This one reaches stderr and exit 1; the scenario's reaches
 * stdout as `THREW:` and exit 0.
 */
struct Unexpected : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Scenario {
  OobInput,
  OobOutput,
  DtypeMismatch,
  NonfiniteInput,
  NonfiniteOutput,
  BoolValue,
  Reentrant,
};

bool parse_scenario(const std::string& text, Scenario* scenario) {
  if (text == "oob-input") {
    *scenario = Scenario::OobInput;
  } else if (text == "oob-output") {
    *scenario = Scenario::OobOutput;
  } else if (text == "dtype-mismatch") {
    *scenario = Scenario::DtypeMismatch;
  } else if (text == "nonfinite-input") {
    *scenario = Scenario::NonfiniteInput;
  } else if (text == "nonfinite-output") {
    *scenario = Scenario::NonfiniteOutput;
  } else if (text == "bool-value") {
    *scenario = Scenario::BoolValue;
  } else if (text == "reentrant") {
    *scenario = Scenario::Reentrant;
  } else {
    return false;
  }
  return true;
}

/// Where the deliberately wrong pointers go.  Volatile because the accessors
/// are a load and a cast once `debug` is off, and a discarded result is a
/// discarded test.
volatile const void* g_sink = nullptr;

//////////////////////////
// artifact recognition //
//////////////////////////

/// Whether this is `01_basic`'s `fun(A, b) -> (x, r)`: a square float64 matrix
/// and a matching right-hand side.  The order is read from the artifact, so a
/// re-export at a different size still matches.
bool looks_like_basic(const pjrt::Function& f) {
  return f.num_inputs() == 2 && f.num_outputs() == 2 &&
         f.input_dtype(0) == pjrt::DType::Float64 && f.input_rank(0) == 2 &&
         f.input_shape(0)[0] == f.input_shape(0)[1] &&
         f.input_dtype(1) == pjrt::DType::Float64 && f.input_rank(1) == 1 &&
         f.input_shape(1)[0] == f.input_shape(0)[0] &&
         f.output_dtype(0) == pjrt::DType::Float64;
}

/**
 * @brief Write a linear system into the `basic` arenas.
 *
 * @param singular All-zero `A`.  `jnp.linalg.inv` of it divides by a zero
 *                 pivot, so the solution is non-finite while every input stays
 *                 finite -- the only way to reach `check_values_after` without
 *                 tripping `check_values_before` first.
 */
void fill_basic(pjrt::Function& f, bool singular) {
  const std::size_t n = f.input_numel(1);
  double* a = f.input<double>(0);
  double* b = f.input<double>(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      // Diagonally dominant, so the system is well conditioned by
      // construction and a non-finite output means this code, not the matrix
      // that happened to be drawn.
      a[i * n + j] = singular ? 0.0 : (i == j ? 4.0 : 0.1);
    }
    b[i] = 1.0;
  }
}

/// The first floating-point input, which is where a nan is written.
std::size_t first_float_input(const pjrt::Function& f) {
  for (std::size_t i = 0; i < f.num_inputs(); ++i) {
    const pjrt::DType dtype = f.input_dtype(i);
    if (dtype == pjrt::DType::Float64 || dtype == pjrt::DType::Float32) {
      return i;
    }
  }
  throw Unexpected(
      "this artifact has no floating-point input to make "
      "non-finite");
}

/// The first bool input, which is where the byte 2 is written.
std::size_t first_bool_input(const pjrt::Function& f) {
  for (std::size_t i = 0; i < f.num_inputs(); ++i) {
    if (f.input_dtype(i) == pjrt::DType::Bool) {
      return i;
    }
  }
  throw Unexpected(
      "this artifact has no bool input; bool-value wants the trajopt "
      "artifact, whose input 5 is 'use_terminal'");
}

/**
 * @brief Put a plausible problem in the input arenas before the mistake is
 *        made.
 *
 * The loader leaves every arena zeroed and the warm-up calls run on those
 * zeros, so an unfilled artifact is callable.  It is not *usable* for these
 * scenarios: a solver handed all-zero plant parameters can produce a
 * non-finite result on its own, and that would be indistinguishable from the
 * corruption the scenario is about.  So the two artifacts this project ships
 * are recognized and filled properly, and anything else is left as loaded.
 */
void fill_inputs(pjrt::Function& f, bool singular) {
  if (looks_like_basic(f)) {
    fill_basic(f, singular);
    return;
  }
  try {
    const cjfc::workload::Dims dims = cjfc::workload::check_signature(f);
    cjfc::workload::init_inputs(f, dims);
    cjfc::workload::write_reference(
        f.input<double>(cjfc::workload::kInXRef), dims, 0);
  } catch (const std::runtime_error&) {
    // Neither of the two known signatures. The zeroed arenas the loader left
    // are a valid input to every function that got this far, since the warm-up
    // calls were made on exactly them.
  }
}

/**
 * @brief Call once on good inputs, so the output arenas hold a clean result
 *        before the mistake is made.
 *
 * Without this, `nan_in_output` would be reporting the warm-up rather than the
 * scenario: the warm-up calls run on zeroed arenas, and a zero matrix inverts
 * to nan, so `basic` sits there with non-finite outputs from the moment it
 * finishes loading.  A flag that reads 1 before anything has gone wrong says
 * nothing about what went wrong afterwards.
 */
void prime(pjrt::Function& f) {
  try {
    f.call();
  } catch (const std::exception& error) {
    throw Unexpected(
        std::string("a call on good inputs failed, so there is no clean "
                    "result for the scenario to corrupt: ") +
        error.what());
  }
}

/// Whether any floating-point output arena holds a nan or an inf.  Integer and
/// bool arenas have no invalid bit patterns and are skipped, exactly as the
/// library's own `check_values` audit skips them.
bool nonfinite_in_outputs(const pjrt::Function& f) {
  for (std::size_t i = 0; i < f.num_outputs(); ++i) {
    const void* arena = f.output_raw(i);
    const std::size_t numel = f.output_numel(i);
    if (f.output_dtype(i) == pjrt::DType::Float64) {
      const double* values = static_cast<const double*>(arena);
      for (std::size_t k = 0; k < numel; ++k) {
        if (!std::isfinite(values[k])) {
          return true;
        }
      }
    } else if (f.output_dtype(i) == pjrt::DType::Float32) {
      const float* values = static_cast<const float*>(arena);
      for (std::size_t k = 0; k < numel; ++k) {
        if (!std::isfinite(values[k])) {
          return true;
        }
      }
    }
  }
  return false;
}

//////////////////
// the mistakes //
//////////////////

/// Everything the scenario needs in place before the mistake is made.  A
/// problem here is an `Unexpected`: the artifact cannot carry this scenario,
/// which is a different thing from the scenario finding nothing.
void prepare(pjrt::Function& f, Scenario scenario) {
  switch (scenario) {
    case Scenario::OobInput:
    case Scenario::OobOutput:
      break;

    case Scenario::DtypeMismatch:
      if (f.num_inputs() == 0) {
        throw Unexpected("this artifact has no inputs to mis-type");
      }
      if (f.input_dtype(0) == pjrt::DType::Float32) {
        throw Unexpected(
            "input 0 of this artifact is already float32, so reading it as "
            "float is not a mistake");
      }
      break;

    case Scenario::NonfiniteInput:
      (void)first_float_input(f);
      fill_inputs(f, /*singular=*/false);
      prime(f);
      break;

    case Scenario::NonfiniteOutput:
      if (!looks_like_basic(f)) {
        throw Unexpected(
            "nonfinite-output needs the basic artifact: it works by solving a "
            "singular system, which is a property of that function rather "
            "than of any artifact");
      }
      // Solved once well-conditioned and only then made singular, so the
      // outputs this scenario reports on are the ones the singular solve
      // produced.
      fill_basic(f, /*singular=*/false);
      prime(f);
      fill_basic(f, /*singular=*/true);
      break;

    case Scenario::BoolValue:
      (void)first_bool_input(f);
      fill_inputs(f, /*singular=*/false);
      prime(f);
      break;

    case Scenario::Reentrant:
      fill_inputs(f, /*singular=*/false);
      break;
  }
}

/// Make the mistake.  Whatever this throws is the result being measured.
void provoke(pjrt::Function& f, Scenario scenario) {
  switch (scenario) {
    case Scenario::OobInput:
      g_sink = f.input<double>(kWildIndex);
      break;

    case Scenario::OobOutput:
      g_sink = f.output<double>(kWildIndex);
      break;

    case Scenario::DtypeMismatch:
      // Taken but never written through: the point is whether the accessor
      // objects, and a write would corrupt the arena on the run where it does
      // not.
      g_sink = f.input<float>(0);
      break;

    case Scenario::NonfiniteInput: {
      const std::size_t index = first_float_input(f);
      if (f.input_dtype(index) == pjrt::DType::Float64) {
        f.input<double>(index)[0] = std::nan("");
      } else {
        f.input<float>(index)[0] = std::nanf("");
      }
      f.call();
      break;
    }

    case Scenario::NonfiniteOutput:
      // The inputs are finite and the arithmetic is not: this one has to reach
      // the audit that runs *after* the call.
      f.call();
      break;

    case Scenario::BoolValue:
      // Through the raw arena, because `*input<bool>(i) = 2` is a conversion
      // to `true` and stores a perfectly legal 1.
      *static_cast<unsigned char*>(f.input_raw(first_bool_input(f))) = 2;
      f.call();
      break;

    case Scenario::Reentrant:
      // Run before the reporting block; see race_for_reentrancy.
      break;
  }
}

////////////////////
// the reentrancy //
////////////////////

/// What one race produced: how many times the intruder actually entered
/// `call()` while the holder was inside one, and what came back on the last
/// of them.
struct RaceResult {
  int attempts = 0;
  std::exception_ptr caught;
};

/// Spin, rather than sleep, for @p ns.  A sleep hands the core back and
/// returns whenever the scheduler gets round to it, which is precisely the
/// timing this needs to control.
void spin_for(std::int64_t ns) {
  if (ns <= 0) {
    return;
  }
  const auto until =
      std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
  while (std::chrono::steady_clock::now() < until) {
  }
}

/// One call's duration, taken as the fastest of three so that a scheduling
/// hiccup does not stretch the intruder's delay past the call it is aiming at.
std::int64_t measure_call(pjrt::Function& f) {
  std::int64_t best = 0;
  for (int i = 0; i < 3; ++i) {
    const auto started = std::chrono::steady_clock::now();
    f.call();
    const auto finished = std::chrono::steady_clock::now();
    const std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
            .count();
    if (i == 0 || ns < best) {
      best = ns;
    }
  }
  return best;
}

/**
 * @brief Call `call()` from a second thread while the first is inside one.
 *
 * A `Function` owns fixed arenas and one executable, and its header says
 * plainly that it belongs to one thread.  The re-entrancy guard is the only
 * thing standing between that rule and a caller who breaks it, and it is worth
 * knowing whether it actually fires.
 *
 * The intruder waits for the holder to enter a call, then deliberately spins
 * for half a call before entering itself.  Both halves of that matter.  Half a
 * call in is where re-entrancy is unambiguous rather than a coin flip on
 * thread start-up.  It also staggers the two calls by half a period, which
 * keeps their *output collection* -- the few microseconds where both threads
 * would be writing and destroying the same `output_buffers_` list -- from
 * coinciding.  With the guard on that never happens anyway, because the
 * intruder throws before it executes; with the guard off it is exactly the
 * corruption being demonstrated, and one staggered overlap demonstrates it
 * without turning the run into a lottery over a double free.
 *
 * That is also why the loop stops where it does.  With the guard on, "did it
 * fire" is observable, so a lost race is retried.  With it off there is
 * nothing to observe but silence, so the first entry made while the holder was
 * inside a call is the whole story and the race ends there.
 *
 * @throws Unexpected when the holder's own calls fail, which means the run
 *         measured the aftermath of a collision rather than the guard.
 */
RaceResult race_for_reentrancy(pjrt::Function& f, bool debug) {
  const std::int64_t call_ns = measure_call(f);

  std::atomic<bool> holder_inside{false};
  std::atomic<bool> stop{false};
  std::exception_ptr holder_error;

  std::thread holder([&] {
    try {
      while (!stop.load(std::memory_order_relaxed)) {
        holder_inside.store(true, std::memory_order_relaxed);
        f.call();
        holder_inside.store(false, std::memory_order_relaxed);
      }
    } catch (...) {
      holder_error = std::current_exception();
    }
    holder_inside.store(false, std::memory_order_relaxed);
    stop.store(true, std::memory_order_relaxed);
  });

  const auto deadline = std::chrono::steady_clock::now() + kRaceDeadline;
  const auto expired = [&deadline] {
    return std::chrono::steady_clock::now() >= deadline;
  };
  const auto inside = [&holder_inside] {
    return holder_inside.load(std::memory_order_relaxed);
  };
  const auto stopped = [&stop] { return stop.load(std::memory_order_relaxed); };

  RaceResult result;
  while (result.attempts < kMaxAttempts && !stopped() && !expired()) {
    while (!inside() && !stopped() && !expired()) {
      std::this_thread::yield();
    }
    if (stopped() || expired()) {
      break;
    }
    spin_for(call_ns / 2);
    if (!inside()) {
      continue;  // The holder finished early; entering now proves nothing.
    }

    ++result.attempts;
    try {
      f.call();
    } catch (...) {
      result.caught = std::current_exception();
      break;
    }
    if (!debug) {
      break;
    }
  }

  stop.store(true, std::memory_order_relaxed);
  holder.join();
  if (holder_error) {
    try {
      std::rethrow_exception(holder_error);
    } catch (const std::exception& error) {
      throw Unexpected(std::string("the holder thread's own call() failed: ") +
                       error.what());
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::string base;
  std::string scenario_name;
  bool debug = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--debug") {
      debug = true;
    } else if (arg == "--artifact" && i + 1 < argc) {
      base = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "test_debug_checks: unknown flag '%s'\n%s",
                   arg.c_str(), kUsage);
      return 1;
    } else if (base.empty()) {
      base = arg;
    } else if (scenario_name.empty()) {
      scenario_name = arg;
    } else {
      std::fprintf(stderr, "test_debug_checks: unexpected argument '%s'\n%s",
                   arg.c_str(), kUsage);
      return 1;
    }
  }
  if (base.empty() || scenario_name.empty()) {
    // Named "unexpected argument" rather than left as a bare usage block: a
    // caller written against a different command line should be able to tell
    // that from a load failure without reading this file.
    std::fprintf(stderr, "test_debug_checks: unexpected argument list\n%s",
                 kUsage);
    return 1;
  }
  Scenario scenario = Scenario::OobInput;
  if (!parse_scenario(scenario_name, &scenario)) {
    std::fprintf(stderr, "test_debug_checks: unknown scenario '%s'\n%s",
                 scenario_name.c_str(), kUsage);
    return 1;
  }

  try {
    pjrt::Runtime runtime;

    pjrt::FunctionOptions options;
    options.debug = debug;
    // The re-entrancy guard is a `debug` check and nothing else. Leaving the
    // value audit on would have the holder thread throwing about the outputs
    // of some call it made -- a different exception, from the other thread,
    // reported as though it were the guard.
    options.check_values = debug && scenario != Scenario::Reentrant;
    pjrt::Function f(runtime, base, options);

    prepare(f, scenario);

    // Outside the reporting block: what this returns is the scenario's result,
    // but a failure of the *race* is a failure of the measurement.
    RaceResult race;
    if (scenario == Scenario::Reentrant) {
      race = race_for_reentrancy(f, debug);
    }

    std::string outcome;
    try {
      if (scenario == Scenario::Reentrant) {
        if (race.caught) {
          std::rethrow_exception(race.caught);
        }
      } else {
        provoke(f, scenario);
      }
      outcome = "NO-THROW";
    } catch (const std::exception& error) {
      outcome = std::string("THREW: ") + error.what();
    }

    switch (scenario) {
      case Scenario::NonfiniteInput:
      case Scenario::NonfiniteOutput:
      case Scenario::BoolValue:
        outcome +=
            nonfinite_in_outputs(f) ? " nan_in_output=1" : " nan_in_output=0";
        break;
      case Scenario::Reentrant:
        outcome += " attempts=" + std::to_string(race.attempts);
        break;
      // Named rather than defaulted, so that adding a scenario is a warning
      // here instead of a line that silently loses its suffix.
      case Scenario::OobInput:
      case Scenario::OobOutput:
      case Scenario::DtypeMismatch:
        break;
    }

    std::printf("%s\n", outcome.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_debug_checks: %s\n", error.what());
    return 1;
  } catch (...) {
    // Reachable only through the rethrow of whatever the holder thread
    // stored. Without it that would leave main() and abort, and a signal is a
    // worse answer than a line and an exit code.
    std::fprintf(stderr, "test_debug_checks: unknown exception\n");
    return 1;
  }
}
