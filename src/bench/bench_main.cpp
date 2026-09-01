/**
 * @file bench_main.cpp
 * @brief Latency benchmark and correctness check for the PJRT call path.
 *
 * This is the measurement spine of the low-jitter work: every change to the
 * runtime is judged by the tail statistics this binary reports.
 *
 * Two traps, both learned the hard way, are designed around here:
 *
 *   - XLA dispatch is asynchronous.  A warm-up that does not wait on *every*
 *     output leaves a backlog that lands on the first timed call and turns a
 *     1.2x max/p50 into a 27x one.  Every call here, warm-up included, waits
 *     on all outputs.
 *   - Sequential A/B comparisons drift with CPU temperature, so anything
 *     comparative is run in short interleaved bursts elsewhere; this binary
 *     measures exactly one configuration and records what it was.
 *
 * Usage:
 *   bench --fixture mpc_solver [--iterations N] [--warmup N] [--case i]
 *         [--csv out.csv] [--samples raw.csv] [--label name]
 */

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "src/bench/fixture.hpp"
#include "src/bench/guard.hpp"
#include "src/bench/stats.hpp"
#include "src/pjrt_exec/pjrt_exec.hpp"
#include "src/pjrt_exec/rt.hpp"
#include "src/pjrt_exec/runtime.hpp"

namespace {

struct Args {
  std::string fixture = "mpc_solver";
  std::string assets_dir = "tests/assets/mpc";
  std::string artifacts_dir = "artifacts";
  std::string csv;
  std::string samples;
  std::string label;
  std::string api = "rt";  // "rt" (new) or "legacy" (per-call buffers)
  std::size_t iterations = 2000;
  std::size_t warmup = 50;
  std::size_t case_index = 0;
  bool check = true;
  bool synchronous = true;
  bool rt_harden = false;
  int pin_cpu = -1;
  int worker_threads = 1;
  int cpu_device_count = 1;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };
    if (flag == "--fixture") {
      a.fixture = next();
    } else if (flag == "--assets-dir") {
      a.assets_dir = next();
    } else if (flag == "--artifacts-dir") {
      a.artifacts_dir = next();
    } else if (flag == "--csv") {
      a.csv = next();
    } else if (flag == "--samples") {
      a.samples = next();
    } else if (flag == "--label") {
      a.label = next();
    } else if (flag == "--iterations") {
      a.iterations = std::stoul(next());
    } else if (flag == "--warmup") {
      a.warmup = std::stoul(next());
    } else if (flag == "--case") {
      a.case_index = std::stoul(next());
    } else if (flag == "--api") {
      a.api = next();
      if (a.api != "rt" && a.api != "legacy") {
        throw std::runtime_error("--api must be rt or legacy");
      }
    } else if (flag == "--async") {
      a.synchronous = false;
    } else if (flag == "--threads") {
      a.worker_threads = std::stoi(next());
    } else if (flag == "--devices") {
      a.cpu_device_count = std::stoi(next());
    } else if (flag == "--rt") {
      a.rt_harden = true;
    } else if (flag == "--cpu") {
      a.pin_cpu = std::stoi(next());
    } else if (flag == "--no-check") {
      a.check = false;
    } else {
      throw std::runtime_error("unknown flag: " + flag);
    }
  }
  if (a.label.empty()) {
    a.label = a.fixture;
  }
  return a;
}

/// Record the knobs that actually move the numbers, so a CSV row is readable
/// months later without guessing what the environment was.  Call this *after*
/// the runtime is constructed: it is the runtime that sets `PJRT_NPROC`.
std::string describe_config(const Args& args) {
  auto env = [](const char* k) -> std::string {
    const char* v = std::getenv(k);
    return v != nullptr ? v : "unset";
  };
  return "api=" + args.api + ";sync=" +
         (args.synchronous ? "on" : "off") + ";devices=" +
         std::to_string(args.cpu_device_count) + ";nproc=" +
         env("PJRT_NPROC") + ";xla_flags=" + env("XLA_FLAGS");
}

/**
 * @brief Apply the optional real-time hardening and report what took effect.
 *
 * Called after the runtime exists, because XLA's worker threads only exist
 * once a client has been created and `corral_xla_threads` looks them up by
 * name.  Failures are reported rather than fatal: a container without
 * `CAP_SYS_NICE` should still produce numbers, just less stable ones.
 */
void apply_hardening(const Args& args, bool have_runtime) {
  auto report = [](const char* what, const pjrt::rt::Status& s) {
    std::cout << "  " << (s.ok ? "[ok]   " : "[skip] ") << what << ": "
              << s.detail << "\n";
  };

  std::cout << "real-time hardening:\n";
  report("harden_malloc", pjrt::rt::harden_malloc());
  report("lock_memory", pjrt::rt::lock_memory());
  if (args.pin_cpu >= 0) {
    report("pin_current_thread", pjrt::rt::pin_current_thread(args.pin_cpu));
    if (have_runtime) {
      // Keep XLA's pools off the core the loop is using. With inline
      // execution they should be idle anyway; this makes sure of it.
      report("corral_xla_threads",
             pjrt::rt::corral_xla_threads({args.pin_cpu + 1}));
    }
  }
  report("set_realtime_priority", pjrt::rt::set_realtime_priority());
}

/// Result of the measured run, so both API paths share one reporting path.
struct Timings {
  std::chrono::nanoseconds load{0};
  std::chrono::nanoseconds first_call{0};
  std::vector<std::int64_t> samples_ns;
};

/**
 * @brief Warm up, check, then time `one_call`.
 *
 * The first call is timed separately: it carries page faults and lazy
 * binding that a steady-state number must not include.  Correctness is
 * verified after warm-up and before timing, so a wrong answer is never
 * reported as a fast one.
 */
template <typename Call>
Timings run_measured(const Args& args, const bench::Fixture& fixture,
                     const std::vector<const double*>& output_ptrs,
                     const bench::AllocGuard& guard, Call&& one_call) {
  Timings t;

  const auto first_start = std::chrono::steady_clock::now();
  one_call();
  t.first_call = std::chrono::steady_clock::now() - first_start;

  for (std::size_t i = 1; i < args.warmup; ++i) {
    one_call();
  }

  if (args.check) {
    const double err = fixture.max_rel_error(args.case_index, output_ptrs);
    const bool integral = fixture.integral_output_ok(output_ptrs);
    std::cout << "max rel err " << err
              << (integral ? " (integral output ok)" : " (INTEGRAL FAIL)")
              << "\n";
    if (err > 1e-6 || !integral) {
      throw std::runtime_error("outputs disagree with the reference");
    }
  }

  t.samples_ns.reserve(args.iterations);
  guard.arm();
  for (std::size_t i = 0; i < args.iterations; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    one_call();
    const auto t1 = std::chrono::steady_clock::now();
    t.samples_ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  }
  guard.disarm();
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const bench::Fixture fixture(args.assets_dir, args.fixture);
    if (args.case_index >= fixture.cases().size()) {
      throw std::runtime_error("case index out of range");
    }
    const bench::Case& ref = fixture.cases()[args.case_index];
    const bench::AllocGuard guard;

    std::cout << "fixture:   " << args.fixture << " (case "
              << args.case_index << " of " << fixture.cases().size() << ")\n"
              << "api:       " << args.api << "\n"
              << "iterations " << args.iterations << ", warmup "
              << args.warmup << "\n";

    const std::string base = args.artifacts_dir + "/" + args.fixture;
    Timings timings;

    if (args.api == "rt") {
      pjrt::RuntimeOptions rt_options;
      rt_options.synchronous = args.synchronous;
      rt_options.worker_threads = args.worker_threads;
      rt_options.cpu_device_count = args.cpu_device_count;

      const auto load_start = std::chrono::steady_clock::now();
      pjrt::Runtime runtime(rt_options);
      pjrt::FunctionOptions fn_options;
      // Warm-up is driven by this harness so that the first call can be timed
      // on its own; the Function must not quietly do it too.
      fn_options.warmup_calls = 0;
      pjrt::Function function(runtime, base, fn_options);
      const auto load_duration = std::chrono::steady_clock::now() - load_start;

      std::cout << "sync:      " << (args.synchronous ? "requested" : "off")
                << (runtime.synchronous_supported()
                        ? " (plugin supports it)"
                        : " (plugin ignores it: async dispatch)")
                << "\n";

      if (args.rt_harden) {
        apply_hardening(args, /*have_runtime=*/true);
      }

      std::vector<const double*> output_ptrs(function.num_outputs());
      for (std::size_t i = 0; i < function.num_outputs(); ++i) {
        output_ptrs[i] = function.output(i);
      }

      // A control loop writes fresh inputs every step, so the copy into the
      // arenas is part of the measured call, not setup.
      timings = run_measured(args, fixture, output_ptrs, guard, [&]() {
        for (std::size_t i = 0; i < function.num_inputs(); ++i) {
          const std::size_t n = function.input_size(i) == 0
                                    ? 1
                                    : function.input_size(i);
          std::memcpy(function.input(i), ref.inputs[i].data(),
                      n * sizeof(double));
        }
        function.call();
      });
      timings.load = load_duration;
    } else {
      const auto load_start = std::chrono::steady_clock::now();
      auto client = std::make_shared<pjrt::Client>();
      auto device = client->get_devices()[0];
      pjrt::AOTComputation comp(base, client);
      const auto load_end = std::chrono::steady_clock::now();

      std::vector<std::vector<double>> outputs;
      outputs.reserve(fixture.output_sizes().size());
      for (std::size_t n : fixture.output_sizes()) {
        outputs.emplace_back(n == 0 ? 1 : n, 0.0);
      }
      std::vector<const double*> output_ptrs(outputs.size());
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        output_ptrs[i] = outputs[i].data();
      }

      // One full call the old way: fresh device buffers per input, execute,
      // then copy every output back to the host.
      timings = run_measured(args, fixture, output_ptrs, guard, [&]() {
        std::vector<std::shared_ptr<pjrt::Buffer>> inputs;
        inputs.reserve(ref.inputs.size());
        for (std::size_t i = 0; i < ref.inputs.size(); ++i) {
          inputs.push_back(pjrt::Buffer::to_device_blocking(
              ref.inputs[i].data(), fixture.input_sizes()[i], client, device));
        }
        auto out_buffers = comp.execute_blocking(inputs);
        for (std::size_t i = 0; i < out_buffers.size(); ++i) {
          out_buffers[i]->to_host_blocking(outputs[i].data(),
                                           fixture.output_sizes()[i]);
        }
      });
      timings.load = load_end - load_start;
    }

    const auto to_us = [](auto d) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() *
             1e-3;
    };
    std::printf("\n=== cold start (microseconds) ===\n");
    std::printf("  load       %10.1f\n", to_us(timings.load));
    std::printf("  first call %10.1f\n", to_us(timings.first_call));

    const std::vector<std::int64_t>& samples_ns = timings.samples_ns;
    const std::string config = describe_config(args);
    std::cout << "config:    " << config << "\n";

    const bench::Summary summary = bench::summarize(samples_ns);
    bench::print_summary(args.label, summary);
    guard.report(args.iterations);

    if (!args.csv.empty()) {
      bench::write_csv(args.csv, args.label, summary, config);
    }
    if (!args.samples.empty()) {
      bench::write_samples(args.samples, samples_ns);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
