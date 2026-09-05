/**
 * @file test_latency.cpp
 * @brief A unit test of `pjrt_exec/latency.hpp`, with no PJRT anywhere in it.
 *
 * The recorder is where every number this project publishes comes from, so it
 * is worth testing against arithmetic rather than against a run.  Nothing here
 * loads a plugin, calls a function, or measures anything real: the samples are
 * written by hand and the percentiles are known in advance.  That is what
 * makes this the one test in the suite that cannot be flaky, and the first one
 * to run when a tail number looks wrong -- if this fails, the numbers were
 * never about the code under test.
 *
 * Each check prints its own name, and the first failure exits 1 naming itself.
 * Bare `assert` was the obvious spelling and does neither: it aborts with a
 * file and a line rather than a statement of what was being tested, and it
 * disappears entirely under `-DNDEBUG`, which is exactly the build a release
 * CI would run.
 *
 *     test_latency [--tmpdir <dir>]
 *
 * `--tmpdir` says where the CSV round trip may write; the default is the
 * system temporary directory.  The file is removed on success.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// For getpid(), so two copies of this test running at once cannot collide on
// the CSV file. The rest of the project is POSIX-only for far larger reasons.
#include <unistd.h>

#include "pjrt_exec/latency.hpp"

namespace {

/// Checks run so far, printed at the end so a truncated run is visible as a
/// short count rather than as a pass.
int g_checks = 0;

/// @brief Report one check by name; exit 1 on the first failure.
/// @param detail Printed only on failure, where the numbers are the diagnosis.
void check(bool ok, const std::string& what, const std::string& detail = {}) {
  ++g_checks;
  if (ok) {
    std::printf("ok   %s\n", what.c_str());
    return;
  }
  std::printf("FAIL %s\n", what.c_str());
  std::fprintf(stderr, "test_latency: %s failed", what.c_str());
  if (!detail.empty()) {
    std::fprintf(stderr, ": %s", detail.c_str());
  }
  std::fputc('\n', stderr);
  std::exit(1);
}

/// A number, at enough precision to diagnose a near miss.
std::string number(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.12g", value);
  return text;
}

/// @brief Check @p got against @p want to a relative tolerance.
/// @param tolerance Relative for values above 1, absolute below it, so a
///                  microsecond figure near zero is not held to an impossible
///                  standard.
void check_close(double got, double want, const std::string& what,
                 double tolerance = 1e-9) {
  const double allowed = tolerance * std::max(1.0, std::fabs(want));
  check(std::fabs(got - want) <= allowed, what,
        "got " + number(got) + ", want " + number(want));
}

/////////////////////////////
// percentiles and moments //
/////////////////////////////

/// A recorder holding 1..101 ns.  101 samples so that the median, p90 and p99
/// all land exactly on a sample and the interpolation can be checked where it
/// must return a sample untouched, as well as where it must not.
pjrt::LatencyRecorder ramp() {
  pjrt::LatencyRecorder recorder(101);
  for (std::int64_t ns = 1; ns <= 101; ++ns) {
    recorder.record(ns);
  }
  return recorder;
}

void test_empty() {
  const pjrt::LatencyRecorder recorder(8);
  const pjrt::LatencySummary s = recorder.summary();
  check(recorder.size() == 0, "empty: size is 0");
  check(s.count == 0 && s.dropped == 0, "empty: count and dropped are 0");
  check(s.p50_us == 0.0 && s.max_us == 0.0, "empty: percentiles are 0");
  // Not NaN and not inf: an empty run divides by nothing.
  check(s.max_over_p50 == 0.0 && s.p999_over_p50 == 0.0,
        "empty: tail ratios are 0, not NaN");
}

void test_ramp() {
  const pjrt::LatencyRecorder recorder = ramp();
  const pjrt::LatencySummary s = recorder.summary();

  check(s.count == 101 && s.dropped == 0, "ramp: 101 samples, none dropped");
  check(s.min_us == 1.0 * 1e-3, "ramp: min is the first sample");
  check(s.max_us == 101.0 * 1e-3, "ramp: max is the last sample");

  // rank = 0.50 * (101 - 1) = 50, an integer, so the interpolation weight on
  // the neighbour is exactly zero and p50 is the middle sample itself. Bit
  // equality is the claim; the sample is spelled the way the recorder spells
  // it, ns * 1e-3, because that is what it has to equal.
  check(s.p50_us == 51.0 * 1e-3, "ramp: p50 is exactly the middle sample",
        "got " + number(s.p50_us));
  check(s.p90_us == 91.0 * 1e-3, "ramp: p90 is exactly sample 91");
  check(s.p99_us == 100.0 * 1e-3, "ramp: p99 is exactly sample 100");

  // rank = 0.999 * 100 = 99.9 falls between the 100th and 101st samples, so
  // p99.9 is a value that is in no sample: nearest-rank would return 100 or
  // 101 ns and disagree with NumPy, which the analysis scripts use.
  check_close(s.p999_us, 0.1009, "ramp: p99.9 interpolates between samples");
  check(s.p999_us > 100.0 * 1e-3 && s.p999_us < 101.0 * 1e-3,
        "ramp: p99.9 lies strictly between its two neighbours");
  check_close(s.p9999_us, 0.10099, "ramp: p99.99 interpolates as well");

  check_close(s.mean_us, 51.0 * 1e-3, "ramp: mean is 51 ns");
  // Population standard deviation of 1..n is sqrt((n^2 - 1) / 12); for n = 101
  // that is sqrt(850) ns. The sample standard deviation would be 29.30 ns, so
  // this check also pins down which of the two the header promises.
  check_close(s.stddev_us, std::sqrt(850.0) * 1e-3,
              "ramp: stddev is the population one, sqrt(850) ns");

  check_close(s.max_over_p50, s.max_us / s.p50_us, "ramp: max/p50");
  check_close(s.p999_over_p50, s.p999_us / s.p50_us, "ramp: p99.9/p50");
}

////////////////////////
// capacity behaviour //
////////////////////////

void test_capacity() {
  pjrt::LatencyRecorder recorder(4);
  const std::int64_t* const storage = recorder.data();
  for (std::int64_t ns = 0; ns < 10; ++ns) {
    recorder.record(ns);
  }

  check(recorder.size() == 4, "capacity: size stops at the capacity");
  check(recorder.dropped() == 6,
        "capacity: the other six are counted as "
        "dropped");
  check(recorder.capacity() == 4, "capacity: capacity is unchanged");
  // The whole point of dropping: growing would allocate in the middle of the
  // run being measured. A reallocation would move the storage.
  check(recorder.data() == storage,
        "capacity: recording past the end does not reallocate");
  check(recorder.data()[0] == 0 && recorder.data()[3] == 3,
        "capacity: the first four samples are the ones kept");

  const pjrt::LatencySummary s = recorder.summary();
  check(s.count == 4 && s.dropped == 6,
        "capacity: the summary reports both counts");

  recorder.clear();
  check(recorder.size() == 0 && recorder.dropped() == 0,
        "capacity: clear() forgets the samples and the drops");
  check(recorder.capacity() == 4 && recorder.data() == storage,
        "capacity: clear() keeps the storage");
}

////////////////////
// signed samples //
////////////////////

void test_signed_samples() {
  pjrt::LatencyRecorder recorder(4);
  // Period jitter is scheduled-minus-actual and is negative whenever a cycle
  // runs early, so half the samples of a healthy real-time loop look like
  // this.
  recorder.record(-500);
  recorder.record(-1);
  recorder.record(0);
  recorder.record(1000);

  const pjrt::LatencySummary s = recorder.summary();
  check(s.count == 4, "signed: all four samples kept");
  check(s.min_us == -0.5, "signed: min is negative, not clamped");
  check(s.max_us == 1.0, "signed: max is the one positive sample");
  check_close(s.mean_us, 499.0 / 4.0 * 1e-3, "signed: mean of a mixed run");
  // rank = 0.5 * 3 = 1.5, halfway between -1 ns and 0 ns.
  check_close(s.p50_us, -0.0005, "signed: p50 interpolates across zero");

  pjrt::HistogramBin bins[8];
  const std::size_t n = recorder.histogram(bins, 8, 2.0, 1000);
  check(n == 2, "signed: two buckets cover the run");
  check(bins[0].lo_ns == std::numeric_limits<std::int64_t>::min(),
        "signed: the first bucket runs from -inf");
  check(bins[0].count == 3,
        "signed: the underflow bucket counts the negatives and the zero");
  check(bins[1].count == 1, "signed: the positive sample is in the second");
}

///////////////
// histogram //
///////////////

/// 500, 1000, 1500, 2000 and 5000 ns: two in one bucket, one each in two more,
/// and a gap, so an off-by-one in the bucket search cannot pass.
pjrt::LatencyRecorder spread() {
  pjrt::LatencyRecorder recorder(8);
  recorder.record(500);
  recorder.record(1000);
  recorder.record(1500);
  recorder.record(2000);
  recorder.record(5000);
  return recorder;
}

/// The counts must always sum to the number of samples: every bucket edge
/// question is really the question of whether a sample went missing.
void check_total(const pjrt::HistogramBin* bins, std::size_t n,
                 std::size_t expected, const std::string& what) {
  std::size_t total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    total += bins[i].count;
  }
  check(total == expected, what,
        "got " + std::to_string(total) + ", want " + std::to_string(expected));
}

void test_histogram() {
  const pjrt::LatencyRecorder recorder = spread();
  pjrt::HistogramBin bins[8];

  const std::size_t n = recorder.histogram(bins, 8, 2.0, 1000);
  check(n == 4, "histogram: four buckets reach the largest sample");
  check(bins[0].lo_ns == std::numeric_limits<std::int64_t>::min() &&
            bins[0].hi_ns == 1000,
        "histogram: bucket 0 is [-inf, 1000)");
  check(bins[1].lo_ns == 1000 && bins[1].hi_ns == 2000,
        "histogram: bucket 1 is [1000, 2000)");
  check(bins[2].lo_ns == 2000 && bins[2].hi_ns == 4000,
        "histogram: bucket 2 is [2000, 4000)");
  check(bins[3].lo_ns == 4000 && bins[3].hi_ns == 8000,
        "histogram: bucket 3 is [4000, 8000)");
  // Half-open [lo, hi): 1000 belongs to bucket 1, not bucket 0, and 2000 to
  // bucket 2, not bucket 1.
  check(bins[0].count == 1 && bins[1].count == 2 && bins[2].count == 1 &&
            bins[3].count == 1,
        "histogram: samples land on the low side of each edge");
  check_total(bins, n, recorder.size(), "histogram: counts sum to size()");

  const std::size_t capped = recorder.histogram(bins, 2, 2.0, 1000);
  check(capped == 2, "histogram: a bucket budget of 2 returns 2");
  check(bins[1].hi_ns == std::numeric_limits<std::int64_t>::max(),
        "histogram: the last bucket becomes the overflow catch-all");
  check(bins[0].count == 1 && bins[1].count == 4,
        "histogram: everything above the budget lands in the overflow");
  check_total(bins, capped, recorder.size(),
              "histogram: counts still sum to size() when capped");

  const std::size_t fallback = recorder.histogram(bins, 8, 0.5, 1000);
  check(fallback == 4 && bins[2].hi_ns == 4000,
        "histogram: a factor <= 1 falls back to 2");

  const std::size_t clamped = recorder.histogram(bins, 8, 2.0, 0);
  check(clamped > 0 && bins[0].hi_ns == 1,
        "histogram: a first edge below 1 ns is clamped to 1");
  check_total(bins, clamped, recorder.size(),
              "histogram: counts sum to size() with a clamped first edge");

  const pjrt::LatencyRecorder nothing(4);
  check(nothing.histogram(bins, 8, 2.0, 1000) == 0,
        "histogram: no samples means no buckets");
  check(recorder.histogram(nullptr, 8, 2.0, 1000) == 0,
        "histogram: null storage writes nothing");
  check(recorder.histogram(bins, 0, 2.0, 1000) == 0,
        "histogram: a bucket budget of 0 writes nothing");
}

//////////////////
// ratio guards //
//////////////////

void test_ratio_guards() {
  pjrt::LatencyRecorder recorder(4);
  for (int i = 0; i < 4; ++i) {
    // A run whose calls are all faster than the clock's resolution: p50 is 0
    // and the ratios would be 0/0 and x/0 if they were taken at face value.
    recorder.record(0);
  }
  const pjrt::LatencySummary s = recorder.summary();
  check(s.p50_us == 0.0, "ratios: p50 of an all-zero run is 0");
  check(s.max_over_p50 == 0.0, "ratios: max/p50 is 0 rather than NaN");
  check(s.p999_over_p50 == 0.0, "ratios: p99.9/p50 is 0 rather than inf");
  check(!std::isnan(s.max_over_p50) && !std::isinf(s.max_over_p50),
        "ratios: the guard leaves a finite number behind");
}

//////////////
// CSV rows //
//////////////

/// Split one CSV line into fields, undoing RFC 4180 quoting.  Deliberately not
/// shared with the writer: a round trip through the writer's own escaping
/// routine would agree with itself no matter what either of them did.
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c != '"') {
        field += c;
      } else if (i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else {
        quoted = false;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.push_back(field);
      field.clear();
    } else {
      field += c;
    }
  }
  fields.push_back(field);
  return fields;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  return lines;
}

/// 1..10 us.  Microsecond-scale on purpose: the row is written with `%.3f`, and
/// a nanosecond-scale run would round to three zeros and round trip trivially.
pjrt::LatencyRecorder decade() {
  pjrt::LatencyRecorder recorder(10);
  for (std::int64_t us = 1; us <= 10; ++us) {
    recorder.record(us * 1000);
  }
  return recorder;
}

void test_csv(const std::filesystem::path& directory) {
  const pjrt::LatencyRecorder recorder = decade();
  const pjrt::LatencySummary s = recorder.summary();

  const std::filesystem::path path =
      directory / ("test_latency_" +
                   std::to_string(static_cast<long>(::getpid())) + ".csv");
  std::error_code ignored;
  std::filesystem::remove(path, ignored);

  check(recorder.write_csv_row(path.c_str(), "round trip", "one,two") == true,
        "csv: the first row is written");

  std::vector<std::string> lines = read_lines(path);
  check(lines.size() == 2, "csv: a new file gets a header and a row",
        "got " + std::to_string(lines.size()) + " lines");
  check(lines[0].rfind("label,config,n,dropped,", 0) == 0,
        "csv: the header names the columns in order");

  const std::vector<std::string> header = split_csv(lines[0]);
  const std::vector<std::string> row = split_csv(lines[1]);
  check(header.size() == 15 && row.size() == 15,
        "csv: fifteen columns in the header and in the row",
        "header " + std::to_string(header.size()) + ", row " +
            std::to_string(row.size()));
  check(row[0] == "round trip", "csv: the label round trips");
  // The config carries $XLA_FLAGS verbatim in a real run, and a flag list is
  // entitled to contain a comma. Unquoted, it would shift every later column
  // and silently misattribute the numbers.
  check(row[1] == "one,two",
        "csv: a comma in the config does not split the "
        "row");
  check(row[2] == "10" && row[3] == "0", "csv: count and dropped");
  // The tolerances are the printed precision: %.3f for the microsecond
  // columns, %.4f for the two ratios.
  check_close(std::stod(row[4]), s.mean_us, "csv: mean_us", 5e-4);
  check_close(std::stod(row[7]), s.p50_us, "csv: p50_us", 5e-4);
  check_close(std::stod(row[10]), s.p999_us, "csv: p999_us", 5e-4);
  check_close(std::stod(row[12]), s.max_us, "csv: max_us", 5e-4);
  check_close(std::stod(row[13]), s.max_over_p50, "csv: max_over_p50", 5e-5);
  check_close(std::stod(row[14]), s.p999_over_p50, "csv: p999_over_p50", 5e-5);

  // Appending is what makes an interleaved A/B sweep possible, so a second row
  // must not bring a second header with it.
  check(recorder.write_csv_row(path.c_str(), "say \"hi\"", "two") == true,
        "csv: a second row is appended");
  lines = read_lines(path);
  check(lines.size() == 3, "csv: three lines, not four",
        "got " + std::to_string(lines.size()) + " lines");
  check(lines[0].rfind("label,", 0) == 0 && lines[2].rfind("label,", 0) != 0,
        "csv: the header is written only when the file is new");
  check(split_csv(lines[2])[0] == "say \"hi\"",
        "csv: an embedded quote round trips doubled");

  check(recorder.write_csv_row(nullptr, "x", "y") == false,
        "csv: a null path is a false return, not a crash");
  check(recorder.write_csv_row("/proc/nonexistent/row.csv", "x", "y") == false,
        "csv: an unopenable path is a false return");

  std::filesystem::remove(path, ignored);
}

////////////////////
// timing helpers //
////////////////////

void test_timing_helpers() {
  pjrt::LatencyRecorder recorder(2);

  // No threshold on the duration anywhere here: this checks the plumbing, and
  // a test that asserted a call took less than some number of nanoseconds
  // would fail on a busy machine for no reason.
  const std::int64_t returned = recorder.time([] {});
  check(recorder.size() == 1, "time(): records exactly one sample");
  check(returned >= 0, "time(): a steady clock never runs backwards");
  check(recorder.data()[0] == returned,
        "time(): returns the sample it recorded");

  {
    pjrt::ScopedLatency timed(recorder);
  }
  check(recorder.size() == 2, "ScopedLatency: records on destruction");
  check(recorder.data()[1] >= 0, "ScopedLatency: the sample is not negative");
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--tmpdir" && i + 1 < argc) {
      tmpdir = argv[++i];
    } else {
      std::fprintf(stderr, "usage: test_latency [--tmpdir <dir>]\n");
      return 1;
    }
  }

  test_empty();
  test_ramp();
  test_capacity();
  test_signed_samples();
  test_histogram();
  test_ratio_guards();
  test_csv(tmpdir);
  test_timing_helpers();

  std::printf("passed=%d\n", g_checks);
  return 0;
}
