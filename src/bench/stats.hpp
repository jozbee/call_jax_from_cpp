/**
 * @file stats.hpp
 * @brief Latency statistics for the benchmark harness.
 *
 * The point of this project is the tail, not the mean, so the reported
 * summary leads with percentiles and the max/p50 ratio.  Samples are kept in
 * nanoseconds: the original example quantized to whole microseconds, which
 * discards exactly the small-scale jitter we are trying to observe.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bench {

/// Latency summary, all fields in microseconds except the ratios.
struct Summary {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double p999 = 0.0;
  double max = 0.0;
  double max_over_p50 = 0.0;
  double p999_over_p50 = 0.0;
  std::size_t count = 0;
};

/// Nearest-rank percentile of an already-sorted sample vector.
inline double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double rank = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(rank);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = rank - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/// Summarize samples given in nanoseconds; output is in microseconds.
inline Summary summarize(const std::vector<std::int64_t>& samples_ns) {
  Summary s;
  if (samples_ns.empty()) {
    return s;
  }
  std::vector<double> us;
  us.reserve(samples_ns.size());
  for (std::int64_t ns : samples_ns) {
    us.push_back(static_cast<double>(ns) * 1e-3);
  }
  std::sort(us.begin(), us.end());

  s.count = us.size();
  for (double v : us) {
    s.mean += v;
  }
  s.mean /= static_cast<double>(us.size());
  for (double v : us) {
    s.stddev += (v - s.mean) * (v - s.mean);
  }
  s.stddev = std::sqrt(s.stddev / static_cast<double>(us.size()));

  s.min = us.front();
  s.max = us.back();
  s.p50 = percentile(us, 0.50);
  s.p90 = percentile(us, 0.90);
  s.p99 = percentile(us, 0.99);
  s.p999 = percentile(us, 0.999);
  s.max_over_p50 = s.p50 > 0.0 ? s.max / s.p50 : 0.0;
  s.p999_over_p50 = s.p50 > 0.0 ? s.p999 / s.p50 : 0.0;
  return s;
}

inline void print_summary(const std::string& label, const Summary& s) {
  std::printf("\n=== %s (n=%zu, microseconds) ===\n", label.c_str(), s.count);
  std::printf("  mean   %10.1f     stddev %10.1f\n", s.mean, s.stddev);
  std::printf("  min    %10.1f     p50    %10.1f\n", s.min, s.p50);
  std::printf("  p90    %10.1f     p99    %10.1f\n", s.p90, s.p99);
  std::printf("  p99.9  %10.1f     max    %10.1f\n", s.p999, s.max);
  std::printf("  --- tail ratios ---\n");
  std::printf("  max/p50   %7.3f      p99.9/p50 %7.3f\n", s.max_over_p50,
              s.p999_over_p50);
}

/// Append one row to a CSV file, writing the header if the file is new.
inline void write_csv(const std::string& path, const std::string& label,
                      const Summary& s, const std::string& config) {
  const bool exists = [&] {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f != nullptr) {
      std::fclose(f);
      return true;
    }
    return false;
  }();
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (f == nullptr) {
    return;
  }
  if (!exists) {
    std::fprintf(f,
                 "label,config,n,mean_us,stddev_us,min_us,p50_us,p90_us,"
                 "p99_us,p999_us,max_us,max_over_p50,p999_over_p50\n");
  }
  std::fprintf(f, "%s,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f\n",
               label.c_str(), config.c_str(), s.count, s.mean, s.stddev, s.min,
               s.p50, s.p90, s.p99, s.p999, s.max, s.max_over_p50,
               s.p999_over_p50);
  std::fclose(f);
}

/// Dump every raw sample, for histogram/timeseries analysis.
inline void write_samples(const std::string& path,
                          const std::vector<std::int64_t>& samples_ns) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (f == nullptr) {
    return;
  }
  std::fprintf(f, "index,ns\n");
  for (std::size_t i = 0; i < samples_ns.size(); ++i) {
    std::fprintf(f, "%zu,%lld\n", i,
                 static_cast<long long>(samples_ns[i]));
  }
  std::fclose(f);
}

}  // namespace bench
