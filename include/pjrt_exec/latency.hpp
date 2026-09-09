/**
 * @file latency.hpp
 * @brief Fixed-capacity, allocation-free latency recorder.
 *
 * The figure of merit is the tail, so the summary leads with percentiles and
 * with the max/p50 and p99.9/p50 ratios.  Samples are signed nanoseconds:
 * whole microseconds would discard the jitter being hunted, and period jitter
 * is negative when a cycle runs early.  Storage is reserved once, `record()`
 * is a store and an increment, and a full recorder drops and counts rather
 * than growing mid-run.  Everything that sorts, formats or writes a file is a
 * separate member, called after the loop.
 *
 * @code
 *   pjrt::LatencyRecorder rec(iterations);
 *   for (std::size_t i = 0; i < iterations; ++i) {
 *     pjrt::ScopedLatency t(rec);
 *     fn.call();
 *   }
 *   rec.report(stdout, "steady state");
 * @endcode
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace pjrt {

// docs: begin latency_summary
/**
 * @brief Descriptive statistics for one run, in microseconds.
 *
 * The two ratios are unitless and are the headline.  Both are 0 when p50 is
 * 0 (an empty or sub-resolution run), never infinite.
 */
struct LatencySummary {
  std::size_t count = 0;    ///< Samples the summary was computed from.
  std::size_t dropped = 0;  ///< Samples dropped because the recorder was full.
  double mean_us = 0.0;
  double stddev_us = 0.0;  ///< Population standard deviation, not sample.
  double min_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p99_us = 0.0;
  double p999_us = 0.0;
  double p9999_us = 0.0;
  double max_us = 0.0;
  double max_over_p50 = 0.0;   ///< Worst call relative to a typical one.
  double p999_over_p50 = 0.0;  ///< The tail ratio this project optimizes.
};
// docs: end latency_summary

/**
 * @brief One bucket of a log-spaced latency histogram, half-open `[lo, hi)`.
 *
 * Bucket 0 always carries `lo_ns == INT64_MIN`, so a negative sample is
 * counted rather than lost; the last carries `hi_ns == INT64_MAX` when the
 * bucket budget ran out, so counts always sum to `LatencyRecorder::size()`.
 */
struct HistogramBin {
  std::int64_t lo_ns = 0;
  std::int64_t hi_ns = 0;
  std::size_t count = 0;
};

/**
 * @brief Records call latencies without allocating, then summarizes them.
 *
 * Not thread-safe: one recorder belongs to one loop on one thread.
 */
class LatencyRecorder {
 public:
  /**
   * @brief Reserve room for @p capacity samples.
   *
   * The only allocation the recorder makes.  The sorting scratch is reserved
   * here too, so `summary()` can run periodically on the real-time thread.
   */
  explicit LatencyRecorder(std::size_t capacity)
      : samples_(capacity), scratch_(capacity), capacity_(capacity) {}

  /// @brief Append one sample.  O(1), allocation-free; counts a drop when full.
  void record(std::int64_t ns) noexcept {
    if (size_ < capacity_) {
      samples_[size_++] = ns;
    } else {
      ++dropped_;
    }
  }

  /**
   * @brief Time @p f with `steady_clock`, record the elapsed nanoseconds, and
   *        return them.
   *
   * `steady_clock` and not `high_resolution_clock`, which can alias the wall
   * clock and turn an NTP step into an outlier that never happened.
   */
  template <class F>
  std::int64_t time(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    std::forward<F>(f)();
    const auto t1 = std::chrono::steady_clock::now();
    const std::int64_t ns = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    record(ns);
    return ns;
  }

  /// @brief Forget every sample; keeps the reserved storage.
  void clear() noexcept {
    size_ = 0;
    dropped_ = 0;
  }

  /// @brief Number of samples held.
  std::size_t size() const noexcept { return size_; }
  /// @brief Number of samples that fit before dropping starts.
  std::size_t capacity() const noexcept { return capacity_; }
  /// @brief Samples discarded because the recorder was full.
  std::size_t dropped() const noexcept { return dropped_; }
  /// @brief The raw samples, in recorded order; `size()` of them.
  const std::int64_t* data() const noexcept { return samples_.data(); }

  /**
   * @brief Sort a copy of the samples and describe them.
   *
   * Percentiles interpolate between neighbouring ranks (`rank = p * (n - 1)`)
   * as NumPy's `percentile` does, so the analysis scripts agree.  It sorts:
   * call it after the loop, not between two timed calls.
   */
  LatencySummary summary() const {
    LatencySummary s;
    s.count = size_;
    s.dropped = dropped_;
    if (size_ == 0) {
      return s;
    }

    double* const us = scratch_.data();
    for (std::size_t i = 0; i < size_; ++i) {
      us[i] = static_cast<double>(samples_[i]) * 1e-3;
    }
    std::sort(us, us + size_);

    const double n = static_cast<double>(size_);
    double sum = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
      sum += us[i];
    }
    s.mean_us = sum / n;
    double sq = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
      const double d = us[i] - s.mean_us;
      sq += d * d;
    }
    s.stddev_us = std::sqrt(sq / n);

    s.min_us = us[0];
    s.max_us = us[size_ - 1];
    s.p50_us = percentile(us, size_, 0.50);
    s.p90_us = percentile(us, size_, 0.90);
    s.p99_us = percentile(us, size_, 0.99);
    s.p999_us = percentile(us, size_, 0.999);
    s.p9999_us = percentile(us, size_, 0.9999);
    // A zero p50 means the clock could not resolve a typical call; a ratio
    // against it would be a division by zero dressed up as a result.
    s.max_over_p50 = s.p50_us > 0.0 ? s.max_us / s.p50_us : 0.0;
    s.p999_over_p50 = s.p50_us > 0.0 ? s.p999_us / s.p50_us : 0.0;
    return s;
  }

  /**
   * @brief Fill caller-provided storage with a log-spaced histogram.
   *
   * The caller owns the storage, so this allocates nothing.
   *
   * @param bins          Storage for at least @p max_bins buckets.
   * @param max_bins      Capacity of @p bins.
   * @param factor        Width ratio between adjacent buckets; values <= 1
   *                      fall back to 2.
   * @param first_bin_ns  Upper edge of the underflow bucket; clamped to >= 1.
   * @return Number of buckets written, which is 0 when there are no samples.
   */
  std::size_t histogram(HistogramBin* bins, std::size_t max_bins,
                        double factor = 2.0,
                        std::int64_t first_bin_ns = 1000) const {
    if (bins == nullptr || max_bins == 0 || size_ == 0) {
      return 0;
    }
    if (!(factor > 1.0)) {
      factor = 2.0;
    }
    if (first_bin_ns < 1) {
      first_bin_ns = 1;
    }

    std::int64_t max_ns = samples_[0];
    for (std::size_t i = 1; i < size_; ++i) {
      if (samples_[i] > max_ns) {
        max_ns = samples_[i];
      }
    }

    bins[0].lo_ns = std::numeric_limits<std::int64_t>::min();
    bins[0].hi_ns = first_bin_ns;
    bins[0].count = 0;
    std::size_t nbins = 1;
    double edge = static_cast<double>(first_bin_ns);
    while (nbins < max_bins && bins[nbins - 1].hi_ns <= max_ns) {
      const std::int64_t lo = bins[nbins - 1].hi_ns;
      edge *= factor;
      std::int64_t hi = edge >= 9.0e18
                            ? std::numeric_limits<std::int64_t>::max()
                            : static_cast<std::int64_t>(edge);
      if (hi <= lo) {
        hi = lo + 1;
      }
      bins[nbins].lo_ns = lo;
      bins[nbins].hi_ns = hi;
      bins[nbins].count = 0;
      ++nbins;
    }
    // Out of buckets before the largest sample: the top one becomes the
    // overflow catch-all so the counts still sum to size().
    if (nbins == max_bins) {
      bins[nbins - 1].hi_ns = std::numeric_limits<std::int64_t>::max();
    }

    for (std::size_t i = 0; i < size_; ++i) {
      const std::int64_t v = samples_[i];
      std::size_t b = 0;
      while (b + 1 < nbins && v >= bins[b].hi_ns) {
        ++b;
      }
      ++bins[b].count;
    }
    return nbins;
  }

  /**
   * @brief Print the summary and an ASCII histogram to @p out.
   *
   * Only buckets between the first and last non-empty one are printed.
   */
  void report(std::FILE* out, const char* label) const {
    if (out == nullptr) {
      return;
    }
    const LatencySummary s = summary();
    std::fprintf(out, "\n=== %s (n=%zu, microseconds) ===\n",
                 label != nullptr ? label : "latency", s.count);
    if (s.count == 0) {
      std::fprintf(out, "  no samples\n");
      return;
    }
    if (s.dropped != 0) {
      std::fprintf(out,
                   "  WARNING: %zu samples dropped (recorder capacity %zu)\n",
                   s.dropped, capacity_);
    }
    std::fprintf(out, "  mean   %10.1f     stddev %10.1f\n", s.mean_us,
                 s.stddev_us);
    std::fprintf(out, "  min    %10.1f     p50    %10.1f\n", s.min_us,
                 s.p50_us);
    std::fprintf(out, "  p90    %10.1f     p99    %10.1f\n", s.p90_us,
                 s.p99_us);
    std::fprintf(out, "  p99.9  %10.1f     p99.99 %10.1f\n", s.p999_us,
                 s.p9999_us);
    std::fprintf(out, "  max    %10.1f\n", s.max_us);
    std::fprintf(out, "  --- tail ratios ---\n");
    std::fprintf(out, "  max/p50   %7.3f      p99.9/p50 %7.3f\n",
                 s.max_over_p50, s.p999_over_p50);

    HistogramBin bins[40];
    const std::size_t nbins =
        histogram(bins, sizeof bins / sizeof bins[0], 2.0, 1000);
    if (nbins == 0) {
      return;
    }
    std::size_t first = 0;
    while (first + 1 < nbins && bins[first].count == 0) {
      ++first;
    }
    std::size_t last = nbins - 1;
    while (last > first && bins[last].count == 0) {
      --last;
    }
    std::size_t peak = 0;
    for (std::size_t i = first; i <= last; ++i) {
      peak = std::max(peak, bins[i].count);
    }
    static const char kBar[] = "########################################";
    const std::size_t bar_max = sizeof kBar - 1;
    std::fprintf(out, "  --- histogram (us) ---\n");
    for (std::size_t i = first; i <= last; ++i) {
      char lo[24];
      char hi[24];
      edge_label(bins[i].lo_ns, lo, sizeof lo);
      edge_label(bins[i].hi_ns, hi, sizeof hi);
      const std::size_t width =
          peak > 0 ? (bins[i].count * bar_max + peak - 1) / peak : 0;
      // The space belongs to the bar, so an empty bucket has no trailing blank.
      std::fprintf(out, "  [%9s,%9s) %10zu%s%.*s\n", lo, hi, bins[i].count,
                   width > 0 ? " " : "", static_cast<int>(width), kBar);
    }
  }

  /**
   * @brief Append one summary row to a CSV file, writing the header only when
   *        the file did not already exist.
   *
   * Appending is what makes an interleaved A/B sweep possible: each short run
   * adds a row, and the comparison happens across rows rather than across one
   * long run that drifts with CPU temperature.
   *
   * @param path CSV file to append to; created with a header if absent.
   * @param label Name of this run, the first column; interleaved rounds of one
   *              configuration share a label.
   * @param config What produced the row (API, thread count, fixture).
   * @return false when the file could not be opened.
   */
  bool write_csv_row(const char* path, const char* label,
                     const char* config) const {
    if (path == nullptr) {
      return false;
    }
    bool exists = false;
    if (std::FILE* probe = std::fopen(path, "r")) {
      exists = true;
      std::fclose(probe);
    }
    std::FILE* f = std::fopen(path, "a");
    if (f == nullptr) {
      return false;
    }
    if (!exists) {
      std::fprintf(f,
                   "label,config,n,dropped,mean_us,stddev_us,min_us,p50_us,"
                   "p90_us,p99_us,p999_us,p9999_us,max_us,max_over_p50,"
                   "p999_over_p50\n");
    }
    const LatencySummary s = summary();
    write_field(f, label);
    std::fputc(',', f);
    write_field(f, config);
    std::fprintf(f,
                 ",%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                 "%.4f,%.4f\n",
                 s.count, s.dropped, s.mean_us, s.stddev_us, s.min_us, s.p50_us,
                 s.p90_us, s.p99_us, s.p999_us, s.p9999_us, s.max_us,
                 s.max_over_p50, s.p999_over_p50);
    std::fclose(f);
    return true;
  }

  /**
   * @brief Write every raw sample as `index,ns`, overwriting @p path.
   *
   * Summaries hide *when* an outlier happened: a spike on call 3 and one on
   * call 30,000 share a p99.9 and have different causes.
   *
   * @return false when the file could not be opened.
   */
  bool write_samples(const char* path) const {
    if (path == nullptr) {
      return false;
    }
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
      return false;
    }
    std::fprintf(f, "index,ns\n");
    for (std::size_t i = 0; i < size_; ++i) {
      std::fprintf(f, "%zu,%lld\n", i, static_cast<long long>(samples_[i]));
    }
    std::fclose(f);
    return true;
  }

 private:
  /// Quote a field only when it would split the row: a config string carries
  /// `$XLA_FLAGS` verbatim, and an unquoted comma shifts every later column.
  static void write_field(std::FILE* f, const char* text) {
    if (text == nullptr) {
      return;
    }
    if (std::strpbrk(text, ",\"\r\n") == nullptr) {
      std::fputs(text, f);
      return;
    }
    std::fputc('"', f);
    for (const char* p = text; *p != '\0'; ++p) {
      if (*p == '"') {
        std::fputc('"', f);  // RFC 4180 escapes a quote by doubling it
      }
      std::fputc(*p, f);
    }
    std::fputc('"', f);
  }

  /// Linear interpolation between the ranks bracketing `q * (n - 1)`.
  static double percentile(const double* sorted, std::size_t n,
                           double q) noexcept {
    if (n == 0) {
      return 0.0;
    }
    const double rank = q * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(rank);
    const std::size_t hi = lo + 1 < n ? lo + 1 : n - 1;
    const double frac = rank - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
  }

  /// Format one bucket edge in microseconds, spelling the sentinels as
  /// infinity.
  static void edge_label(std::int64_t ns, char* buf, std::size_t len) noexcept {
    if (ns == std::numeric_limits<std::int64_t>::min()) {
      std::snprintf(buf, len, "-inf");
    } else if (ns == std::numeric_limits<std::int64_t>::max()) {
      std::snprintf(buf, len, "+inf");
    } else {
      std::snprintf(buf, len, "%.1f", static_cast<double>(ns) * 1e-3);
    }
  }

  std::vector<std::int64_t> samples_;
  /// Sorted copy used by summary(); mutable so that summary() stays const.
  mutable std::vector<double> scratch_;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
  std::size_t dropped_ = 0;
};

// docs: begin scoped_latency
/**
 * @brief Times the enclosing scope and records it on destruction.
 *
 * An early `return` or a thrown exception still records, so a run does not
 * silently lose exactly the calls that went wrong.
 */
class ScopedLatency {
 public:
  /// @brief Start timing; the sample lands in @p recorder at end of scope.
  explicit ScopedLatency(LatencyRecorder& recorder) noexcept
      : recorder_(recorder), start_(std::chrono::steady_clock::now()) {}

  ~ScopedLatency() {
    const auto end = std::chrono::steady_clock::now();
    recorder_.record(static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
            .count()));
  }

  ScopedLatency(const ScopedLatency&) = delete;
  ScopedLatency& operator=(const ScopedLatency&) = delete;

 private:
  LatencyRecorder& recorder_;
  std::chrono::steady_clock::time_point start_;
};
// docs: end scoped_latency

}  // namespace pjrt
