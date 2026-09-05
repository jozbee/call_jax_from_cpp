# Latency

`pjrt::LatencyRecorder` is a fixed-capacity sample buffer that never allocates
and never touches a file while recording. Storage is reserved once by the
constructor, so `record()` is a store and an increment, and everything that
could take a lock, fault a page or enter the allocator — sorting, formatting,
writing CSV — is a separate member you call after the loop.

Samples are signed nanoseconds. Nanoseconds because quantizing to whole
microseconds discards exactly the small-scale jitter being hunted; signed
because the same recorder measures period jitter, scheduled time minus actual
time, which is negative whenever a cycle runs early. Timing uses
`steady_clock` rather than `high_resolution_clock`, which on some standard
libraries is an alias for the wall clock — an NTP step mid-run would show up as
a spectacular outlier that never happened.

A full recorder drops and counts rather than growing, because growing would
allocate in the middle of the run being measured. That is the mistake to watch
for: size the recorder for the whole campaign, and read `dropped` before you
read anything else, since a nonzero drop count means the tail you are looking
at is the tail of a truncated prefix. One recorder belongs to one loop on one
thread; it is not thread-safe.

## Recording a loop

```cpp
pjrt::LatencyRecorder rec(iterations);   // the only allocation of the run
for (std::size_t i = 0; i < iterations; ++i) {
  pjrt::ScopedLatency t(rec);
  fn.call();
}
rec.report(stdout, "steady state");
rec.write_csv_row("artifacts/runs.csv", "steady state", config);
```

## The summary

```{doxygenstruct} pjrt::LatencySummary
```

Every duration is in microseconds; the two ratios are unitless and are the
headline.

| Field | Meaning |
|---|---|
| `count`, `dropped` | samples summarized, and samples the recorder had no room for |
| `mean_us`, `stddev_us` | mean and *population* standard deviation, not the sample one |
| `min_us`, `max_us` | the fastest and the slowest single call |
| `p50_us`, `p90_us`, `p99_us`, `p999_us`, `p9999_us` | percentiles, linearly interpolated |
| `max_over_p50` | the worst call relative to a typical one |
| `p999_over_p50` | the same for the one-in-a-thousand call — the ratio this project optimizes |

Percentiles interpolate linearly between the two neighbouring ranks
(`rank = p * (n - 1)`), which is what NumPy's `percentile` does by default.
Nearest-rank would quantize p99.9 to whichever single sample happens to sit at
that index and disagree with the analysis scripts. Both ratios are 0 when p50
is 0 — an empty or sub-resolution run — rather than infinite.

Read `max_over_p50` and `p999_over_p50`, not `mean_us`. A mean that improves
while `max_over_p50` grows is a worse result for a control loop, and this
project does not report an average-latency objective.

## Histogram buckets

```{doxygenstruct} pjrt::HistogramBin
```

One log-spaced bucket, half-open over `[lo_ns, hi_ns)`, holding `count`
samples. Bucket 0 is the underflow catch-all and carries
`lo_ns == INT64_MIN`, so a negative sample is counted rather than lost; the
last carries `hi_ns == INT64_MAX` when the bucket budget ran out before the
largest sample. Counts therefore always sum to `LatencyRecorder::size()`,
which is what makes a histogram checkable against the run it came from.

## The recorder

```{doxygenclass} pjrt::LatencyRecorder
:members: LatencyRecorder, record, time, clear, size, capacity, dropped, data, summary, histogram, report, write_csv_row, write_samples
```

Three of these write files, and each answers a different question.

`report` prints the summary and an ASCII histogram, trimmed to the range
between the first and last non-empty bucket. It is what a run prints when it
ends.

`write_csv_row` **appends** one summary row, writing the header only when the
file did not already exist. Appending is what makes an interleaved A/B sweep
possible: each short run adds a row to the same file and the comparison happens
across rows, rather than across one long sequential run that drifts with CPU
temperature. The `config` argument is not optional in spirit — a row without it
is unattributable — and the writer quotes a field containing a comma, because
an `$XLA_FLAGS` string is entitled to contain one and an unquoted comma shifts
every later column.

`write_samples` writes every raw sample as `index,ns`. Summaries hide *when* an
outlier happened, and that is usually the whole diagnosis: a spike on call 3 is
a page the warm-up never touched, a spike on call 30,000 is something periodic,
and the two have the same p99.9.

## Timing a scope

```{doxygenclass} pjrt::ScopedLatency
:members: ScopedLatency
```

Records on destruction, which is why it is preferred over a manual pair of
`now()` calls around a body with several exits: an early `return` or a thrown
exception still records, so a run does not silently lose exactly the calls that
went wrong.
