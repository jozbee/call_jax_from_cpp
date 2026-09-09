"""The two periodic loops: ``example_03_minimal`` and ``example_04_realtime``.

03 is the loop with nothing around it, so all that is asked of it is that it
runs and prints what it measured; the rest is about 04's report.  The
structural assertions run everywhere.  The statistical ones are marked
``rt`` and run only where the conftest's audit says the host is tuned and
idle, because on an untuned host the same artifact's p50 moves with nothing
but the caller's period (``docs/benchmarks.md``), and a test that fails on a
healthy machine is worse than no test.
"""

from __future__ import annotations

import math
import numbers
import re

import helpers
import pytest

#: Every top-level key a schema-1 report must carry.  Checked as a set: a
#: report that silently stops writing its host audit is still valid JSON, and
#: the missing provenance is what makes a latency number unusable later.
TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "example",
        "artifact",
        "config",
        "host",
        "hardening",
        "runtime",
        "compute_us",
        "cycle_us",
        "wake_latency_us",
        "period_jitter_us",
        "deadlines",
        "rusage",
        "allocations",
        "checks",
        "exit_code",
    }
)

#: The four distributions, each of which must be a complete summary.
SERIES = ("compute_us", "cycle_us", "wake_latency_us", "period_jitter_us")

#: The control period every run here uses: comfortably above what the
#: artifact takes, so a missed deadline means the schedule slipped.
PERIOD_US = 10000

#: Cycles for the census run.  Allocations are counted per cycle, which 300
#: establish as firmly as 2000.
CENSUS_ITERATIONS = 300

#: Gates for the ``rt`` mark only: what a quiet, pinned, SCHED_FIFO run on a
#: tuned host reaches.
MAX_P999_OVER_P50 = 1.3
MAX_MAX_OVER_P50 = 2.0
MAX_WAKE_P99_US = 100.0

#: Hardening details whose ``ok: false`` is not the host failing to provide
#: something.  ``cpu_dma_latency`` needs write access to the device and is
#: off unless asked for; ``corral_xla_threads`` found no pool threads, which
#: is a plugin that named none.  Matched on the detail, not waved through by
#: name, so a corral that genuinely fails still fails the gate.
NOT_A_FAILURE = {
    "cpu_dma_latency": ("needs root", "not requested", "not supported"),
    "corral_xla_threads": ("no XLA worker threads found",),
}


def realtime_argv(build, artifacts, out, iterations, warmup=50):
    return [
        build.bin("example_04_realtime"),
        "--artifact",
        artifacts.trajopt,
        "--iterations",
        str(iterations),
        "--period-us",
        str(PERIOD_US),
        "--warmup",
        str(warmup),
        "--json",
        out,
    ]


@pytest.fixture(scope="session")
def iterations():
    """Timed cycles, from ``$CJFC_RT_ITERATIONS``."""
    return helpers.rt_iterations()


@pytest.fixture
def realtime(
    run, build, plugin, artifacts, tmp_path_factory, load_json, iterations
):
    """One periodic run, shared by every test that reads its report: 2000
    cycles at 10 ms is twenty seconds of wall clock.
    """

    def go():
        out = tmp_path_factory.mktemp("realtime") / "report.json"
        result = run(
            realtime_argv(build, artifacts, out, iterations),
            timeout=120.0 + iterations * PERIOD_US / 1e6,
        )
        return result, load_json(out)

    return helpers.cached("realtime", go)


@pytest.fixture
def realtime_census(
    run, build, plugin, artifacts, guard_so, tmp_path_factory, load_json
):
    """The same loop under the allocation interposer."""

    def go():
        out = tmp_path_factory.mktemp("realtime_census") / "census.json"
        result = run(
            realtime_argv(build, artifacts, out, CENSUS_ITERATIONS),
            env=helpers.preload(guard_so),
        )
        return result, load_json(out)

    return helpers.cached("realtime_census", go)


def test_minimal_loop_exits_zero_and_prints_two_summaries(
    run, build, plugin, artifacts
):
    """``example_03_minimal`` loaded, computed the residual, and each recorder
    printed a summary: the failure this catches is a copied file that
    compiles and then measures nothing.
    """
    result = run(
        [
            build.bin("example_03_minimal"),
            artifacts.basic,
            "2000",
            "100",
        ]
    )
    assert result.returncode == 0
    for label in ("wake-up latency", "call latency"):
        section = re.search(
            rf"^=== {re.escape(label)} .*?^\s*$",
            result.stdout,
            re.MULTILINE | re.DOTALL,
        )
        assert section is not None, f"no {label} summary in:\n{result.stdout}"
        assert "p50" in section.group(0)
    assert "max_residual=" in result.stdout


def test_exits_zero(realtime):
    result, report = realtime
    assert result.returncode == 0
    assert report["exit_code"] == 0


def test_report_has_every_top_level_key(realtime):
    _, report = realtime
    assert TOP_LEVEL_KEYS <= set(report), TOP_LEVEL_KEYS - set(report)
    assert report["schema"] == 1
    assert report["example"] == "04_realtime"


def test_config_records_what_was_asked_for(realtime, iterations):
    _, report = realtime
    assert report["config"]["period_us"] == PERIOD_US
    assert report["config"]["iterations"] == iterations


def test_every_hardening_step_says_what_it_did(realtime):
    """Each step reports whether it worked *and* why, in words.  An
    unprivileged run is expected to lose some of these, and the sentence
    saying which one and what it needed is what makes that survivable.
    """
    _, report = realtime
    hardening = report["hardening"]
    assert hardening, "no hardening steps were recorded"
    for name, step in hardening.items():
        assert isinstance(step["ok"], bool), name
        assert step["detail"].strip(), f"{name} reports ok without a reason"


def test_the_corral_finds_xlas_pool_threads(
    run, build, plugin, artifacts, tmp_path, load_json
):
    """``corral_xla_threads`` moves the threads XLA actually started.

    Its own run: two workers and the affinity mask left whole (``--cpu
    none``), so the corral has something to find and somewhere to move it.
    A helper matching a thread name XLA no longer uses finds nothing on every
    host and says so in a sentence that reads like success, which is why the
    count is asserted.
    """
    out = tmp_path / "corral.json"
    result = run(
        realtime_argv(build, artifacts, out, 20, warmup=5)
        + ["--threads", "2", "--cpu", "none"],
        timeout=180.0,
    )
    assert result.returncode == 0, result.stderr
    step = load_json(out)["hardening"]["corral_xla_threads"]
    assert step["ok"] is True, step["detail"]

    # "moved 2 of 2 XLA threads", read as numbers so a version of XLA that
    # names more threads than were asked for still passes.
    counts = re.search(r"moved (\d+) of (\d+)", step["detail"])
    assert counts, f"unreadable detail: {step['detail']!r}"
    moved, seen = int(counts.group(1)), int(counts.group(2))
    assert moved == seen, step["detail"]
    assert moved >= 2, step["detail"]


def test_every_cycle_was_recorded(realtime, iterations):
    _, report = realtime
    assert report["compute_us"]["count"] == iterations
    assert report["compute_us"]["dropped"] == 0


@pytest.mark.parametrize("series", SERIES)
def test_each_series_is_a_full_summary(realtime, series):
    """Present, numeric and ordered.  ``period_jitter_us`` is signed --
    waking early is as much a defect as waking late -- so the only honest
    check on it is that its numbers are numbers.
    """
    _, report = realtime
    summary = report[series]
    for field in ("min_us", "p50_us", "p99_us", "p999_us", "max_us"):
        value = summary[field]
        assert isinstance(value, numbers.Real), f"{series}.{field}"
        assert not math.isnan(value), f"{series}.{field} is NaN"
    assert helpers.ascending(
        [summary[f] for f in ("min_us", "p50_us", "p99_us", "max_us")]
    ), summary


def test_period_jitter_is_allowed_to_be_negative(realtime):
    """The signed field is signed, and the test says so rather than clamps."""
    _, report = realtime
    jitter = report["period_jitter_us"]
    assert isinstance(jitter["min_us"], numbers.Real)
    assert jitter["min_us"] <= jitter["p50_us"] <= jitter["max_us"]


def test_the_step_counter_agrees(realtime):
    _, report = realtime
    assert report["checks"]["step_counter_ok"] is True
    assert report["checks"]["step_errors"] == 0


def test_no_major_faults(realtime):
    _, report = realtime
    assert report["rusage"]["majflt"] == 0


def test_the_census_measured_something(realtime_census):
    """``guard_present: false`` is a record of "not measured", which must
    never read as zero allocations.
    """
    result, report = realtime_census
    assert result.returncode == 0
    allocations = report["allocations"]
    assert allocations["guard_present"] is True
    assert allocations["classified"] is True


def test_the_loop_allocates_nothing_of_its_own(realtime_census):
    """``self`` is the number that must be zero; the plugin's count is XLA's
    thunk runtime, not reachable from this side of the C API.
    """
    _, report = realtime_census
    allocations = report["allocations"]
    assert allocations["self"] == 0
    assert allocations["per_iteration"]["self"] == 0
    if helpers.alloc_strict():
        assert allocations["runtime"] == 0


@pytest.mark.rt
def test_tail_ratios(realtime):
    """p99.9/p50 and max/p50, not the mean: the mean of a control loop's call
    latency cannot tell you whether it will make its deadline.
    """
    _, report = realtime
    compute = report["compute_us"]
    assert compute["p999_over_p50"] <= MAX_P999_OVER_P50
    assert compute["max_over_p50"] <= MAX_MAX_OVER_P50


@pytest.mark.rt
def test_wake_latency(realtime):
    """How late the kernel returned from the sleep, at p99."""
    _, report = realtime
    assert report["wake_latency_us"]["p99_us"] <= MAX_WAKE_P99_US


@pytest.mark.rt
def test_no_deadline_was_missed(realtime):
    _, report = realtime
    assert report["deadlines"]["missed"] == 0


@pytest.mark.rt
def test_no_faults_and_few_involuntary_switches(realtime, iterations):
    """A minor fault inside the window means ``mlockall`` did not hold; an
    involuntary switch means something outranked a SCHED_FIFO loop.  The
    switch budget scales with the run.
    """
    _, report = realtime
    rusage = report["rusage"]
    assert rusage["minflt"] == 0
    assert rusage["majflt"] == 0
    assert rusage["nivcsw"] <= max(2, iterations // 200)


@pytest.mark.rt
def test_hardening_succeeded(realtime):
    """Everything the loop asked the kernel for, and could have had, it got.
    ``cpu_dma_latency`` is exempt unless this process is root; the other
    exemptions are :data:`NOT_A_FAILURE`, matched on the detail.
    """
    _, report = realtime
    failed = {}
    for name, step in report["hardening"].items():
        if step["ok"]:
            continue
        if name == "cpu_dma_latency" and helpers.is_root():
            failed[name] = step["detail"]
            continue
        excused = NOT_A_FAILURE.get(name, ())
        if not any(reason in step["detail"] for reason in excused):
            failed[name] = step["detail"]
    assert not failed, failed


def test_run_realtime_script_audits_before_it_measures(run, repo, plugin):
    """``run_realtime.sh`` prints the host audit first, then runs.  Positions,
    not presence: a latency figure whose provenance was written down
    afterwards is one nobody can defend later.
    """
    launcher = repo.root / "examples/04_realtime/run_realtime.sh"
    if not launcher.is_file():
        pytest.skip(f"{launcher} is not in this checkout")

    result = run(
        [
            launcher,
            "--iterations",
            "60",
            "--period-us",
            "5000",
            "--warmup",
            "10",
            "--quiet",
        ],
        timeout=300.0,
    )
    audit = result.stdout.find("=== real-time host audit ===")
    assert audit >= 0, "the script ran without printing tools/rt_check.sh"

    launched = result.stdout.find("example_04_realtime")
    assert launched > audit, "the audit must come before the run, not after"
    assert "04_realtime n=60" in result.stdout
