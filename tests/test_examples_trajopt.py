"""``example_02_trajopt``: the JSON report, and a long campaign.

This is the example the project quotes numbers from, so what is checked here
is the *shape* of the measurement rather than its value: that the recorder
kept every sample it was asked for, that the percentiles are ordered, that
the step counter and the finite-output audit both passed, and that no page
fault was taken inside the timed window.

What is deliberately not checked is how fast it was.  On this host the same
artifact on the same pinned core reports p50 2027 us at a 3 ms period and
5196 us at a 10 ms period, because the powersave governor clocks the core
down while the loop idles between calls; ``min`` barely moves throughout.  A
fixed threshold would therefore fail on a perfectly healthy machine.  The
long campaign records its tail ratios as properties instead, where a human
comparing two runs can see them without a test having pretended to know what
they should be.
"""

from __future__ import annotations

import json
import math

import helpers
import pytest

#: The trajectory optimizer runs at most five Gauss-Newton iterations; the
#: exported function's ``cost_history`` has exactly that many slots.
MAX_SOLVER_ITERATIONS = 5

#: Calls in the short campaign.  Enough for a p99.9 to exist, quick enough
#: that every test in this file can share one run.
SHORT_ITERATIONS = 300

#: Calls in the campaign behind ``--runslow``.
LONG_ITERATIONS = 20000

#: The percentile fields, in the order they must not violate.
ORDERED = ("min_us", "p50_us", "p90_us", "p99_us", "p999_us", "max_us")


def trajopt(run, build, artifacts, out, iterations, warmup):
    """One measured run of the example, writing its report to @p out."""
    return run(
        [
            build.bin("example_02_trajopt"),
            "--artifact",
            artifacts.trajopt,
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--json",
            out,
        ]
    )


@pytest.fixture
def short_campaign(run, build, plugin, artifacts, tmp_path_factory, load_json):
    """300 timed calls, reported as JSON, shared by the tests below."""

    def go():
        out = tmp_path_factory.mktemp("trajopt") / "report.json"
        result = trajopt(
            run, build, artifacts, out, SHORT_ITERATIONS, warmup=20
        )
        return result, load_json(out)

    return helpers.cached("trajopt_short", go)


def test_exits_zero(short_campaign):
    result, report = short_campaign
    assert result.returncode == 0
    assert report["exit_code"] == 0


def test_report_identifies_itself(short_campaign):
    _, report = short_campaign
    assert report["schema"] == 1
    assert report["example"] == "02_trajopt"


def test_every_sample_was_kept(short_campaign):
    """300 asked for, 300 recorded, none dropped.

    The recorder reserves its capacity once and drops rather than grows,
    because growing would allocate in the middle of the window being
    measured.  A non-zero ``dropped`` means the summary describes fewer
    calls than the run made, which quietly changes what a percentile is a
    percentile *of*.
    """
    _, report = short_campaign
    assert report["compute_us"]["count"] == SHORT_ITERATIONS
    assert report["compute_us"]["dropped"] == 0


def test_percentiles_are_ordered(short_campaign):
    _, report = short_campaign
    compute = report["compute_us"]
    values = [compute[field] for field in ORDERED]
    assert helpers.ascending(values), dict(zip(ORDERED, values, strict=True))


def test_the_step_counter_agrees(short_campaign):
    """``step_next`` was exactly ``step + 1`` on every call.

    An integer identity rather than a tolerance: it is what a stale or
    unread input arena breaks, and no float comparison would see it.
    """
    _, report = short_campaign
    assert report["checks"]["step_counter_ok"] is True
    assert report["checks"]["step_errors"] == 0


def test_every_output_stayed_finite(short_campaign):
    _, report = short_campaign
    assert report["checks"]["finite_outputs"] is True


def test_no_major_faults_in_the_timed_window(short_campaign):
    """A major fault is a disk read inside a call.  There must be none."""
    _, report = short_campaign
    assert report["rusage"]["majflt"] == 0


def test_the_solver_reports_a_real_iteration_count(
    short_campaign, parse_kv_lines
):
    """The last call's ``iterations_used`` is in range.

    Out of range means the integer outputs are not being read back from the
    arenas the executable wrote, which the step counter alone would not
    necessarily catch: it checks one scalar, and this checks another.
    """
    result, _ = short_campaign
    used = int(parse_kv_lines(result.stdout)["result"]["iterations_used"])
    assert 0 <= used <= MAX_SOLVER_ITERATIONS


@pytest.mark.slow
def test_long_campaign(
    run, build, plugin, artifacts, tmp_path, load_json, record_property
):
    """20,000 calls: does it stay correct and consistent for a long run?

    Gated on the counters, not on the clock.  The tail ratios are recorded
    as properties so that a reader can compare two runs, because on this
    host the number depends on the governor and the caller's period rather
    than on the call path -- see the module docstring.
    """
    out = tmp_path / "long.json"
    result = trajopt(run, build, artifacts, out, LONG_ITERATIONS, warmup=200)
    assert result.returncode == 0

    report = load_json(out)
    compute = report["compute_us"]
    assert compute["count"] == LONG_ITERATIONS
    assert compute["dropped"] == 0
    assert helpers.ascending([compute[field] for field in ORDERED])

    assert report["checks"]["step_counter_ok"] is True
    assert report["checks"]["step_errors"] == 0
    assert report["checks"]["finite_outputs"] is True
    assert report["rusage"]["majflt"] == 0

    # Every reported number has to be a number.  A NaN percentile is how a
    # recorder that overflowed, or divided by an empty count, announces
    # itself -- and it would satisfy every ordering check above.
    for field, value in compute.items():
        if isinstance(value, float):
            assert not math.isnan(value), f"compute_us.{field} is NaN"

    record_property("loadavg1", report["host"]["loadavg1"])
    record_property("host_busy", report["host"]["busy"])
    record_property("p50_us", compute["p50_us"])
    record_property("p999_over_p50", compute["p999_over_p50"])
    record_property("max_over_p50", compute["max_over_p50"])


# ------------------------------------------------------------ the gates fire
#
# Every other assertion here is that a gate reported success, which is also
# what a gate that cannot fail reports. `--inject-fault` corrupts exactly one
# cycle so the gates can be watched failing. Until these existed, deleting
# either check left the whole suite green.


@pytest.mark.parametrize(
    ("fault", "flipped", "still_ok"),
    [
        ("step", "step_counter_ok", "finite_outputs"),
        ("nonfinite", "finite_outputs", "step_counter_ok"),
    ],
)
def test_an_injected_fault_trips_its_own_gate(
    build, plugin, artifacts, repo, run, fault, flipped, still_ok
):
    """One corrupted cycle must fail the run, and only through its own check."""
    out = repo.report_dir / f"trajopt_fault_{fault}.json"
    result = run(
        [
            build.bin("example_02_trajopt"),
            "--iterations",
            "40",
            "--warmup",
            "5",
            "--inject-fault",
            fault,
            "--json",
            out,
        ],
        check=False,
    )

    assert result.returncode == 2, (
        f"--inject-fault {fault} must exit 2 (a correctness failure), "
        f"got {result.returncode}"
    )
    checks = json.loads(out.read_text())["checks"]
    assert checks[flipped] is False, f"{flipped} did not notice the fault"
    assert checks[still_ok] is True, (
        f"{still_ok} also failed; the fault is not as targeted as it claims"
    )


def test_no_fault_is_the_default(build, plugin, artifacts, repo, run):
    """The control for the two above: the same run, uncorrupted, passes."""
    out = repo.report_dir / "trajopt_fault_none.json"
    result = run(
        [
            build.bin("example_02_trajopt"),
            "--iterations",
            "40",
            "--warmup",
            "5",
            "--json",
            out,
        ]
    )
    assert result.returncode == 0
    checks = json.loads(out.read_text())["checks"]
    assert checks["step_counter_ok"] is True
    assert checks["finite_outputs"] is True
