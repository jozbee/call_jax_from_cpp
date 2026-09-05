"""``bench``: does the fast path compute the same thing as the reference?

The sweep is the load-bearing test here, and the reason it runs two passes
is worth stating.  The steady-state path reuses one set of zero-copy input
buffers for the life of a ``Function``, so what has to be exercised is
*changing* the inputs between calls -- which a benchmark that feeds the same
case every time never does.  Two passes, forwards then backwards, give every
case a different predecessor: a buffer left stale by the previous call
produces the right answer for exactly one ordering, and one pass would call
that acceptable.

Tolerances come from the fixture manifest rather than from a constant in this
file.  They are what ``jax2exec.reference`` froze the cases with, and a test
carrying its own copy would be free to drift away from what the exporter
promised.
"""

from __future__ import annotations

import re

import helpers
import pytest

#: ``  pass 0 case 3: max rel err ...`` -- one line per case per pass.
CASE_LINE = re.compile(r"^\s*pass (\d+) case (\d+):", re.MULTILINE)

#: The sweep's verdict, printed only when every case agreed.
AGREEMENT = "all cases agree with the reference"


def sweep_argv(build, artifacts, *extra, artifacts_dir=None):
    return [
        build.bin("bench"),
        "--all-cases",
        "--fixture",
        "trajopt",
        "--assets-dir",
        artifacts.dir,
        "--artifacts-dir",
        artifacts_dir or artifacts.dir,
        *extra,
    ]


@pytest.fixture
def tolerance(artifacts, load_json):
    """The loosest per-dtype tolerance the fixture manifest declares.

    ``max_rel_err`` is a maximum over every floating element of every
    output, float32 and float64 together, so the only number it can honestly
    be compared against is the looser of the two.  Gating per dtype is
    ``bench``'s own job, and its verdict -- ``correctness.ok`` -- is asserted
    separately below.
    """

    def go():
        declared = load_json(artifacts.trajopt_cases).get("tolerance") or {}
        values = [float(value) for value in declared.values()]
        if not values:
            pytest.fail(f"{artifacts.trajopt_cases} declares no tolerance")
        return max(values)

    return helpers.cached("trajopt_tolerance", go)


@pytest.fixture
def all_cases(run, build, plugin, artifacts):
    """The correctness sweep, shared by the tests that read it."""

    def go():
        return run(sweep_argv(build, artifacts))

    return helpers.cached("bench_all_cases", go)


@pytest.fixture
def timed_run(run, build, plugin, artifacts, tmp_path_factory, load_json):
    """A short timed run, which gates correctness before it times anything."""

    def go():
        out = tmp_path_factory.mktemp("bench") / "timed.json"
        result = run(
            [
                build.bin("bench"),
                "--fixture",
                "trajopt",
                "--assets-dir",
                artifacts.dir,
                "--artifacts-dir",
                artifacts.dir,
                "--iterations",
                "50",
                "--warmup",
                "5",
                "--json",
                out,
            ]
        )
        return result, load_json(out)

    return helpers.cached("bench_timed", go)


def test_sweep_exits_zero(all_cases):
    assert all_cases.returncode == 0


def test_sweep_covers_every_case_twice(all_cases, artifacts, load_json):
    """Two passes over all four cases, forwards then backwards.

    Counting the lines rather than trusting the closing verdict: a sweep
    that silently ran one pass would still print that everything agreed, and
    the second pass is the half that catches a stale reused buffer.
    """
    n_cases = len(load_json(artifacts.trajopt_cases)["cases"])
    seen = CASE_LINE.findall(all_cases.stdout)
    assert len(seen) == 2 * n_cases, all_cases.stdout

    forwards = [int(case) for passno, case in seen if passno == "0"]
    backwards = [int(case) for passno, case in seen if passno == "1"]
    assert forwards == list(range(n_cases))
    assert backwards == list(reversed(range(n_cases)))


def test_sweep_agrees_with_the_reference(all_cases):
    assert AGREEMENT in all_cases.stdout
    assert "FAIL" not in all_cases.stdout


def test_every_case_is_within_tolerance(all_cases, tolerance):
    """Each printed comparison, not only the summary line.

    ``bench`` prints ``max rel err``, the exact mismatches and the NaN count
    for every case; the NaN count is the one a maximum-of-relative-errors
    check cannot see at all, since a NaN difference is not a large error but
    an absent one.
    """
    reported = re.findall(
        r"max rel err ([\d.e+-]+), (\d+) exact mismatches, (\d+) nan",
        all_cases.stdout,
    )
    assert reported, all_cases.stdout
    for rel_err, mismatches, nans in reported:
        assert float(rel_err) <= tolerance
        assert int(mismatches) == 0
        assert int(nans) == 0


def test_async_is_still_correct(run, build, plugin, artifacts):
    """A plugin that ignores ``--async`` must still be right.

    Whether the plugin honours the request is a latency question.  Whether
    the answers are still correct is not negotiable either way, and this is
    the test that says so.
    """
    result = run(sweep_argv(build, artifacts, "--async"))
    assert result.returncode == 0
    assert AGREEMENT in result.stdout


def test_timed_run_exits_zero(timed_run):
    result, _ = timed_run
    assert result.returncode == 0


def test_timed_run_checked_its_answer(timed_run):
    """The gate ran.  A run that skipped it must not read as a clean one."""
    _, report = timed_run
    assert report["checked"] is True
    assert report["correctness"]["ok"] is True


def test_timed_run_is_within_tolerance(timed_run, tolerance):
    _, report = timed_run
    correctness = report["correctness"]
    assert correctness["nan_count"] == 0
    assert correctness["exact_mismatches"] == 0
    assert correctness["max_rel_err"] <= tolerance
