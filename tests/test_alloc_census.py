"""The allocation census: what the steady-state path allocates, and what it
must not.

``self``, the calling binary's own allocations, must be zero.  ``runtime``,
the wrapper library's, is zero here too but asserted only under
``$CJFC_ALLOC_STRICT=1``.  ``plugin`` is XLA's thunk runtime, thousands per
call and not reachable from this side of the C API.  ``--require-guard``
exits 4 when the interposer was not preloaded, distinct from the 3 that means
the path allocated: in a green log "not measured" and "zero" are otherwise
the same thing.
"""

from __future__ import annotations

import helpers
import pytest

#: Exit codes the examples and the benchmark share; see report.hpp.
EXIT_OK = 0
EXIT_GUARD_MISSING = 4

#: A floor, not a count: the selftest makes a ``malloc``/``free`` pair, a
#: ``new``/``delete`` pair and a growing ``std::vector``, and how often the
#: vector reallocates is the standard library's business.
MIN_SELFTEST_ALLOCS = 3


def census_argv(build, artifacts, out):
    return [
        build.bin("bench"),
        "--fixture",
        "trajopt",
        "--assets-dir",
        artifacts.dir,
        "--artifacts-dir",
        artifacts.dir,
        "--iterations",
        "100",
        "--warmup",
        "10",
        "--alloc-gate",
        "self",
        "--require-guard",
        "--json",
        out,
    ]


@pytest.fixture
def bench_census(
    run, build, plugin, artifacts, guard_so, tmp_path_factory, load_json
):
    """One armed benchmark run, shared by the tests that read it."""

    def go():
        out = tmp_path_factory.mktemp("census") / "census.json"
        result = run(
            census_argv(build, artifacts, out),
            env=helpers.preload(guard_so),
        )
        return result, load_json(out)

    return helpers.cached("bench_census", go)


def test_the_selftest_without_the_preload_reports_absence(
    run, build, parse_kv_lines
):
    """No interposer, no census -- and it says so rather than reporting zero."""
    result = run([build.bin("test_guard_selftest")])
    reported = parse_kv_lines(result.stdout)
    assert reported["present"] == "0"
    assert reported["classified"] == "0"


def test_the_selftest_under_the_preload_counts_its_own_allocations(
    run, build, guard_so, parse_kv_lines
):
    """The interposer sees the allocations and charges them to this binary.

    ``classified=1`` is the half that matters: without module ranges the
    guard can count but not attribute, and the ``self`` gate falls back to
    the total.
    """
    result = run(
        [build.bin("test_guard_selftest")], env=helpers.preload(guard_so)
    )
    reported = parse_kv_lines(result.stdout)
    assert reported["present"] == "1"
    assert reported["classified"] == "1"
    assert int(reported["self"]) >= MIN_SELFTEST_ALLOCS
    assert int(reported["runtime"]) == 0
    assert int(reported["plugin"]) == 0


def test_bench_passes_the_self_gate(bench_census):
    result, _ = bench_census
    assert result.returncode == EXIT_OK


def test_bench_allocates_nothing_of_its_own(bench_census):
    _, report = bench_census
    allocations = report["allocations"]
    assert allocations["guard_present"] is True
    assert allocations["classified"] is True
    assert allocations["self"] == 0
    assert allocations["per_iteration"]["self"] == 0
    if helpers.alloc_strict():
        assert allocations["runtime"] == 0


def test_the_plugin_allocation_is_visible_and_nonzero(bench_census):
    """A census that saw nothing at all would pass every gate above; the
    plugin's count is what proves the interposer was armed over the right
    window rather than merely loaded."""
    _, report = bench_census
    allocations = report["allocations"]
    assert allocations["plugin"] > 0
    assert allocations["process_total"] > 0


def test_require_guard_fails_when_nothing_was_measured(
    run, build, plugin, artifacts, tmp_path
):
    """Exit 4, not 0: "nobody measured" is not a pass.

    Takes ``plugin`` although the guard is the subject: without a plugin the
    binary exits earlier with a load error and never reaches the gate.
    """
    out = tmp_path / "unmeasured.json"
    result = run(census_argv(build, artifacts, out), check=False)
    assert result.returncode == EXIT_GUARD_MISSING, result.stderr


def test_the_value_audit_does_not_allocate_either(
    run, build, plugin, artifacts, guard_so, tmp_path, load_json
):
    """The per-call finite-output audit is on, and ``self`` is still zero.

    ``example_02_trajopt`` walks every floating output inside the armed
    window unless ``--no-check`` is given, so this is the census of the
    *checked* path.  The check has to be as allocation-free as the call it
    guards, or nobody could afford to leave it on.
    """
    out = tmp_path / "checked.json"
    result = run(
        [
            build.bin("example_02_trajopt"),
            "--artifact",
            artifacts.trajopt,
            "--iterations",
            "50",
            "--warmup",
            "10",
            "--alloc-gate",
            "self",
            "--require-guard",
            "--json",
            out,
        ],
        env=helpers.preload(guard_so),
    )
    assert result.returncode == EXIT_OK

    report = load_json(out)
    assert report["checks"]["finite_outputs"] is True
    allocations = report["allocations"]
    assert allocations["guard_present"] is True
    assert allocations["self"] == 0
    if helpers.alloc_strict():
        assert allocations["runtime"] == 0
