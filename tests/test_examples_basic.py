"""``example_01_basic``: the whole call path, driven from outside.

The smallest end-to-end statement the project makes -- load an artifact,
call it once, get an answer that solves the system -- so these are the
assertions every other test assumes have passed.  ``load_kind=deserialized``
is a gate, not a report: ``compiled`` means the ``.binpb`` did not load and
the ``.mlirbc`` fallback quietly took over, a working program and a broken
deployment.
"""

from __future__ import annotations

import helpers
import pytest

#: The example's own tolerance, from ``kResidualTolerance`` in basic.cpp.
RESIDUAL_TOLERANCE = 1e-9

#: Each debug check the example demonstrates, and the fragment of its message
#: that docs/guides/debugging.md documents.
DEBUG_CHECKS = [
    ("out_of_range", "is out of range: function"),
    ("dtype_mismatch", "but was accessed as"),
    ("non_finite", "is nan"),
]


def basic_argv(build, artifacts, *extra):
    return [
        build.bin("example_01_basic"),
        "--artifact",
        artifacts.basic,
        *extra,
    ]


@pytest.fixture
def basic_run(run, build, plugin, artifacts):
    """One plain run of the example, shared by the tests that read it."""

    def go():
        return run(basic_argv(build, artifacts))

    return helpers.cached("basic_run", go)


@pytest.fixture
def basic_debug_run(run, build, plugin, artifacts):
    """The same run with the per-call checks turned on."""

    def go():
        return run(basic_argv(build, artifacts, "--debug"))

    return helpers.cached("basic_debug_run", go)


def test_exits_zero(basic_run):
    assert basic_run.returncode == 0


def test_loaded_the_serialized_executable(basic_run, parse_kv_lines):
    """The AOT path, not the compile fallback: nothing downstream of this line
    says which of the two happened."""
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["load_kind"] == "deserialized", (
        "the .binpb did not deserialize and the .mlirbc fallback took over; "
        "re-export, or read load_detail in example_02_trajopt's output"
    )


def test_reports_inline_execution(basic_run, parse_kv_lines):
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["synchronous_supported"] == "1"
    assert reported["sync_mode"] == "inline"


def test_reports_the_arity(basic_run, parse_kv_lines):
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["num_inputs"] == "2"
    assert reported["num_outputs"] == "2"


def test_reports_the_rank_two_input(basic_run):
    """The spec line, whole: ``numel`` and ``nbytes`` are the arithmetic the
    loader allocates against, so the three numbers are one statement."""
    assert (
        "input[0]: dtype=float64 shape=[4,4] numel=16 nbytes=128"
        in basic_run.stdout.splitlines()
    )


def test_reports_the_scalar_output(basic_run, parse_kv_lines):
    """A rank-0 output is ``shape=[]``, one element, eight bytes."""
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["output[1]"] == {
        "dtype": "float64",
        "shape": "[]",
        "numel": "1",
        "nbytes": "8",
    }


def test_the_solution_solves_the_system(basic_run, parse_kv_lines):
    """Two independent residuals: one recomputed in C++ from the input
    arenas, one the executable itself produced."""
    reported = parse_kv_lines(basic_run.stdout)
    assert float(reported["residual_inf_norm"]) < RESIDUAL_TOLERANCE
    assert float(reported["residual_from_jax"]) < RESIDUAL_TOLERANCE


def test_without_debug_says_the_checks_are_off(basic_run, parse_kv_lines):
    """``debug=0`` is stated, not merely silent: a run with no
    ``debug_check`` line could otherwise mean the checks passed."""
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["debug"] == "0"
    assert not [key for key in reported if key.startswith("debug_check[")]


@pytest.mark.parametrize(("check", "fragment"), DEBUG_CHECKS)
def test_debug_demonstrates_each_check(
    basic_debug_run, parse_kv_lines, check, fragment
):
    """Each deliberate mistake is caught and names the array it was about."""
    reported = parse_kv_lines(basic_debug_run.stdout)
    message = reported.get(f"debug_check[{check}]")
    assert message is not None, basic_debug_run.stdout
    assert fragment in message


def test_debug_names_the_array_by_name_not_only_by_index(
    basic_debug_run, parse_kv_lines
):
    """``input 0 ('A')``: the index *and* the name the exporter gave it."""
    reported = parse_kv_lines(basic_debug_run.stdout)
    assert "input 0 ('A')" in reported["debug_check[dtype_mismatch]"]
    assert "input 0 ('A')" in reported["debug_check[non_finite]"]


def test_debug_run_still_solves_the_system(basic_debug_run, parse_kv_lines):
    """``--debug`` changes what is checked, not what is computed: the NaN
    goes in after the good call, and the next call is refused."""
    reported = parse_kv_lines(basic_debug_run.stdout)
    assert float(reported["residual_inf_norm"]) < RESIDUAL_TOLERANCE
