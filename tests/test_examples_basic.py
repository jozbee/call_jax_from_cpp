"""``example_01_basic``: the whole call path, driven from outside.

This is the smallest end-to-end statement the project makes -- load an
exported artifact, call it once, and get an answer that solves the system --
so the assertions here are the ones every other test assumes have passed.

Two of them are worth naming.  ``load_kind=deserialized`` is a gate rather
than a report: ``compiled`` there means the ``.binpb`` did not load and the
``.mlirbc`` fallback quietly took over, which is a working program and a
broken deployment, because the fallback spends seconds running the XLA
pipeline where the AOT path spends milliseconds relinking.  And the residual
is recomputed in C++ from the same arenas XLA read, so a layout mistake on
the calling side shows up here as a large number rather than as a
plausible-looking ``x``.
"""

from __future__ import annotations

import helpers
import pytest

#: The example's own tolerance, from ``kResidualTolerance`` in basic.cpp.
RESIDUAL_TOLERANCE = 1e-9

#: Each debug check the example demonstrates, and the fragment of its
#: message that identifies it.  These fragments are what
#: docs/guides/debugging.md documents and what anyone greps a log for, so a
#: rewording that keeps the exception type but loses the words is still a
#: break.
DEBUG_CHECKS = [
    ("out_of_range", "is out of range: function"),
    ("dtype_mismatch", "but was accessed as"),
    ("non_finite", "is nan"),
]


@pytest.fixture
def basic_run(run, build, plugin, artifacts):
    """One plain run of the example, shared by the tests that read it."""

    def go():
        return run(
            [build.bin("example_01_basic"), "--artifact", artifacts.basic]
        )

    return helpers.cached("basic_run", go)


@pytest.fixture
def basic_debug_run(run, build, plugin, artifacts):
    """The same run with the per-call checks turned on."""

    def go():
        return run(
            [
                build.bin("example_01_basic"),
                "--artifact",
                artifacts.basic,
                "--debug",
            ]
        )

    return helpers.cached("basic_debug_run", go)


def test_exits_zero(basic_run):
    assert basic_run.returncode == 0


def test_loaded_the_serialized_executable(basic_run, parse_kv_lines):
    """The AOT path, not the compile fallback.

    Failing on ``compiled`` is the point: it is the difference between a
    load that relinks machine code and one that runs the whole compiler, and
    nothing downstream of this line says which of the two happened.
    """
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
    """The spec line, spelled exactly.

    Asserted as a whole line rather than field by field: this line is a
    machine interface, and ``numel=16`` with ``nbytes=128`` for a float64
    ``[4,4]`` is the arithmetic the loader allocates against.  A sidecar
    whose own numbers disagree with its shape is refused, so the three of
    them are one statement, not three.
    """
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
    """Two independent residuals, both small.

    ``residual_inf_norm`` is recomputed in C++ from the input arenas;
    ``residual_from_jax`` is the 2-norm the executable itself produced.
    Neither is derived from the other, so both being small is two
    statements rather than one repeated.
    """
    reported = parse_kv_lines(basic_run.stdout)
    assert float(reported["residual_inf_norm"]) < RESIDUAL_TOLERANCE
    assert float(reported["residual_from_jax"]) < RESIDUAL_TOLERANCE


def test_without_debug_says_the_checks_are_off(basic_run, parse_kv_lines):
    """The absence is stated, not merely silent.

    A run with no ``debug_check`` line could mean the checks passed or that
    they were never enabled; the ``debug=0`` line is what tells those apart,
    which is why this also asserts that nothing reported.
    """
    reported = parse_kv_lines(basic_run.stdout)
    assert reported["debug"] == "0"
    assert not [key for key in reported if key.startswith("debug_check[")]


@pytest.mark.parametrize(("check", "fragment"), DEBUG_CHECKS)
def test_debug_demonstrates_each_check(
    basic_debug_run, parse_kv_lines, check, fragment
):
    """Each deliberate mistake is caught and names the array it was about.

    The example makes all three on purpose -- an index past the end, a
    ``float*`` into a float64 arena, and a NaN written into an input -- and
    every one of them is silent with the checks off: the first walks past
    the end of a vector, the second corrupts half the matrix on its first
    write, and the third is simply computed with.
    """
    reported = parse_kv_lines(basic_debug_run.stdout)
    message = reported.get(f"debug_check[{check}]")
    assert message is not None, basic_debug_run.stdout
    assert fragment in message


def test_debug_names_the_array_by_name_not_only_by_index(
    basic_debug_run, parse_kv_lines
):
    """``input 0 ('A')``: the index *and* the name the exporter gave it.

    The index alone sends a reader counting arguments in a sidecar; the name
    is the half that makes the message actionable.
    """
    reported = parse_kv_lines(basic_debug_run.stdout)
    assert "input 0 ('A')" in reported["debug_check[dtype_mismatch]"]
    assert "input 0 ('A')" in reported["debug_check[non_finite]"]


def test_debug_run_still_solves_the_system(basic_debug_run, parse_kv_lines):
    """``--debug`` changes what is checked, not what is computed.

    The NaN the demonstration writes goes in after the good call, and
    ``check_values`` refuses the next call rather than running it, so the
    results printed above are the untouched ones.
    """
    reported = parse_kv_lines(basic_debug_run.stdout)
    assert float(reported["residual_inf_norm"]) < RESIDUAL_TOLERANCE
