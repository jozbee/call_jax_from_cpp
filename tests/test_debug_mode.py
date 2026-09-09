"""The signature the loader resolved, and what the debug checks catch.

``fn_info`` prints the resolved signature; the assertions on it spell out the
trajopt calling convention rather than read it back from the sidecar, which
would make the test agree with whatever the sidecar said.

``test_debug_checks`` makes one specific mistake, once with ``--debug`` and
once without; the pair is the test.  With the checks off nothing fails and
nothing is logged, and the answer the caller goes on to use is a NaN.

The message fragments asserted are the ones docs/guides/debugging.md
documents, so a rewording that loses the words breaks anyone grepping a log.
"""

from __future__ import annotations

import json

import pytest
from conftest import rt_strict

#: The trajopt input dtypes, in the order the executable takes them.  Order
#: *is* the calling convention: there is no name-based binding at the C API.
TRAJOPT_INPUT_DTYPES = [
    "float64",
    "float64",
    "float64",
    "float64",
    "float32",
    "bool",
    "int32",
]

TRAJOPT_NUM_OUTPUTS = 8

#: Scenarios on the trajopt artifact, and the fragment each message carries.
TRAJOPT_SCENARIOS = [
    ("oob-input", "is out of range: function"),
    ("oob-output", "is out of range: function"),
    ("dtype-mismatch", "but was accessed as"),
    ("nonfinite-input", "is nan"),
    ("bool-value", "bool arenas must hold 0 or 1"),
]


def check(run, build, base, scenario, *, debug):
    """Run one scenario and return the single line it printed."""
    argv = [build.bin("test_debug_checks"), base, scenario]
    if debug:
        argv.append("--debug")
    result = run(argv)
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("THREW: ", "NO-THROW"))
    ]
    assert len(lines) == 1, result.stdout
    return lines[0]


@pytest.fixture
def signature(run, build, plugin, artifacts):
    """``fn_info`` on the trajopt artifact, parsed."""
    result = run([build.bin("fn_info"), artifacts.trajopt])
    return json.loads(result.stdout)


def test_reports_the_trajopt_arity(signature):
    assert len(signature["inputs"]) == len(TRAJOPT_INPUT_DTYPES)
    assert len(signature["outputs"]) == TRAJOPT_NUM_OUTPUTS


def test_reports_the_trajopt_input_dtypes(signature):
    dtypes = [entry["dtype"] for entry in signature["inputs"]]
    assert dtypes == TRAJOPT_INPUT_DTYPES


def test_reports_the_reference_trajectory_size(signature):
    """``x_ref`` is 50 by 48; ``numel`` is what the loader allocates against."""
    x_ref = signature["inputs"][1]
    assert x_ref["name"] == "x_ref"
    assert x_ref["shape"] == [50, 48]
    assert x_ref["numel"] == 50 * 48


def test_reports_the_load_kind(signature):
    """The signature came from the deserialized executable, not a compile."""
    assert signature["load_kind"] == "deserialized"


@pytest.mark.parametrize(("scenario", "fragment"), TRAJOPT_SCENARIOS)
def test_the_checks_catch_it(run, build, plugin, artifacts, scenario, fragment):
    line = check(run, build, artifacts.trajopt, scenario, debug=True)
    assert line.startswith("THREW: "), line
    assert fragment in line


@pytest.mark.parametrize(("scenario", "fragment"), TRAJOPT_SCENARIOS)
def test_without_the_checks_nothing_notices(
    run, build, plugin, artifacts, scenario, fragment
):
    """What "off" costs: for two of these the run continues with a corrupted
    arena and reports nothing at all.
    """
    line = check(run, build, artifacts.trajopt, scenario, debug=False)
    assert line.startswith("NO-THROW"), line


def test_a_nonfinite_output_is_caught(run, build, plugin, artifacts):
    """A singular system: the inputs are finite and the arithmetic is not, so
    ``check_values`` has to catch it on the way *out*.
    """
    line = check(run, build, artifacts.basic, "nonfinite-output", debug=True)
    assert line.startswith("THREW: "), line
    assert "is nan" in line or "is inf" in line or "is -inf" in line


def test_a_nonfinite_output_is_handed_back_unchecked(
    run, build, plugin, artifacts
):
    """With the checks off it is returned to the caller, silently."""
    line = check(run, build, artifacts.basic, "nonfinite-output", debug=False)
    assert line.startswith("NO-THROW")
    assert "nan_in_output=1" in line, line


def test_reentrant_call_is_caught(run, build, plugin, artifacts, request):
    """The re-entry guard, which is inherently a race.

    The intruding thread can land between two of the holder's calls instead
    of inside one, so gating on the guard having fired would be flaky by
    construction; on a host nobody has declared tuned and quiet a miss is an
    expected failure, not a real one.
    """
    line = check(run, build, artifacts.trajopt, "reentrant", debug=True)
    assert "attempts=" in line, line
    attempts = int(line.rsplit("attempts=", 1)[1].split()[0])
    assert attempts > 0

    if not line.startswith("THREW: "):
        message = (
            f"the re-entry guard never fired in {attempts} attempts; "
            "the intruding thread never landed inside a call()"
        )
        if rt_strict(request.config):
            pytest.fail(message)
        pytest.xfail(message)
    assert "call() re-entered" in line
