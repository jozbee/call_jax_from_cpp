"""The signature the loader resolved, and what the debug checks catch.

``fn_info`` prints the resolved signature as JSON; the assertions on it are
the trajopt calling convention, spelled out here rather than read back from
the sidecar, because reading it from the sidecar would make the test agree
with whatever the sidecar happened to say.

``test_debug_checks`` makes one specific mistake and prints whether anything
noticed, once with ``--debug`` and once without.  The pair is the test.  With
the checks on, an exception names the array by index *and* by the name the
exporter gave it; with them off, nothing fails, nothing is logged, and the
answer the caller goes on to use is a nan.  The scenarios whose damage lands
in an output arena report ``nan_in_output=`` for exactly that reason, so the
unchecked run says out loud what it handed back.

The message fragments asserted here are the ones docs/guides/debugging.md
documents.  A rewording that keeps the exception type but loses the words is
still a break for anyone grepping a log, which is why the fragment and not
just the fact of a throw is what these check.
"""

from __future__ import annotations

import json

import pytest
from conftest import rt_strict

#: The trajopt input dtypes, in the order the executable takes them.  Order
#: *is* the calling convention -- there is no name-based binding at the C
#: API -- so a permutation here is a caller writing into the wrong arena.
TRAJOPT_INPUT_DTYPES = [
    "float64",
    "float64",
    "float64",
    "float64",
    "float32",
    "bool",
    "int32",
]

#: ``x_ref`` is ``f64[50,48]``, which is 2400 elements.
X_REF = ("x_ref", 2400)

TRAJOPT_NUM_OUTPUTS = 8

#: Scenarios that run on the trajopt artifact, and the fragment of the
#: message each one's exception must carry.
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
    """``x_ref`` is 50 by 48.

    ``numel`` is the number the loader allocates against, so a sidecar whose
    own arithmetic disagrees with its shape is refused rather than
    reinterpreted -- which makes this one number worth naming.
    """
    name, numel = X_REF
    x_ref = signature["inputs"][1]
    assert x_ref["name"] == name
    assert x_ref["numel"] == numel
    assert x_ref["shape"] == [50, 48]


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
    """The measurement that argues for the checks.

    Not an endorsement of running without them: this is what "off" costs,
    and for two of these scenarios the run continues with a corrupted arena
    and reports nothing at all.
    """
    line = check(run, build, artifacts.trajopt, scenario, debug=False)
    assert line.startswith("NO-THROW"), line


def test_a_nonfinite_output_is_caught(run, build, plugin, artifacts):
    """Solving a singular system, on the artifact that can pose one.

    The inputs are finite and the arithmetic is not, so this is the case
    that ``check_values`` has to catch on the way *out*: an input audit
    alone would pass it.
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

    Two threads colliding inside one ``call()`` cannot be forced: the
    intruder can arrive between two of the holder's calls instead of during
    one.  The binary reports how many times it tried, and gating on the
    guard having fired would make this flaky by construction -- so on a host
    nobody has declared tuned and quiet it is an expected failure instead of
    a real one.
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
