"""``bench``: does the fast path compute the same thing as the reference?

The sweep runs two passes, forwards then backwards, so every case has a
different predecessor.  The steady-state path reuses one set of zero-copy
input buffers for the life of a ``Function``, and a buffer left stale by the
previous call gives the right answer for exactly one ordering.

Tolerances come from the fixture manifest, not from a constant here: they are
what ``jax2exec.reference`` froze the cases with.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import helpers
import pytest

#: ``  pass 0 case 3: max rel err ...`` -- one line per case per pass.
CASE_LINE = re.compile(r"^\s*pass (\d+) case (\d+):", re.MULTILINE)

#: The sweep's verdict, printed only when every case agreed.
AGREEMENT = "all cases agree with the reference"


def sweep_argv(build, directory, *extra):
    """The correctness sweep over the artifact set in @p directory."""
    return [
        build.bin("bench"),
        "--all-cases",
        "--fixture",
        "trajopt",
        "--assets-dir",
        directory,
        "--artifacts-dir",
        directory,
        *extra,
    ]


@pytest.fixture
def tolerance(artifacts, load_json):
    """The loosest per-dtype tolerance the manifest declares.

    ``max_rel_err`` is a maximum over float32 and float64 outputs together,
    so the looser of the two is the only honest comparison; the per-dtype
    gate is ``bench``'s own ``correctness.ok``, asserted separately.
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
        return run(sweep_argv(build, artifacts.dir))

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
    """Two passes over every case, forwards then backwards -- counted, since
    a sweep that ran one pass would still print that everything agreed.
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
    """Each printed comparison, not only the summary line.  The NaN count is
    the one a maximum-of-relative-errors check cannot see: a NaN difference
    is not a large error but an absent one.
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
    """Whether the plugin honours ``--async`` is a latency question; whether
    the answers are still right is not negotiable either way.
    """
    result = run(sweep_argv(build, artifacts.dir, "--async"))
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


# ------------------------------------------------------------ negative controls
#
# Everything above asserts that the comparison REPORTED agreement, which is
# what a comparator that never compares would print too.  These damage a
# reference case and require the sweep to notice.


def _corrupt_case_output(base: Path, *, nan: bool) -> None:
    """Damage the first case's output region in a copied fixture.

    The `.bin` layout is every input then every output, in call order, with no
    header, so the outputs start after the summed input bytes.
    """
    manifest = json.loads((base.parent / f"{base.name}_cases.json").read_text())
    itemsize = {
        "float64": 8,
        "float32": 4,
        "int32": 4,
        "int64": 8,
        "bool": 1,
        "int8": 1,
        "int16": 2,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
    }
    offset = sum(
        math.prod(spec["shape"]) * itemsize[spec["dtype"]]
        for spec in manifest["inputs"]
    )

    case = base.parent / manifest["cases"][0]
    raw = bytearray(case.read_bytes())
    assert manifest["outputs"][0]["dtype"] == "float64", "adjust for the dtype"
    poison = struct.pack("<d", float("nan") if nan else 1.0e9)
    raw[offset : offset + 8] = poison
    case.write_bytes(bytes(raw))


@pytest.mark.parametrize("nan", [False, True], ids=["wrong-value", "nan"])
def test_a_damaged_reference_case_is_caught(
    build, plugin, artifacts, tmp_artifacts, run, nan
):
    """A corrupted output must fail the sweep, not pass it quietly."""
    base = tmp_artifacts.copy("trajopt")
    _corrupt_case_output(base, nan=nan)

    result = run(sweep_argv(build, base.parent), check=False)

    assert result.returncode != 0, (
        "bench reported agreement against a case whose output was corrupted; "
        "the comparison is not looking at the data"
    )
    assert "FAIL" in result.stdout or "disagree" in result.stdout.lower()
    if nan:
        assert "nan" in result.stdout.lower()


def test_a_fixture_wider_than_the_artifact_is_refused(
    build, plugin, artifacts, tmp_artifacts, run
):
    """A manifest declaring a larger output than the artifact produces.

    The comparison reads `numel` elements straight out of the arenas the
    `Function` owns, so an overstated output would read past the end of one.
    `--assets-dir` and `--artifacts-dir` are separate options, so pairing a
    manifest with a different export takes one wrong flag.
    """
    base = tmp_artifacts.copy("trajopt")
    manifest_path = base.parent / f"{base.name}_cases.json"
    manifest = json.loads(manifest_path.read_text())
    first = manifest["outputs"][0]
    assert first["shape"][0] == 50, "the fixture changed; update this test"
    first["shape"] = [60, first["shape"][1]]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Pad the cases so the total-length check cannot catch it first.
    for case in sorted(base.parent.glob(f"{base.name}_case*.bin")):
        case.write_bytes(case.read_bytes() + b"\0" * (10 * 6 * 8))

    result = run(sweep_argv(build, base.parent), check=False)

    assert result.returncode != 0
    assert "different exports" in (result.stdout + result.stderr), (
        "the mismatch must be named, not read past"
    )
