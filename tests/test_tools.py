"""``tools/``: the host audit, the plugin probe and the configuration sweep.

``rt_check.sh`` exits non-zero when anything about the host is worth fixing,
which is its whole purpose -- and on an ordinary developer machine (powersave
governor, no isolcpus, SMT on) plenty is.  Its exit code is therefore
recorded and not asserted.  What is asserted is that it ran and printed every
section, because those sections are the provenance a latency number has to be
quoted with.

``plugin_probe`` answers the one question that cannot be answered by
experiment: this XLA version validates create-option names and fails client
creation on anything it does not recognise, so "try it and see" is not
available.  A plugin built from this project's fork advertises
``supports_synchronous_execution``; a stock one does not, and the runtime
then reports ``SyncMode::Accepted`` rather than ``Inline``.  Against a stock
plugin the inline assertion skips: it would be a statement about which
plugin is installed, not about this code.
"""

from __future__ import annotations

import csv
import re

import pytest

#: The greppable sections ``rt_check.sh`` prints.
RT_CHECK_SECTIONS = [
    "cpu",
    "kernel",
    "isolation",
    "memory",
    "process limits",
    "container",
]

#: The attribute the patched CPU plugin carries and a stock one does not.
SYNC_ATTRIBUTE = "attribute[supports_synchronous_execution]"

#: The configurations ``run_matrix.sh`` sweeps; one CSV row per round each.
MATRIX_CONFIGS = [
    "sync_t1",
    "sync_t2",
    "sync_t4",
    "sync_tdefault",
    "async_t1",
    "async_t4",
    "async_tdefault",
    "sync_t1_rt",
]

#: Iterations and rounds for the sweep.  Far too small to mean anything as a
#: measurement, which is right: what is under test is that every
#: configuration ran and wrote a well-formed row, not what the numbers were.
MATRIX_ITERATIONS = 20
MATRIX_ROUNDS = 1


def tool(repo, relative):
    """An executable script in the checkout, or a skip naming it."""
    path = repo.root / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not in this checkout")
    return path


@pytest.fixture
def rt_check(run, repo):
    """The host audit.  Non-zero exit is expected on an untuned machine."""
    return run([tool(repo, "tools/rt_check.sh")], check=False, timeout=120.0)


def test_rt_check_runs(rt_check):
    """It produced its report.

    The exit code is deliberately not asserted: a non-zero one means the
    host has something worth fixing, which is a true statement about the
    machine rather than a failure of the script.
    """
    assert "=== real-time host audit ===" in rt_check.stdout


@pytest.mark.parametrize("section", RT_CHECK_SECTIONS)
def test_rt_check_prints_every_section(rt_check, section):
    lines = [line.rstrip() for line in rt_check.stdout.splitlines()]
    assert section in lines, f"no '{section}' section in the audit"


def test_rt_check_counts_its_findings(rt_check):
    """The closing tally, which is what a script downstream reads."""
    assert re.search(r"=== \d+ ok, \d+ worth fixing ===", rt_check.stdout)


@pytest.fixture
def probe(run, build, plugin, parse_kv_lines):
    """``plugin_probe`` against the configured plugin."""
    result = run([build.bin("plugin_probe")])
    return result, parse_kv_lines(result.stdout)


def test_probe_reports_the_api_version(probe):
    _, reported = probe
    assert re.fullmatch(r"\d+\.\d+", reported["api_version"])
    assert reported["platform_name"]
    assert reported["vendored_api_minor"]


def test_probe_reports_the_sync_mode(probe):
    _, reported = probe
    assert reported["sync_mode"] in {
        "inline",
        "accepted",
        "rejected",
        "async",
    }


def test_probe_lists_the_attributes(probe):
    """The attribute list is the plugin's own answer about what it accepts.

    Counted against ``num_attributes`` rather than merely non-empty: a probe
    that printed a count it did not then list would be reporting a surface
    nobody can see.
    """
    _, reported = probe
    listed = [key for key in reported if key.startswith("attribute[")]
    assert listed
    assert len(listed) == int(reported["num_attributes"])


def test_the_forked_plugin_executes_inline(probe):
    """Against this project's fork, execution really is inline.

    Skipped against a stock plugin, which advertises no such option.
    """
    _, reported = probe
    if SYNC_ATTRIBUTE not in reported:
        pytest.skip(
            "this is a stock PJRT CPU plugin (no "
            "supports_synchronous_execution attribute); `make plugin` "
            "builds the fork that exercises the inline path"
        )
    assert reported[SYNC_ATTRIBUTE] == "1"
    assert reported["advertises_synchronous_execution"] == "1"
    assert reported["synchronous_supported"] == "1"
    assert reported["sync_mode"] == "inline"


@pytest.mark.slow
def test_run_matrix_writes_a_row_per_configuration(
    run, repo, build, plugin, tmp_path
):
    """A tiny sweep: one CSV row per configuration, plus the header.

    The sweep interleaves configurations in short rounds rather than running
    each to completion, because a sequential A/B drifts by a few percent as
    the CPU heats up -- the same order as the differences being measured.
    That structure is what this test checks is intact; the numbers it
    produces at twenty iterations are not a measurement of anything.
    """
    sweep = tool(repo, "tools/run_matrix.sh")
    csv_path = tmp_path / "matrix.csv"
    result = run(
        [
            sweep,
            "trajopt",
            str(MATRIX_ITERATIONS),
            str(MATRIX_ROUNDS),
            csv_path,
        ],
        env={"BENCH": str(build.bin("bench"))},
        timeout=900.0,
    )
    assert result.returncode == 0
    assert csv_path.is_file(), result.stdout

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(MATRIX_CONFIGS) * MATRIX_ROUNDS

    assert {row["label"] for row in rows} == {
        f"{name}_r{round_}"
        for name in MATRIX_CONFIGS
        for round_ in range(1, MATRIX_ROUNDS + 1)
    }
    for row in rows:
        assert int(row["n"]) == MATRIX_ITERATIONS
        assert int(row["dropped"]) == 0
        assert float(row["p50_us"]) > 0
        assert row["config"], row["label"]
