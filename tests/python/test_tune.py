"""The parts of the flag tuner that can be decided without measuring.

A verdict, a noise band, a flag merge and a target name are all pure
functions, and they are where a protocol error would hide: a campaign that
calls one round "faster" or compares two arms that set the same switch is
wrong in a way no amount of wall clock finds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The underscored names are protocol rather than API -- a child's argv and
# the smoke probe -- and they are exactly what a test has to pin.
from jax2exec.tune import (
    CATALOG,
    MIN_VERDICT_ROUNDS,
    RESULT_SCHEMA,
    ArmResult,
    Candidate,
    TuneResult,
    _child_argv,
    _smoke_one,
    _target_name,
    aa_band,
    combination_arm,
    combinations,
    drop_flag,
    load_result,
    merge_flags,
    verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def module_level_target(x):
    """A target the child could import, used by the naming tests."""
    return x


########################
#    1. THE VERDICT    #
########################


def test_verdict_reads_every_direction():
    """Outside the band in every round, either way, or inside it."""
    band = (0.98, 1.02)
    assert verdict([0.90, 0.91, 0.89], *band, 3) == "faster"
    assert verdict([1.10, 1.11, 1.09], *band, 3) == "slower"
    assert verdict([0.90, 1.10, 0.99], *band, 3) == "within A/A"


def test_one_surviving_ratio_never_says_faster():
    """``all()`` over one ratio is trivially true, so it must not count."""
    assert verdict([0.5], 0.98, 1.02, 5) == "insufficient rounds"
    assert MIN_VERDICT_ROUNDS >= 2


########################
#     2. THE BAND      #
########################


def test_band_is_symmetric_about_one():
    """A ratio and its reciprocal imply the same width."""
    assert aa_band([1.25]) == pytest.approx(aa_band([1 / 1.25]))
    low, high = aa_band([1.01, 0.99])
    assert low == pytest.approx(2 - high)


def test_band_without_a_positive_ratio_is_an_error():
    """A list of zeros is a missing ruler, not a band of width zero."""
    with pytest.raises(ValueError, match="no positive"):
        aa_band([0.0, 0.0])


########################
#    3. FLAG MERGING   #
########################


def test_later_group_wins_per_flag_name():
    assert merge_flags(
        ("--xla_cpu_prefer_vector_width=256",),
        ("--xla_cpu_prefer_vector_width=128",),
    ) == ("--xla_cpu_prefer_vector_width=128",)


def test_two_inner_extra_option_keys_both_survive():
    """That flag is a map: replacing it wholesale would drop a key."""
    merged = merge_flags(
        ("--xla_backend_extra_options=a=1",),
        ("--xla_backend_extra_options=b=2",),
    )
    assert len(merged) == 1
    assert "a=1" in merged[0] and "b=2" in merged[0]


def test_a_bare_token_passes_through():
    assert merge_flags(("--xla_cpu_enable_fast_math",)) == (
        "--xla_cpu_enable_fast_math",
    )


def test_drop_flag_removes_by_key():
    flags = ("--xla_cpu_max_isa=AVX2", "--xla_cpu_use_xnnpack=false")
    assert drop_flag(flags, "--xla_cpu_max_isa") == (
        "--xla_cpu_use_xnnpack=false",
    )


########################
#   4. APPLICABILITY   #
########################


def test_an_avx512_candidate_is_dropped_on_an_avx2_host():
    from jax2exec._flags import applicable

    entry = Candidate("wide", "--flag=1", min_isa="x86-64-v4")
    kept, dropped = applicable(
        [entry], isa_level="x86-64-v3", jax_version="0.11.0"
    )
    assert kept == []
    assert dropped[0][0] is entry
    assert "x86-64-v4" in dropped[0][1] and "x86-64-v3" in dropped[0][1]
    assert entry.requires == "host at x86-64-v4 or better"


def test_since_is_inclusive_and_until_is_exclusive():
    from jax2exec._flags import applicable

    new = Candidate("new", "--flag=1", since="0.11.1")
    old = Candidate("old", "--flag=2", until="0.11.1")
    isa = "x86-64-v3"

    kept, dropped = applicable([new], isa_level=isa, jax_version="0.11.0")
    assert kept == [] and "0.11.1" in dropped[0][1]
    kept, _ = applicable([new], isa_level=isa, jax_version="0.11.1")
    assert kept == [new]

    kept, _ = applicable([old], isa_level=isa, jax_version="0.11.0")
    assert kept == [old]
    kept, dropped = applicable([old], isa_level=isa, jax_version="0.11.1")
    assert kept == [] and "gone at jax 0.11.1" in dropped[0][1]


########################
#  5. TARGET NAMING    #
########################


def test_a_module_level_function_and_a_string_are_both_accepted():
    assert _target_name(module_level_target).endswith(":module_level_target")
    assert _target_name("jax.numpy:matmul") == "jax.numpy:matmul"


def test_a_lambda_cannot_be_named():
    with pytest.raises(ValueError, match="lambda"):
        _target_name(lambda x: x)


def test_a_closure_cannot_be_named():
    def inner(x):
        return x

    with pytest.raises(ValueError, match="inside a function"):
        _target_name(inner)


def test_main_cannot_be_named(monkeypatch):
    """``__main__`` in the child is the child's own module, not the caller's."""
    monkeypatch.setattr(module_level_target, "__module__", "__main__")
    with pytest.raises(ValueError, match="__main__"):
        _target_name(module_level_target)


########################
#   6. COMBINATIONS    #
########################


def test_every_non_conflicting_subset_of_three_winners():
    winners = ["vector-width-128", "vector-width-64", "no-xnnpack"]
    sets = combinations(winners)
    # 7 non-empty subsets, less the two that set the vector width twice
    assert len(sets) == 5
    assert ("vector-width-128", "vector-width-64") not in sets
    assert tuple(winners) not in sets
    assert ("vector-width-128", "no-xnnpack") in sets


def test_greedy_beyond_the_exhaustive_size():
    winners = [
        "vector-width-128",
        "no-xnnpack",
        "opt-level-2",
        "region-copy",
        "no-slp",
    ]
    assert combinations(winners) == [(label,) for label in winners]


def test_a_combination_arm_carries_its_flags_in_catalog_order():
    arm = combination_arm(["no-xnnpack", "vector-width-128"])
    assert arm.label == "no-xnnpack+vector-width-128"
    assert arm.flags == (
        "--xla_cpu_prefer_vector_width=128",
        "--xla_cpu_use_xnnpack=false",
    )
    assert arm.members == ("no-xnnpack", "vector-width-128")


########################
#  7. SAVE AND LOAD    #
########################


def _arm_result(label: str, **overrides) -> ArmResult:
    """One plausible :class:`ArmResult`, for the rendering and IO tests."""
    fields = {
        "label": label,
        "flags": (),
        "members": (),
        "role": "candidate",
        "numerics": "exact",
        "aa_of": None,
        "status": "ok",
        "median_ms": {1: 1.0, 2: 1.0},
        "p95_ms": {1: 1.2, 2: 1.2},
        "p99_ms": {1: 1.4, 2: 1.4},
        "max_ms": {1: 2.0, 2: 2.0},
        "compile_s": {1: 0.5, 2: 0.5},
        "ratio": {1: 1.0, 2: 1.0},
        "median_ratio": 1.0,
        "verdict": "within A/A",
        "span": (1.0, 1.0),
        "rounds_ok": 2,
        "max_abs_dev": 0.0,
        "max_rel_dev": 0.0,
        "hygiene": (),
        "note": "",
    }
    fields.update(overrides)
    return ArmResult(**fields)  # type: ignore[arg-type]


def _result(*arms: ArmResult) -> TuneResult:
    """A :class:`TuneResult` around some arms."""
    return TuneResult(
        schema=RESULT_SCHEMA,
        host={"cpu_model": "not recorded"},
        jax="0.11.0",
        jaxlib="0.11.0",
        reference="baseline",
        aa="baseline-aa",
        band=(0.98, 1.02),
        rounds=2,
        reps=200,
        warmup=3,
        min_gain=0.03,
        base_flags=(),
        target="jax.numpy:matmul",
        started_at="2026-01-01T00:00:00Z",
        wall_s=12.5,
        arms=arms,
        dropped=(("new-xtile", "needs jax >= 0.11.1"),),
        winners=(),
        valid=True,
        invalid_reason=None,
        records_dir=None,
    )


def test_save_and_load_round_trip(tmp_path):
    result = _result(
        _arm_result("baseline", role="baseline", verdict="reference"),
        _arm_result("baseline-aa", aa_of="baseline"),
    )
    path = result.save(tmp_path / "result.json")
    assert load_result(path) == result


def test_a_newer_schema_is_refused_by_number(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema": RESULT_SCHEMA + 1, "arms": []}))
    with pytest.raises(ValueError, match=f"schema {RESULT_SCHEMA + 1}"):
        load_result(path)


########################
#    8. THE TABLE      #
########################


def test_the_table_shows_the_tail_the_band_and_the_caveat():
    extra = _arm_result(
        "no-unroll",
        flags=("--xla_backend_extra_options=xla_cpu_disable_loop_unrolling=1",),
        role="control",
        verdict="slower",
    )
    text = _result(
        _arm_result("baseline", role="baseline", verdict="reference"),
        extra,
        _arm_result("baseline-aa", aa_of="baseline"),
    ).table()
    assert "p99 ms" in text and "max ms" in text
    assert "A/A band: 0.9800 - 1.0200" in text
    assert "†" in text
    assert "no-unroll" in text


def test_a_tail_that_disagrees_with_the_median_is_marked():
    """The figure of merit is the tail, so a p50 win with a p99 loss says so."""
    text = _result(
        _arm_result("baseline", role="baseline", verdict="reference"),
        _arm_result(
            "loud",
            median_ms={1: 0.5, 2: 0.5},
            p99_ms={1: 2.8, 2: 2.8},
            ratio={1: 0.5, 2: 0.5},
            median_ratio=0.5,
            span=(0.5, 0.5),
            verdict="faster",
        ),
        _arm_result("baseline-aa", aa_of="baseline"),
    ).table()
    assert "tail^" in text


########################
#   9. THE CHILD SIDE  #
########################


def test_importing_the_tuner_does_not_import_jax():
    """A driver that imported JAX would have a backend of its own already."""
    code = (
        "import sys, jax2exec.tune; "
        "print('jax' in sys.modules or 'jaxlib' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert proc.stdout.strip() == "False"


def test_a_pinned_child_is_spelled_with_taskset(tmp_path):
    argv = _child_argv("/usr/bin/python3", 3, tmp_path / "spec.json")
    assert argv[:3] == ["taskset", "-c", "3"]
    assert argv[3:6] == ["/usr/bin/python3", "-m", "jax2exec._tune_child"]
    assert (
        _child_argv("/usr/bin/python3", None, tmp_path / "spec.json")[0]
        == "/usr/bin/python3"
    )


def test_the_smoke_probe_pins_and_passes_the_flag(monkeypatch):
    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["flags"] = kwargs["env"]["XLA_FLAGS"]
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    status, _ = _smoke_one(CATALOG[0], sys.executable, 3, (), 30.0, None)
    assert status == "ok"
    assert seen["argv"][:3] == ["taskset", "-c", "3"]
    assert seen["flags"] == CATALOG[0].flag
