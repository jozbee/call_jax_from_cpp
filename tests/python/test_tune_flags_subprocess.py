"""One real campaign, end to end, on a matmul small enough to be cheap.

Two rounds of twenty calls decide nothing about any flag -- that is noise,
and the assertions here say nothing about a verdict.  What it proves is that
the children spawn, accept their flags, write their timings, and reduce to a
table and a file that reads back.
"""

from __future__ import annotations

import platform

import numpy as np
import pytest
from jax2exec.tune import Candidate, load_result, tune_flags

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="the driver reads /proc and /sys, which only Linux has",
)


@pytest.mark.timeout(180)
def test_a_real_campaign_produces_a_readable_result(tmp_path):
    a = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    b = np.random.default_rng(1).random((64, 64), dtype=np.float32)

    # docs: begin tune-call
    vector_width = Candidate("vec-128", "--xla_cpu_prefer_vector_width=128")
    control = Candidate(
        "no-fast-math",
        "--xla_cpu_enable_fast_math=false",
        role="control",
    )
    result = tune_flags(
        "jax.numpy:matmul",
        (a, b),
        candidates=[vector_width, control],
        rounds=2,
        reps=20,
        warmup=2,
        cpu=None,
        burn_in=False,
        idle_gap_s=0.0,
        smoke=False,
        timeout_s=120,
    )
    # docs: end tune-call

    labels = [arm.label for arm in result.arms]
    assert labels == ["baseline", "vec-128", "no-fast-math", "baseline-aa"]
    assert all(arm.status == "ok" for arm in result.arms)

    # The band is as wide as this machine was noisy; only that it exists
    # and brackets 1 is a property of the protocol rather than of the host.
    assert result.band is not None
    low, high = result.band
    assert low <= 1 <= high

    for arm in result.arms:
        assert sorted(arm.ratio) == [1, 2]
        assert arm.max_abs_dev is not None
        assert np.isfinite(arm.max_abs_dev)

    text = result.table()
    assert all(label in text for label in labels)

    # Two rounds of twenty calls is noise, so the control need not read
    # slower and the result may say so; only the shape of that is asserted.
    assert result.invalid_reason is None or isinstance(
        result.invalid_reason, str
    )

    path = result.save(tmp_path / "result.json")
    assert load_result(path) == result
