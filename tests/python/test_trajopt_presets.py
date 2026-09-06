"""The ``small`` preset of the 02_trajopt workload still traces.

``default`` is exercised by every other test in the suite -- it is the
workload the benchmark, example 03 and the shape-pinning tests all describe.
``small`` is exported by nobody, so without this it would break silently and
be discovered by whoever reached for it on a slow machine.

Lowering rather than compiling: what can go wrong at a different size is the
tracing -- a shape that no longer broadcasts, an actuator row index that
collides, a horizon a warm start cannot shift -- and lowering catches all of
it without paying for a second XLA compilation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax", reason="tracing the workload runs JAX")

# Process-global, and before anything traces: the model is float64 throughout.
jax.config.update("jax_enable_x64", True)

#: ``examples/`` is not a package, so the export script is loaded by path.
SCRIPT = Path(__file__).resolve().parents[2] / "examples/02_trajopt/export.py"


@pytest.fixture(scope="module")
def export_module():
    """``examples/02_trajopt/export.py``, loaded under a name of its own.

    By path rather than by ``import export``: three scripts in this repository
    are called ``export.py``, and whichever one reached ``sys.modules`` first
    would answer for all of them.
    """
    pytest.importorskip("jax2exec", reason="the script imports the exporter")
    spec = importlib.util.spec_from_file_location("trajopt_export", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: ``dataclasses`` resolves the string
    # annotations ``from __future__ import annotations`` leaves behind through
    # ``sys.modules``, and a module that is not there cannot define ``Preset``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small(export_module):
    return export_module.Model(export_module.PRESETS["small"])


def test_the_default_preset_is_the_one_everything_else_pins(export_module):
    """24 masses, 6 actuators, a 50-step horizon, 5 iterations.

    Spelled out here rather than read from the preset, because the point is
    that these four numbers cannot move: ``x_ref`` is ``f64[50,48]`` in three
    other tests, and every recorded latency figure describes this size.
    """
    preset = export_module.PRESETS["default"]
    assert (preset.nq, preset.nu, preset.h, preset.n_iters) == (24, 6, 50, 5)


def test_small_specs_follow_its_preset(export_module, small):
    """The declared shapes are derived from the preset, not hard-coded."""
    preset = export_module.PRESETS["small"]
    shapes = [tuple(spec.shape) for spec in small.specs()]
    assert shapes == [
        (2 * preset.nq,),  # x0
        (preset.h, 2 * preset.nq),  # x_ref
        (export_module.NP,),  # params
        (preset.h, preset.nu),  # u_warm
        (4,),  # weights
        (),  # use_terminal
        (),  # step
    ]


def test_small_lowers(small):
    """It traces, and the lowering reports the outputs the C++ side indexes."""
    lowered = jax.jit(small.solve).lower(*small.specs())
    assert lowered.as_text()

    outputs = jax.eval_shape(small.solve, *small.specs())
    assert len(outputs) == 8
    assert outputs.cost_history.shape == (small.n_iters,)
    assert outputs.u_opt.shape == (small.h, small.nu)


def test_every_actuator_drives_a_different_mass(small):
    """``b_act`` has one 1 per column, on distinct rows.

    ``rint(linspace(0, nq - 1, nu))`` can collide when ``nu`` approaches
    ``nq``, and a duplicated row would silently give two actuators the same
    authority -- a weaker problem, not a broken one, which is the kind of
    change that goes unnoticed.
    """
    rows = [int(column.argmax()) for column in small.b_act.T]
    assert small.b_act.sum() == small.nu
    assert len(set(rows)) == small.nu
