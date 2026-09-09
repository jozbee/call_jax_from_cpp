"""``examples/05_ros2_control``: the exported step, and the controller.

Two halves that need very different things.  The first is pure JAX: it loads
``export.py`` by path and iterates ``step`` the way the controller does, so
that a change to the kinematics or to the damping is caught here rather than
by a robot.  Tracking, not convergence, is what it asserts -- resolved-rate
control against a target moving at 0.15 m/s lags by about ``speed / GAIN``
and never catches up, so a tolerance tighter than that would be a test of
arithmetic that is not happening.

The second half needs a ROS 2 workspace, which is why it skips with the
remedy rather than failing: on a machine with no ``colcon`` the thing being
tested cannot be built, and a green result there would mean nothing.  It runs
in the ``ros2`` compose service, and CI runs it there.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax", reason="iterating the step runs JAX")

# Process-global, and before anything traces: the export is float64 throughout.
jax.config.update("jax_enable_x64", True)

#: ``examples/`` is not a package, so the export script is loaded by path.
SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "examples/05_ros2_control/export.py"
)

#: The script `run.sh` builds and launches, from the repository root.
RUN_SH = "examples/05_ros2_control/run.sh"

#: Seconds of launch.  Long enough for the spawner to configure and activate
#: both controllers -- which took under a second in the container -- and for
#: /joint_states to publish more than once, which is what the last assertion
#: compares.
LAUNCH_SECONDS = 8

#: How far the end-effector may sit from the target after the run.  The lag a
#: proportional law leaves against a target moving at ``0.3 * 0.5`` m/s is
#: about ``0.15 / GAIN``; 0.05 leaves room above it without admitting a
#: controller that is not tracking at all.
TRACKING_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def export_module():
    """``examples/05_ros2_control/export.py``, loaded under a name of its own.

    By path rather than by ``import export``: four scripts in this repository
    are called ``export.py``, and whichever one reached ``sys.modules`` first
    would answer for all of them.
    """
    pytest.importorskip("jax2exec", reason="the script imports the exporter")
    spec = importlib.util.spec_from_file_location("arm_export", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_step_tracks_the_circle(export_module):
    """Iterated at the controller's period, the end-effector reaches the target.

    The initial pose is the one ``urdf/arm.urdf`` gives the mock hardware, so
    this is the trajectory the container run walks along.
    """
    import jax.numpy as jnp

    step = jax.jit(export_module.step)
    q = jnp.array([0.5, 1.0])
    dt, t = 0.002, 0.0
    for _ in range(4000):
        q = step(q, t, dt)
        t += dt

    position = export_module.fk(q)
    assert bool(jnp.all(jnp.isfinite(q))), f"q went non-finite: {q}"
    error = float(jnp.linalg.norm(position - export_module.target(t)))
    assert error < TRACKING_TOLERANCE, (
        f"end-effector is {error} from the target after {t} s; the lag a gain "
        f"of {export_module.GAIN} leaves is about 0.03"
    )


def test_the_sidecar_names_the_three_inputs(artifacts):
    """``q``, ``t`` and ``dt``, in that order, and one output.

    The controller resolves those three arenas by index in ``on_configure``,
    so a re-export that reorders the arguments has to be seen here rather
    than as an arm that moves strangely.
    """
    sidecar = json.loads(Path(f"{artifacts.arm}.json").read_text())
    assert [entry["name"] for entry in sidecar["inputs"]] == ["q", "t", "dt"]
    assert len(sidecar["outputs"]) == 1
    assert sidecar["inputs"][0]["shape"] == [2]


def test_the_controller_runs_under_ros2_control(repo, artifacts):
    """Build with colcon, launch, and read what the run says about itself.

    Four claims, and each one fails differently: the two hardening lines say
    the controller applied the process's half of the bargain, ``active`` says
    ``controller_manager`` accepted the plugin and configured it, and the two
    joint vectors say the loop actually ran -- a controller that loaded and
    then wrote nothing prints the first three and fails the fourth.
    """
    for tool in ("colcon", "ros2"):
        if shutil.which(tool) is None:
            pytest.skip(
                f"no {tool} on PATH; the ROS 2 workspace is the `ros2` "
                "compose service, which runs the same script directly: "
                "`docker compose -f docker/compose.yml run --rm ros2 "
                f"{RUN_SH} --seconds {LAUNCH_SECONDS}`"
            )

    result = subprocess.run(
        [RUN_SH, "--seconds", str(LAUNCH_SECONDS)],
        cwd=repo.root,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    for line in ("harden_malloc", "lock_memory"):
        assert line in output, f"no {line} status line:\n{output}"
    assert "Configured and activated jax_arm_controller" in output, output

    positions = dict(
        line.split(None, 1)
        for line in output.splitlines()
        if line.startswith(("first ", "last "))
    )
    assert set(positions) == {"first", "last"}, output
    assert "<nothing published>" not in positions.values(), output
    assert positions["first"] != positions["last"], (
        "the joints did not move over the run:\n" + output
    )
