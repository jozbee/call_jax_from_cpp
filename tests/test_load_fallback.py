"""What the loader does when an artifact is wrong, stale or half-present.

``tests/cpp/test_load_errors.cpp`` loads one artifact under one awkward
condition and prints a single line -- ``LOADED kind=... detail=...`` or
``THREW: <what()>`` -- so what is asserted here is the *message*: a load
that stops is only useful if it names the file and says what was wrong.

The distinction defended is fallback against failure.  Falling back from a
stale ``.binpb`` to compiling the ``.mlirbc`` is correct but costs seconds
where the AOT path costs milliseconds, so it must be *reported*.  A sidecar
that disagrees with its executable is a buffer overrun waiting to surface as
corrupted output, so it is refused.
"""

from __future__ import annotations

import pytest

#: What the binary prints when the load succeeded, and when it did not.
LOADED = "LOADED "
THREW = "THREW: "


def load(run, build, base, scenario):
    """Load @p base under @p scenario and return the single reported line."""
    result = run([build.bin("test_load_errors"), base, scenario])
    assert result.returncode == 0, result.stderr
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith((LOADED, THREW))
    ]
    assert len(lines) == 1, result.stdout
    return lines[0]


def test_an_intact_artifact_deserializes(run, build, plugin, artifacts):
    """The baseline: without it, a loader that refuses everything would
    satisfy every assertion below.
    """
    line = load(run, build, artifacts.trajopt, "ok")
    assert line.startswith(LOADED)
    assert "kind=deserialized" in line


def test_a_sidecar_missing_an_output_is_refused(
    run, build, plugin, tmp_artifacts
):
    """The count mismatch names both sides.  Unchecked, the loader would
    allocate one arena too few and the executable would write past the end
    of the last one.
    """
    base = tmp_artifacts.copy("trajopt")
    tmp_artifacts.drop_output(base)

    line = load(run, build, base, "stale")
    assert line.startswith(THREW)
    assert "declares 7 outputs" in line
    assert "produces 8" in line


def test_a_sidecar_with_the_wrong_shape_is_refused(
    run, build, plugin, tmp_artifacts
):
    """The shape mismatch names the declared shape and the produced one:
    which is wrong decides whether to re-export or to fix the function.
    """
    base = tmp_artifacts.copy("trajopt")

    def shrink(sidecar):
        output = sidecar["outputs"][0]
        output["shape"] = [49, 6]
        output["numel"] = 49 * 6
        output["nbytes"] = 49 * 6 * 8

    tmp_artifacts.edit_sidecar(base, shrink)

    line = load(run, build, base, "stale")
    assert line.startswith(THREW)
    assert "declares output 0 as float64[49,6]" in line
    assert "produces float64[50,6]" in line


def test_a_truncated_executable_falls_back_and_says_why(
    run, build, plugin, tmp_artifacts
):
    """A stale ``.binpb`` compiles the ``.mlirbc`` instead, out loud: a load
    that takes seconds instead of milliseconds and does not explain itself
    is how a deployment discovers from a latency graph that its AOT path
    stopped working.
    """
    base = tmp_artifacts.copy("trajopt")
    tmp_artifacts.truncate_executable(base)

    line = load(run, build, base, "fallback")
    assert line.startswith(LOADED)
    assert "kind=compiled" in line
    assert "trajopt.mlirbc" in line
    assert "could not be deserialized" in line, line


def test_a_truncated_executable_with_no_bytecode_is_fatal(
    run, build, plugin, tmp_artifacts
):
    """Both routes gone: the message names both files, since either reason
    alone sends the reader to the wrong one.
    """
    base = tmp_artifacts.copy("trajopt")
    tmp_artifacts.truncate_executable(base)
    tmp_artifacts.remove(base, ".mlirbc")

    line = load(run, build, base, "both")
    assert line.startswith(THREW)
    assert "could not load" in line
    assert "could not compile" in line
    assert "trajopt.binpb" in line
    assert "trajopt.mlirbc" in line


def test_compile_only_ignores_the_binary_beside_it(
    run, build, plugin, artifacts
):
    """``LoadPolicy::CompileOnly`` compiles even with a good ``.binpb`` there,
    and the detail names the policy, which is how a slow load is told apart
    from a fallback nobody asked for.
    """
    line = load(run, build, artifacts.trajopt, "compile-only")
    assert line.startswith(LOADED)
    assert "kind=compiled" in line
    assert "LoadPolicy::CompileOnly" in line


def test_binary_only_deserializes(run, build, plugin, artifacts):
    """``LoadPolicy::BinaryOnly`` refuses rather than substitutes: a fallback
    that compiles for seconds is not a fallback in a control loop.
    """
    line = load(run, build, artifacts.trajopt, "binary-only")
    assert line.startswith(LOADED)
    assert "kind=deserialized" in line
    assert "LoadPolicy::BinaryOnly" in line


def test_a_missing_artifact_is_refused(run, build, plugin, tmp_path):
    """A base path with nothing under it fails at the sidecar, naming the
    file it tried to open -- the path is given *without* an extension.
    """
    line = load(run, build, tmp_path / "not_an_artifact", "missing")
    assert line.startswith(THREW)
    assert "cannot open sidecar" in line


def test_a_plugin_that_will_not_open_names_dlopen(run, build, artifacts):
    """``dlopen(`` plus the raw ``dlerror``: the plugin is opened ``RTLD_NOW``,
    so a missing ``liblapack`` surfaces here rather than on the first
    ``jnp.linalg`` call.  The scenario points the runtime at the sidecar, a
    file that exists and is not an ELF object, so it cannot succeed by
    finding a real plugin at a made-up path.
    """
    line = load(run, build, artifacts.trajopt, "bad-plugin")
    assert line.startswith(THREW)
    assert "dlopen(" in line


@pytest.mark.slow
def test_the_compiled_path_computes_the_same_thing(
    run, build, plugin, artifacts, tmp_artifacts
):
    """After a fallback, every reference case still agrees: the compiled and
    the deserialized executable are two routes to the same numbers.  The
    reference cases stay in the real artifacts directory; only the executable
    being loaded is the damaged copy.
    """
    base = tmp_artifacts.copy("trajopt")
    tmp_artifacts.truncate_executable(base)

    result = run(
        [
            build.bin("bench"),
            "--all-cases",
            "--fixture",
            "trajopt",
            "--assets-dir",
            artifacts.dir,
            "--artifacts-dir",
            base.parent,
        ],
        timeout=600.0,
    )
    assert result.returncode == 0
    assert "all cases agree with the reference" in result.stdout
    assert "FAIL" not in result.stdout
