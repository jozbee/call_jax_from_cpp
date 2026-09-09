"""Fixtures shared by the Python and the C++ halves of the suite.

The C++ tests are pytest tests too: each builds a binary, runs it and reads
what it printed.  Everything expensive -- the build, the plugin, the exported
artifacts -- is session scoped and resolved once.  Anything that can be
legitimately absent on a healthy machine skips with the remedy, and nothing
degrades to a no-op assertion: a test that cannot run its subject must not
pass.

The ``rt`` gates run only where the host has been audited as tuned and idle
(:func:`rt_strict`).  On an untuned host the idle state between calls moves a
latency by more than the thresholds (``docs/benchmarks.md``), so a fixed one
would fail a healthy machine.

Environment variables, all optional:

``CJFC_BUILD_DIR``
    Build tree.  Default ``build``.
``CJFC_ARTIFACTS_DIR``
    Exported artifacts.  Default ``artifacts``.
``CJFC_REPORT_DIR``
    JSON reports.  Default ``$CJFC_ARTIFACTS_DIR/reports``.
``CJFC_BUILD_TOOL``
    ``make`` (default), ``cmake``, or ``none``.
``CJFC_SKIP_BUILD``
    ``1`` to take the tree as it stands.
``CJFC_SKIP_EXPORT``
    ``1`` to reuse the artifacts on disk instead of re-exporting.
``CJFC_RT_STRICT``
    ``1``/``0`` forces the real-time gates on or off; outranks ``--rt-strict``.
``PJRT_CPU_PLUGIN``
    The plugin to load, overriding ``$CJFC_BUILD_DIR/plugin``.
``PYTHON``
    Interpreter for the export scripts.  Default: the one running pytest.

Relative ``CJFC_*`` paths resolve against the repository root, not pytest's
working directory, so they mean what the same ``make`` variables mean.
:func:`run`, :func:`load_json` and :func:`parse_kv_lines` exist both as
module-level functions and as fixtures of the same name.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: What ``make plugin`` and ``tools/get_plugin.sh`` write.
PLUGIN_FILE_NAME = "libpjrt_c_api_cpu_plugin.so"

#: Requesting one of these means the test runs a native binary, which is
#: Linux-only here: the published plugin is an ELF shared object.
_BINARY_FIXTURES = frozenset({"build", "plugin", "guard_so"})

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


# --------------------------------------------------------------- environment


def _env_bool(name: str) -> bool | None:
    """Read a tri-state flag: None when unset or empty, else a bool.

    None is distinct from False on purpose: unset falls back to the automatic
    real-time audit, ``0`` skips the gates outright.  Anything else raises
    rather than guesses, so a typo cannot silently skip the suite.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise pytest.UsageError(
        f"${name}={raw!r} is neither {'/'.join(sorted(_TRUE))} nor "
        f"{'/'.join(sorted(_FALSE))}"
    )


def _dir_from_env(name: str, default: Path) -> Path:
    """Resolve a directory override, relative values against the repository."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _export_python() -> list[str]:
    """The interpreter for the export scripts, as argv.

    ``$PYTHON`` is the Makefile's escape hatch.  The default is the
    interpreter running pytest, not ``uv run python``: ``uv`` is not on PATH
    in a non-interactive shell, and ``uv run pytest`` is already inside the
    environment that has JAX.
    """
    raw = os.environ.get("PYTHON", "").strip()
    return shlex.split(raw) if raw else [sys.executable]


def _loadavg() -> tuple[float, float, float]:
    """The 1, 5 and 15 minute load averages, or -1 where unavailable."""
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - not on Linux
        return (-1.0, -1.0, -1.0)
    return (one, five, fifteen)


def _in_container() -> bool:
    """The same three signals ``examples/common/rt_env.hpp`` and
    ``tools/rt_check.sh`` use, so all three agree about one host."""
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or bool(os.environ.get("CJFC_IN_CONTAINER"))
    )


def _parse_cpulist(text: str) -> tuple[int, ...]:
    """Parse a kernel cpulist (``0-3,8``) into cpu numbers."""
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part or part == "(null)":
            continue
        first, _, last = part.partition("-")
        try:
            cpus.update(range(int(first), int(last or first) + 1))
        except ValueError:
            continue
    return tuple(sorted(cpus))


def _read_text(path: str | Path) -> str:
    """Read a sysfs file, returning "" when it is not there."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ------------------------------------------------------------------ options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("call_jax_from_cpp")
    group.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the tests marked slow (long campaigns, whole builds)",
    )
    group.addoption(
        "--rt-strict",
        action="store_true",
        default=False,
        help=(
            "run the tests marked rt, asserting real-time statistics; only "
            "meaningful on a tuned, idle host ($CJFC_RT_STRICT overrides)"
        ),
    )


@functools.cache
def _auto_rt_strict() -> tuple[bool, str]:
    """Decide whether this host may be gated on its tail, and say why.

    The reason goes into the skip message: "skipped" without a cause is
    indistinguishable from a test that was quietly deleted.  Three conditions,
    all necessary: ``tools/rt_check.sh`` finds nothing worth fixing; the
    one-minute load is below 1.0, because a concurrent build does not add
    noise to a tail measurement, it invalidates it
    (``docs/developer/measurement.md``); and this is not a container, which
    can see the host's settings but not set them.  Cached: the audit is a
    subprocess, and the answer does not change within a session.
    """
    if platform.system() != "Linux":
        return False, f"this host runs {platform.system()}, not Linux"

    script = REPO_ROOT / "tools" / "rt_check.sh"
    if not script.is_file():
        return False, f"{script} is missing"
    try:
        audit = subprocess.run(
            [str(script), "--quiet"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"tools/rt_check.sh could not be run ({exc})"
    if audit.returncode != 0:
        lines = [line for line in audit.stdout.splitlines() if line.strip()]
        summary = lines[-1] if lines else "see tools/rt_check.sh"
        return False, f"tools/rt_check.sh reports work to do: {summary}"

    load1 = _loadavg()[0]
    if not load1 < 1.0:
        return False, f"the 1-minute load average is {load1:.2f}, not below 1.0"
    if _in_container():
        return False, "this is a container, which cannot tune its host"
    return True, "tools/rt_check.sh passes, the host is idle and not contained"


def _rt_strict_reason(config: pytest.Config) -> tuple[bool, str]:
    """The strict-mode decision and the reason behind it."""
    forced = _env_bool("CJFC_RT_STRICT")
    if forced is not None:
        return forced, f"$CJFC_RT_STRICT={'1' if forced else '0'}"
    if config.getoption("--rt-strict"):
        return True, "--rt-strict was passed"
    return _auto_rt_strict()


#: How a skipped real-time gate explains itself.
_RT_SKIP = (
    "real-time gates are off: {why}. Turn them on with --rt-strict or "
    "CJFC_RT_STRICT=1, on a tuned and idle host"
)


def rt_strict(config: pytest.Config) -> bool:
    """Whether the real-time gates may assert on this host.

    ``$CJFC_RT_STRICT`` when set, else ``--rt-strict``, else the automatic
    audit.  False on an ordinary developer machine is correct, not a defect:
    numbers measured there describe the machine, not the code.
    """
    return _rt_strict_reason(config)[0]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """The three conditional skips: ``slow``, ``rt``, and native off Linux.

    A test is native when it asks for a fixture that resolves a binary, a
    plugin or the preloadable guard.
    """
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(
            reason="long campaign; pass --runslow (or run `make test-slow`)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if any("rt" in item.keywords for item in items):
        strict, why = _rt_strict_reason(config)
        if not strict:
            skip_rt = pytest.mark.skip(reason=_RT_SKIP.format(why=why))
            for item in items:
                if "rt" in item.keywords:
                    item.add_marker(skip_rt)

    if platform.system() != "Linux":
        skip_native = pytest.mark.skip(
            reason=(
                f"the C++ binaries and the PJRT plugin are Linux-only here; "
                f"this host is {platform.system()}"
            )
        )
        for item in items:
            requested = getattr(item, "fixturenames", ())
            if _BINARY_FIXTURES.intersection(requested):
                item.add_marker(skip_native)


# ------------------------------------------------------------------ helpers


def run(
    cmd: Sequence[str | os.PathLike[str]] | str,
    *,
    cwd: str | os.PathLike[str] = REPO_ROOT,
    env: Mapping[str, str | None] | None = None,
    timeout: float = 300.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capture its output, and report all of it on failure.

    ``cmd`` is a string (split with :mod:`shlex`) or a sequence whose entries
    are stringified, so ``Path`` values need no conversion.  ``env`` is
    **added to** the current environment, not a replacement for it:
    ``$PJRT_CPU_PLUGIN``, ``$LD_PRELOAD`` and whatever ``uv run`` set have to
    survive.  A value of None removes a variable, the only way to guarantee a
    child runs with nothing preloaded.  ``check=False`` returns a non-zero
    exit instead of failing, for tests where the exit is the subject.

    The command line and its output are printed unconditionally.  pytest
    shows them only for a failing test, so a green run costs nothing and a
    red one arrives with the LoadError it printed rather than "exited 1".
    """
    argv = shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]
    child_env = dict(os.environ)
    for key, value in (env or {}).items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = str(value)

    print(f"$ {shlex.join(argv)}")
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        pytest.fail(f"{argv[0]} is not installed or not on PATH ({exc})")
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{shlex.join(argv)} did not finish within {timeout:g}s\n"
            f"{_output_report(exc.stdout, exc.stderr)}"
        )

    if completed.stdout:
        print(
            completed.stdout,
            end="" if completed.stdout.endswith("\n") else "\n",
        )
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
        )

    if check and completed.returncode != 0:
        pytest.fail(
            f"{shlex.join(argv)} exited {completed.returncode}\n"
            f"{_output_report(completed.stdout, completed.stderr)}"
        )
    return completed


def _output_report(stdout: Any, stderr: Any) -> str:
    """Format captured output for a failure message."""

    def decode(stream: Any) -> str:
        if stream is None:
            return ""
        if isinstance(stream, bytes):
            return stream.decode("utf-8", errors="replace")
        return str(stream)

    return f"--- stdout ---\n{decode(stdout)}\n--- stderr ---\n{decode(stderr)}"


def load_json(path: str | os.PathLike[str]) -> Any:
    """Read and parse a JSON file, failing with the path when it will not."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{path} is not valid JSON: {exc}\n{text[:2000]}")


def parse_kv_lines(stdout: str) -> dict[str, Any]:
    """Turn a binary's machine-parseable output into a dictionary.

    Two line shapes are recognised and everything else is ignored.
    ``key=value`` pairs, several per line, land at the top level.
    ``label: key=value ...`` lands as a nested dict under the label, and a
    labelled line with no pair keeps its text verbatim, so that
    ``result["debug_check[non_finite]"]`` is the whole message.  Values stay
    strings -- ``shape=[4,4]`` has no obvious number to become -- and a
    repeated key takes the last value.
    """
    parsed: dict[str, Any] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue

        label: str | None = None
        body = line
        head, sep, tail = line.partition(":")
        if sep and "=" not in head and " " not in head.strip():
            label = head.strip()
            body = tail.strip()

        pairs = {}
        for token in body.split():
            key, eq, value = token.partition("=")
            if eq and key:
                pairs[key] = value

        if label is None:
            parsed.update(pairs)
        elif pairs:
            group = parsed.get(label)
            if not isinstance(group, dict):
                group = {}
                parsed[label] = group
            group.update(pairs)
        else:
            parsed[label] = body
    return parsed


@pytest.fixture(name="run")
def run_fixture() -> Callable[..., subprocess.CompletedProcess[str]]:
    """The :func:`run` helper, as a fixture."""
    return run


@pytest.fixture(name="load_json")
def load_json_fixture() -> Callable[..., Any]:
    """The :func:`load_json` helper, as a fixture."""
    return load_json


@pytest.fixture(name="parse_kv_lines")
def parse_kv_lines_fixture() -> Callable[[str], dict[str, Any]]:
    """The :func:`parse_kv_lines` helper, as a fixture."""
    return parse_kv_lines


# ------------------------------------------------------------------- layout


@dataclasses.dataclass(frozen=True)
class Repo:
    """Where everything lives, after the ``CJFC_*`` overrides are applied."""

    root: Path
    build_dir: Path
    bin_dir: Path
    lib_dir: Path
    plugin_dir: Path
    artifacts_dir: Path
    report_dir: Path
    python_dir: Path

    def bin(self, name: str) -> Path:
        """Path a built binary would have.  Existence is not implied."""
        return self.bin_dir / name

    def artifact(self, name: str) -> Path:
        """Path of an exported artifact, extension and all."""
        return self.artifacts_dir / name

    def __fspath__(self) -> str:
        """The repository root, so ``Path(repo, "versions.env")`` works."""
        return str(self.root)


@pytest.fixture(scope="session")
def repo() -> Repo:
    """The resolved directory layout for this session."""
    build_dir = _dir_from_env("CJFC_BUILD_DIR", REPO_ROOT / "build")
    artifacts_dir = _dir_from_env("CJFC_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
    report_dir = _dir_from_env("CJFC_REPORT_DIR", artifacts_dir / "reports")

    # The binaries write --json <path> without creating the directory.
    report_dir.mkdir(parents=True, exist_ok=True)

    return Repo(
        root=REPO_ROOT,
        build_dir=build_dir,
        bin_dir=build_dir / "bin",
        lib_dir=build_dir / "lib",
        plugin_dir=build_dir / "plugin",
        artifacts_dir=artifacts_dir,
        report_dir=report_dir,
        python_dir=REPO_ROOT / "python",
    )


# -------------------------------------------------------------------- build


@dataclasses.dataclass(frozen=True)
class Build:
    """The built tree, and how it was built."""

    repo: Repo
    tool: str

    def bin(self, name: str) -> Path:
        """A built binary.  Missing after a successful build is a defect in
        the build, not a reason to skip."""
        path = self.repo.bin(name)
        if not path.is_file():
            pytest.fail(
                f"{path} does not exist after the build (tool: {self.tool}); "
                "the build makes every binary this suite drives, so either "
                f"{name} is misspelled or its source is not in this checkout "
                "-- CMake skips a test binary that has no tests/cpp/<name>.cpp"
            )
        return path

    def __fspath__(self) -> str:
        """The build directory, so ``Path(build) / "bin" / name`` works."""
        return str(self.repo.build_dir)


@pytest.fixture(scope="session")
def build(repo: Repo) -> Build:
    """Build the library, the examples, the C++ tests, the guard and bench.

    Honours ``$CJFC_BUILD_TOOL`` and ``$CJFC_SKIP_BUILD=1``.  A build failure
    fails loudly with the compiler's output; skipping would report a green
    run for a tree that does not compile.
    """
    tool = (os.environ.get("CJFC_BUILD_TOOL") or "make").strip().lower()
    if _env_bool("CJFC_SKIP_BUILD"):
        return Build(repo=repo, tool="none ($CJFC_SKIP_BUILD)")
    if tool == "none":
        return Build(repo=repo, tool="none")

    if tool == "make":
        command: list[str] = ["make", "-s"]
        # Only the overrides that differ, so the command in a failure message
        # can be pasted into a shell.  `bench` has no phony target and is
        # named by path, which must be spelled the way $(BIN_DIR) is: make
        # compares target names as strings, and `make /abs/build/bin/bench`
        # against the default BUILD_DIR reports "Nothing to be done".
        bench = Path("build") / "bin" / "bench"
        if repo.build_dir != repo.root / "build":
            command.append(f"BUILD_DIR={repo.build_dir}")
            bench = repo.bin("bench")
        if repo.artifacts_dir != repo.root / "artifacts":
            command.append(f"ARTIFACTS_DIR={repo.artifacts_dir}")
        command += ["lib", "examples", "tests-cpp", "guard", "tools"]
        command.append(str(bench))
        commands = [command]
    elif tool == "cmake":
        commands = [
            [
                "cmake",
                "-S",
                str(repo.root),
                "-B",
                str(repo.build_dir),
                # The plugin is the `plugin` fixture's business; a test
                # session must not start a download.
                "-DPJRT_EXEC_FETCH_PLUGIN=OFF",
            ],
            ["cmake", "--build", str(repo.build_dir), "--parallel"],
        ]
    else:
        raise pytest.UsageError(
            f"$CJFC_BUILD_TOOL={tool!r} is not one of make, cmake, none"
        )

    for command in commands:
        run(command, cwd=repo.root, timeout=1800)
    return Build(repo=repo, tool=tool)


# ------------------------------------------------------------------- plugin


@pytest.fixture(scope="session")
def plugin(repo: Repo) -> Path:
    """Resolve the PJRT CPU plugin, or skip the tests that need one.

    Absent is normal for a fresh clone -- the plugin is a download or an hour
    of bazel -- so this skips with the remedy.  The runtime resolves the same
    two sources in the same order: ``$PJRT_CPU_PLUGIN``, then the path
    compiled in at build time.
    """
    from_env = os.environ.get("PJRT_CPU_PLUGIN", "").strip()
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            pytest.skip(
                f"$PJRT_CPU_PLUGIN={from_env} does not name a file; unset it "
                "to fall back to the built-in path, or point it at a plugin"
            )
        return path

    path = repo.plugin_dir / PLUGIN_FILE_NAME
    if not path.is_file():
        pytest.skip(
            f"no PJRT CPU plugin at {path}; run `make plugin` to download the "
            "prebuilt one, `make plugin-source` to build it from the XLA "
            "fork, or set $PJRT_CPU_PLUGIN"
        )

    # Publish it, so every binary launched here opens this plugin rather than
    # the absolute path baked in at build time: a tree built on the host and
    # tested in the container points at a directory that does not exist there.
    os.environ["PJRT_CPU_PLUGIN"] = str(path)
    return path


# ---------------------------------------------------------------- artifacts


@dataclasses.dataclass(frozen=True)
class Artifacts:
    """The exported artifact set, and the directory holding it."""

    dir: Path

    def base(self, name: str) -> Path:
        """The base path a C++ caller is handed: no extension."""
        return self.dir / name

    @property
    def basic(self) -> Path:
        """Base path of the ``examples/01_basic`` artifacts."""
        return self.base("basic")

    @property
    def trajopt(self) -> Path:
        """Base path of the ``examples/02_trajopt`` artifacts."""
        return self.base("trajopt")

    @property
    def arm(self) -> Path:
        """Base path of the ``examples/05_ros2_control`` artifacts."""
        return self.base("arm")

    @property
    def trajopt_cases(self) -> Path:
        """The reference-case manifest the C++ comparison reads."""
        return self.dir / "trajopt_cases.json"

    def __truediv__(self, other: str) -> Path:
        """``artifacts / "basic.json"`` reads better than ``.dir / ...``."""
        return self.dir / other

    def __fspath__(self) -> str:
        """So the directory itself can be passed to a subprocess."""
        return str(self.dir)


def _require_jax(python: Sequence[str]) -> None:
    """Skip unless ``python`` can import JAX.

    ``find_spec`` rather than ``import jax``: this runs before every export,
    importing JAX costs seconds, and the question is only whether it is there.
    """
    probe = (
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec('jax') else 1)"
    )
    try:
        found = subprocess.run(
            [*python, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"cannot run {shlex.join(python)} for the export ({exc})")
    if found.returncode != 0:
        pytest.skip(
            f"{shlex.join(python)} has no jax, so the artifacts cannot be "
            "exported; run `uv sync`, or set $PYTHON to an interpreter that "
            "has it, or set CJFC_SKIP_EXPORT=1 to reuse artifacts on disk"
        )


@pytest.fixture(scope="session")
def artifacts(repo: Repo) -> Artifacts:
    """Export ``basic``, ``trajopt`` and ``arm``, and return where they landed.

    A ``.binpb`` embeds machine code for the host that produced it, so
    artifacts are exported per session rather than committed.
    ``CJFC_SKIP_EXPORT=1`` reuses what is on disk, but only when the whole
    set is there: a partial set reused silently ends in a test asserting
    against a sidecar from a different function.
    """
    names = ("basic", "trajopt", "arm")
    wanted = [
        repo.artifact(f"{name}{suffix}")
        for name in names
        for suffix in (".binpb", ".mlirbc", ".json")
    ]
    wanted.append(repo.artifact("trajopt_cases.json"))

    if _env_bool("CJFC_SKIP_EXPORT") and all(p.is_file() for p in wanted):
        return Artifacts(repo.artifacts_dir)

    python = _export_python()
    _require_jax(python)
    repo.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # As the Makefile does: an interpreter that has jax but not this package
    # still has to find jax2exec in the source tree.
    pythonpath = os.pathsep.join(
        p for p in (str(repo.python_dir), os.environ.get("PYTHONPATH", "")) if p
    )
    env = {"PYTHONPATH": pythonpath}
    out = str(repo.artifacts_dir)
    exports = {
        "examples/01_basic/export.py": (),
        "examples/02_trajopt/export.py": ("--cases", "4"),
        "examples/05_ros2_control/export.py": (),
    }
    for script, extra in exports.items():
        run(
            [*python, script, "--out", out, *extra],
            cwd=repo.root,
            env=env,
            timeout=900,
        )

    missing = [str(p) for p in wanted if not p.is_file()]
    if missing:
        pytest.fail(
            "the export scripts reported success but did not write "
            + ", ".join(missing)
        )
    return Artifacts(repo.artifacts_dir)


# -------------------------------------------------------------------- guard


@pytest.fixture(scope="session")
def guard_so(repo: Repo) -> Path:
    """The preloadable allocation counter, or a skip.

    Never linked, only ``LD_PRELOAD``-ed.  Request ``build`` as well if the
    test should build it rather than find it.
    """
    path = repo.lib_dir / "malloc_guard.so"
    if not path.is_file():
        pytest.skip(f"no allocation counter at {path}; run `make guard`")
    return path


# --------------------------------------------------------------------- host


@dataclasses.dataclass(frozen=True)
class Host:
    """What this machine is, for tests that have to reason about jitter."""

    system: str
    nproc: int
    in_container: bool
    isolated: tuple[int, ...]
    loadavg: tuple[float, float, float]

    @property
    def loadavg1(self) -> float:
        """The 1-minute load average as it was when the session started."""
        return self.loadavg[0]

    @property
    def busy(self) -> bool:
        """Whether the session started on a loaded machine.  The threshold is
        ``examples/common/rt_env.hpp``'s, so a C++ report and a Python skip
        never disagree about one host."""
        return self.loadavg[0] > 1.0


@pytest.fixture(scope="session")
def host() -> Host:
    """This host's cpu count, isolation, container status and load."""
    try:
        nproc = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not on Linux
        nproc = os.cpu_count() or 1
    return Host(
        system=platform.system(),
        nproc=nproc,
        in_container=_in_container(),
        isolated=_parse_cpulist(_read_text("/sys/devices/system/cpu/isolated")),
        loadavg=_loadavg(),
    )


# ------------------------------------------------- damaged artifact copies


class TmpArtifacts:
    """Copies of an artifact set that a test may damage on purpose.

    Each of the loader's refusals needs an artifact wrong in a specific way,
    and damaging ``artifacts/`` would poison every later test in the session.
    """

    #: One artifact set.  ``.mlirbc`` is optional: ``export(write_mlir=False)``
    #: produces a set without it.
    SUFFIXES = (".binpb", ".mlirbc", ".json")

    def __init__(self, source: Path, root: Path) -> None:
        self._source = source
        self._root = root
        self._made = 0

    def copy(self, name: str = "basic") -> Path:
        """Copy an artifact set into a fresh directory and return its base
        path -- no extension, as ``pjrt::Runtime`` is handed it."""
        into = self._root / f"copy{self._made}"
        self._made += 1
        into.mkdir(parents=True, exist_ok=True)

        copied = []
        for suffix in self.SUFFIXES:
            source = self._source / f"{name}{suffix}"
            if source.is_file():
                shutil.copy2(source, into / source.name)
                copied.append(suffix)
        if ".binpb" not in copied or ".json" not in copied:
            pytest.fail(
                f"{self._source / name} is not a complete artifact set "
                f"(found {copied or 'nothing'}); the `artifacts` fixture "
                "exports one"
            )

        # The reference cases travel with the copy, so a test can damage one.
        manifest = self._source / f"{name}_cases.json"
        if manifest.is_file():
            shutil.copy2(manifest, into / manifest.name)
            for case in sorted(self._source.glob(f"{name}_case*.bin")):
                shutil.copy2(case, into / case.name)

        return into / name

    def edit_sidecar(
        self, base: Path, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        """Apply ``mutate`` to the sidecar, write it back, and return it."""
        path = base.with_suffix(".json")
        sidecar = load_json(path)
        mutate(sidecar)
        path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        return sidecar

    def drop_output(self, base: Path) -> dict[str, Any]:
        """Remove the last output from the sidecar, renumbering the rest.

        The executable still produces it, so the sidecar now under-declares:
        the disagreement the loader checks for, and the one the C API can
        see (it answers ``PJRT_Executable_NumOutputs``; there is no query for
        the inputs).
        """

        def mutate(sidecar: dict[str, Any]) -> None:
            outputs = sidecar.get("outputs")
            if not outputs:
                pytest.fail(f"{base}.json declares no outputs to drop")
            outputs.pop()
            for position, entry in enumerate(outputs):
                entry["index"] = position

        return self.edit_sidecar(base, mutate)

    def truncate_executable(self, base: Path) -> Path:
        """Cut the ``.binpb`` to half: past any header, inside the code."""
        path = base.with_suffix(".binpb")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        return path

    def remove(self, base: Path, suffix: str) -> Path:
        """Delete one file of a set, e.g. the ``.mlirbc`` fallback."""
        path = base.with_suffix(suffix)
        path.unlink(missing_ok=True)
        return path


@pytest.fixture
def tmp_artifacts(artifacts: Artifacts, tmp_path: Path) -> TmpArtifacts:
    """A factory for artifact copies this test is free to damage."""
    return TmpArtifacts(source=artifacts.dir, root=tmp_path)
