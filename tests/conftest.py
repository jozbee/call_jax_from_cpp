"""Fixtures shared by the Python and the C++ halves of the suite.

The C++ tests are pytest tests as well: each one builds a binary, runs it and
reads what it printed.  Everything expensive is therefore session scoped and
resolved exactly once -- the build, the plugin, the exported artifacts -- and
everything that can be legitimately absent on a healthy machine produces a
SKIP carrying the remedy rather than a failure.  A clone with no PJRT plugin
should report a handful of skips, not a wall of errors; a test that cannot run
its subject must never pass vacuously either, which is why nothing here
degrades to a no-op assertion.

The real-time gates are conditional for a measured reason.  On this
developer host the same artifact on the same pinned core reports p50 2027 us
at a 3 ms period and p50 5196 us at a 10 ms period, because the powersave
governor clocks the core down while the loop idles.  A fixed latency
threshold would fail on a perfectly healthy machine, so the ``rt`` marker runs
only where the host has been audited as tuned and idle (see :func:`rt_strict`).

Environment variables, all optional:

``CJFC_BUILD_DIR``
    Build tree.  Default ``build`` under the repository root.
``CJFC_ARTIFACTS_DIR``
    Where the exported artifacts live.  Default ``artifacts``.
``CJFC_REPORT_DIR``
    Where binaries write JSON reports.  Default ``$CJFC_ARTIFACTS_DIR/reports``.
``CJFC_BUILD_TOOL``
    ``make`` (default), ``cmake``, or ``none``.
``CJFC_SKIP_BUILD``
    ``1`` to take the tree exactly as it stands.
``CJFC_SKIP_EXPORT``
    ``1`` to reuse the artifacts already on disk instead of re-exporting.
``CJFC_RT_STRICT``
    ``1``/``0`` to force the real-time gates on or off, ahead of ``--rt-strict``.
``PJRT_CPU_PLUGIN``
    The plugin to load, overriding ``$CJFC_BUILD_DIR/plugin``.
``PYTHON``
    Interpreter for the export scripts.  Default: the one running pytest,
    which under ``uv run pytest`` is the project's virtual environment.

Relative paths in the ``CJFC_*`` variables are resolved against the repository
root, not against pytest's working directory, so they mean the same thing the
identically named ``make`` variables do.

The helpers :func:`run`, :func:`load_json` and :func:`parse_kv_lines` exist
both as module-level functions (``from conftest import parse_kv_lines``) and
as fixtures of the same name, so a test can take whichever is less noise.
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

#: The repository root: this file lives in ``tests/``.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: What ``make plugin`` and ``tools/get_plugin.sh`` write.
PLUGIN_FILE_NAME = "libpjrt_c_api_cpu_plugin.so"

#: Fixtures whose presence means the test builds or runs a native binary.
#: Those are Linux-only here -- the plugin this project publishes is an ELF
#: shared object -- so collection skips any test that asks for one elsewhere.
_BINARY_FIXTURES = frozenset({"build", "plugin", "guard_so"})

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


# --------------------------------------------------------------- environment


def _env_bool(name: str) -> bool | None:
    """Read a tri-state flag from the environment.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    bool or None
        None when unset or empty, so a caller can tell "not configured" from
        "configured off" -- the difference between falling back to the
        automatic real-time audit and being told to skip the gates outright.

    Raises
    ------
    pytest.UsageError
        For a value that is neither true-ish nor false-ish.  Guessing would
        turn a typo into a silently skipped test suite.
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
    """Return the interpreter to run the export scripts with, as argv.

    ``$PYTHON`` is honoured because that is the escape hatch the Makefile
    documents (``make PYTHON=python3 ...``).  The default is the interpreter
    running pytest rather than ``uv run python``: ``uv`` comes from mise and is
    not on PATH in a non-interactive shell, while ``uv run pytest`` has already
    put us inside the environment that has JAX.
    """
    raw = os.environ.get("PYTHON", "").strip()
    return shlex.split(raw) if raw else [sys.executable]


def _loadavg() -> tuple[float, float, float]:
    """Return the 1, 5 and 15 minute load averages, or -1 where unavailable."""
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - not on Linux
        return (-1.0, -1.0, -1.0)
    return (one, five, fifteen)


def _in_container() -> bool:
    """Whether this process is running inside a container.

    The same three signals ``examples/common/rt_env.hpp`` and
    ``tools/rt_check.sh`` use, so all three agree about one host.  It matters
    for the real-time gates because the settings that decide the tail --
    governor, isolcpus, nohz_full, C-states -- belong to the host kernel and
    cannot be read, let alone fixed, from inside.
    """
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or bool(os.environ.get("CJFC_IN_CONTAINER"))
    )


def _parse_cpulist(text: str) -> tuple[int, ...]:
    """Parse a kernel cpulist (``0-3,8``) into cpu numbers."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part or part == "(null)":
            continue
        if "-" in part:
            first, _, last = part.partition("-")
            try:
                cpus.extend(range(int(first), int(last) + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return tuple(sorted(set(cpus)))


def _read_text(path: str | Path) -> str:
    """Read a sysfs file, returning "" when it is not there."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ------------------------------------------------------------------ options


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the two switches that decide what the suite is allowed to run."""
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
    """Decide, by measurement, whether this host may be gated on its tail.

    Returns
    -------
    tuple of (bool, str)
        Whether the gates should run, and why -- the reason goes into the skip
        message, because "skipped" without a cause is indistinguishable from a
        test that was quietly deleted.

    Notes
    -----
    Three conditions, all necessary.  ``tools/rt_check.sh`` exits non-zero when
    anything on the host is worth fixing (governor, PREEMPT_RT, isolcpus,
    nohz_full, THP, rtprio, memlock).  A one-minute load average at or above
    1.0 means something else is running, and a concurrent build does not add
    noise to a tail measurement, it invalidates it: the same configuration
    measured during a bazel build reported a p50 2.4x high and a max/p50 of
    4.4 instead of 1.1.  A container can only see the host's kernel settings,
    not set them.

    Cached: the audit is a subprocess, and the answer cannot change usefully
    within one session.
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
    """Return the strict-mode decision and the reason behind it."""
    forced = _env_bool("CJFC_RT_STRICT")
    if forced is not None:
        return forced, f"$CJFC_RT_STRICT={'1' if forced else '0'}"
    if config.getoption("--rt-strict"):
        return True, "--rt-strict was passed"
    return _auto_rt_strict()


#: How a skipped real-time gate explains itself.  The reason comes first
#: because it is the part that differs between hosts.
_RT_SKIP = (
    "real-time gates are off: {why}. Turn them on with --rt-strict or "
    "CJFC_RT_STRICT=1, on a tuned and idle host"
)


def rt_strict(config: pytest.Config) -> bool:
    """Whether the real-time gates may assert on this host.

    Parameters
    ----------
    config : pytest.Config
        The session configuration, for the ``--rt-strict`` flag.

    Returns
    -------
    bool
        ``$CJFC_RT_STRICT`` when it is set, else true when ``--rt-strict`` was
        passed, else the automatic audit in :func:`_auto_rt_strict`.

    Notes
    -----
    The automatic answer is False on an ordinary developer machine, and that is
    correct rather than a defect: a powersave governor, a shared timer tick and
    no isolated cpus produce latencies that vary by 2.5x with nothing but the
    caller's period.  Numbers measured there describe the machine, not the code.
    """
    return _rt_strict_reason(config)[0]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply the three conditional skips.

    ``slow`` needs ``--runslow``; ``rt`` needs strict mode; and anything that
    builds or launches a native binary needs Linux, because the plugin this
    project publishes is an ELF shared object and the examples use Linux-only
    scheduling calls.  A test is recognised as native by asking for one of the
    fixtures that resolve a binary, a plugin or the preloadable guard.
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
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capture its output, and report all of it on failure.

    Parameters
    ----------
    cmd : Sequence or str
        Argv.  A string is split with :mod:`shlex`; ``Path`` entries are
        stringified, so ``run([build.bin("fn_info"), artifacts / "basic"])``
        needs no conversions.
    cwd : path-like, optional
        Working directory.  Defaults to the repository root, which is where
        every path in this suite is anchored.
    env : Mapping, optional
        Variables **added to** the current environment, not a replacement for
        it: ``$PJRT_CPU_PLUGIN`` and ``$LD_PRELOAD`` have to survive, and so
        does whatever ``uv run`` put there.  A value of None removes a
        variable instead, which is the only way to guarantee a child runs with
        nothing preloaded whatever the parent inherited.
    timeout : float, optional
        Seconds before the process is killed and the test fails.
    check : bool, optional
        Fail the test on a non-zero exit.  Pass False when the non-zero exit
        is the thing being tested, then assert on the returned object.
    stdin : str, optional
        Text written to the process's standard input.

    Returns
    -------
    subprocess.CompletedProcess
        With ``stdout`` and ``stderr`` as text.

    Notes
    -----
    The command line and any output are printed unconditionally.  pytest shows
    captured output only for tests that fail, so this costs nothing on a green
    run and means a red one arrives with the evidence attached -- which is the
    difference between "the binary exited 1" and seeing the LoadError it
    printed.
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
            input=stdin,
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

    Parameters
    ----------
    stdout : str
        What the program printed.

    Returns
    -------
    dict
        Two line shapes are recognised, and everything else is ignored:

        ``key=value``
            One or more per line, whitespace separated, landing at the top
            level: ``num_inputs=2 num_outputs=2`` gives two entries.
        ``label: key=value ...``
            A labelled group, landing as a nested dict under the label:
            ``input[0]: dtype=float64 shape=[4,4] numel=16 nbytes=128``
            gives ``result["input[0]"]["dtype"] == "float64"``.  A labelled
            line with no ``key=value`` in it keeps its text verbatim, which is
            what makes the debug checks readable:
            ``result["debug_check[non_finite]"]`` is the whole message.

        Values stay strings, deliberately.  ``shape=[4,4]`` has no obvious
        number to convert to, and a test that wants an int should say so.
        A repeated key takes the last value, so a binary that prints its
        summary twice reports the second pass.
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
        """The repository root, so ``Path(repo, "versions.env")`` works.

        A fixture that names a directory should be usable as one; the fields
        above are for the tests that want a specific directory rather than the
        root.
        """
        return str(self.root)


@pytest.fixture(scope="session")
def repo() -> Repo:
    """The resolved directory layout for this session."""
    build_dir = _dir_from_env("CJFC_BUILD_DIR", REPO_ROOT / "build")
    artifacts_dir = _dir_from_env("CJFC_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
    report_dir = _dir_from_env("CJFC_REPORT_DIR", artifacts_dir / "reports")

    # The binaries write reports with --json <path> and do not create the
    # directory first; make declares it an order-only prerequisite for the
    # same reason.
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
        """Return a built binary, failing with the target that produces it.

        A binary missing after a successful build is a defect in the build, not
        a reason to skip: the suite was told to build it and reported success.
        """
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

    Honours ``$CJFC_BUILD_TOOL`` (``make``, ``cmake`` or ``none``) and
    ``$CJFC_SKIP_BUILD=1``, which is what CI wants when a previous step already
    built the tree, and what a bisect wants when the tree on disk is the point.

    A build failure fails the tests loudly, with the compiler's output: the
    alternative -- skipping -- would report a green run for a tree that does
    not compile.
    """
    tool = (os.environ.get("CJFC_BUILD_TOOL") or "make").strip().lower()
    if _env_bool("CJFC_SKIP_BUILD"):
        return Build(repo=repo, tool="none ($CJFC_SKIP_BUILD)")
    if tool == "none":
        return Build(repo=repo, tool="none")

    if tool == "make":
        command: list[str] = ["make", "-s"]
        # Only pass the overrides that differ, so the command in the failure
        # message is the one a reader can paste into a shell.
        #
        # `bench` has no phony target of its own and has to be named by path.
        # make compares target names as strings, so the path must be spelled
        # the way $(BIN_DIR) is: relative under the default BUILD_DIR, and
        # absolute only when BUILD_DIR was overridden with an absolute path.
        # `make /abs/path/build/bin/bench` against the default reports
        # "Nothing to be done" and builds nothing at all.
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
                # The plugin is the `plugin` fixture's business, and a test
                # session must not start a download of its own.
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

    Absent is a normal state for a fresh clone -- the plugin is a 100 MB
    download or an hour of bazel, and nothing in the build depends on it -- so
    this skips with the remedy instead of failing.  The runtime resolves the
    same two sources in the same order (``$PJRT_CPU_PLUGIN``, then the path
    compiled in at build time), so what this fixture finds is what a binary
    started from this environment will open.
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

    # Publish it, so every binary these tests launch opens the plugin the
    # fixture just resolved rather than the path compiled into it. Those are
    # usually the same file, but not always: the build bakes in an absolute
    # path, so a tree built on the host and then tested inside the container
    # has binaries pointing at a directory that does not exist there. The
    # environment variable outranks the compiled default, which is exactly the
    # override it is for.
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

    ``find_spec`` rather than ``import jax``: this runs before every export and
    importing JAX costs a couple of seconds, while the question asked is only
    whether the module is installed at all.
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
    """Export ``basic`` and ``trajopt``, and return where they landed.

    A serialized executable embeds machine code for the host that produced it,
    so artifacts are exported rather than committed, and re-exported by default
    on every session -- the exporter is the thing half of these tests are
    about.  ``CJFC_SKIP_EXPORT=1`` reuses what is on disk, but only when the
    whole set is there: a partial set reused silently is how a test ends up
    asserting against a sidecar from a different function.
    """
    names = ("basic", "trajopt")
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

    # The Makefile does the same: an interpreter that has jax but not this
    # package installed still has to find jax2exec in the source tree.
    pythonpath = os.pathsep.join(
        p for p in (str(repo.python_dir), os.environ.get("PYTHONPATH", "")) if p
    )
    env = {"PYTHONPATH": pythonpath}
    out = str(repo.artifacts_dir)
    run(
        [*python, "examples/01_basic/export.py", "--out", out],
        cwd=repo.root,
        env=env,
        timeout=900,
    )
    run(
        [
            *python,
            "examples/02_trajopt/export.py",
            "--out",
            out,
            "--cases",
            "4",
        ],
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

    It is never linked, only ``LD_PRELOAD``-ed, and the binaries resolve its
    markers with ``dlsym`` and no-op when it is absent.  Request ``build`` as
    well if the test should build it rather than find it.
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
        """Whether the session started on a machine that was already loaded.

        The same threshold ``examples/common/rt_env.hpp`` uses, so a C++ report
        and a Python skip message never disagree about one host.
        """
        return self.loadavg[0] > 1.0

    def current_loadavg(self) -> tuple[float, float, float]:
        """Re-read the load average now.

        The snapshot above is from session start; a test about to measure
        something wants the number for the moment it measures it.
        """
        return _loadavg()


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

    The loader's refusals are half its value -- a stale sidecar used to be a
    heap-overrun class of bug -- and every one of them needs an artifact that
    is wrong in a specific way.  Damaging the originals in ``artifacts/`` would
    poison every later test in the session, so each copy lands in its own
    directory under ``tmp_path``.
    """

    #: Extensions that make up one artifact set.  ``.mlirbc`` is optional:
    #: `export(write_mlir=False)` produces a set without it.
    SUFFIXES = (".binpb", ".mlirbc", ".json")

    def __init__(self, source: Path, root: Path) -> None:
        self._source = source
        self._root = root
        self._made = 0

    def copy(self, name: str = "basic") -> Path:
        """Copy an artifact set into a fresh directory.

        Parameters
        ----------
        name : str, optional
            Base name of the set, e.g. ``basic`` or ``trajopt``.

        Returns
        -------
        Path
            The base path of the copy, without an extension -- which is what
            ``pjrt::Runtime`` is handed.
        """
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

        # The reference cases travel with the artifact when there are any, so a
        # test can damage a case without touching the originals every later
        # test reads.
        manifest = self._source / f"{name}_cases.json"
        if manifest.is_file():
            shutil.copy2(manifest, into / manifest.name)
            for case in sorted(self._source.glob(f"{name}_case*.bin")):
                shutil.copy2(case, into / case.name)

        return into / name

    def sidecar(self, base: Path) -> dict[str, Any]:
        """Read the sidecar of a copy."""
        return load_json(base.with_suffix(".json"))

    def write_sidecar(self, base: Path, sidecar: Mapping[str, Any]) -> None:
        """Write a sidecar back, in the layout the exporter writes."""
        text = json.dumps(sidecar, indent=2, sort_keys=False) + "\n"
        base.with_suffix(".json").write_text(text, encoding="utf-8")

    def edit_sidecar(
        self, base: Path, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        """Apply ``mutate`` to the sidecar in place and write it back.

        Returns the sidecar as written, so a test can assert on what it did.
        """
        sidecar = self.sidecar(base)
        mutate(sidecar)
        self.write_sidecar(base, sidecar)
        return sidecar

    def drop_output(self, base: Path, index: int = -1) -> dict[str, Any]:
        """Remove one output from the sidecar, renumbering the rest.

        The executable still returns what it always returned, so the sidecar
        now under-declares -- the disagreement the loader checks for, and the
        one the PJRT C API can actually see, since it answers
        ``PJRT_Executable_NumOutputs`` but has no query for the inputs.
        """

        def mutate(sidecar: dict[str, Any]) -> None:
            outputs = sidecar.get("outputs")
            if not outputs:
                pytest.fail(f"{base}.json declares no outputs to drop")
            outputs.pop(index)
            for position, entry in enumerate(outputs):
                entry["index"] = position

        return self.edit_sidecar(base, mutate)

    def truncate_executable(
        self, base: Path, *, keep: int | None = None
    ) -> Path:
        """Cut the ``.binpb`` short, leaving a file no plugin can deserialize.

        Parameters
        ----------
        base : Path
            Base path from :meth:`copy`.
        keep : int, optional
            Bytes to keep.  Default: half the file, which is past any header
            and firmly inside the executable.

        Returns
        -------
        Path
            The truncated file.
        """
        path = base.with_suffix(".binpb")
        data = path.read_bytes()
        cut = len(data) // 2 if keep is None else max(0, min(keep, len(data)))
        path.write_bytes(data[:cut])
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
