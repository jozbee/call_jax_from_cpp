"""Measure which XLA code-generation flags are faster for one JAX function.

Arms are interleaved in rounds against an A/A arm that measures the noise,
because a sequential A/B drifts with CPU temperature by the same order as
the effect being looked for.  Imports no JAX: every measurement happens in
a child process, because ``XLA_FLAGS`` is read once at backend start-up.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import pickle
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._flags import (
    CATALOG,
    EXTRA_OPTIONS,
    Candidate,
    applicable,
    drop_flag,
    flag_key,
    merge_flags,
    split_flags,
    version_tuple,
)
from ._hygiene import (
    child_affinity,
    confine_driver,
    cpu_facts,
    cpu_jiffies,
    cpu_occupancy,
    loadavg_1min,
)

__all__ = [
    "CATALOG",
    "COMPILE_DRIFT_LIMIT",
    "LOADAVG_LIMIT",
    "MIN_GAIN",
    "MIN_VERDICT_ROUNDS",
    "REJECT_PATTERNS",
    "RESULT_SCHEMA",
    "SIBLING_BUSY_LIMIT",
    "SMOKE_CODE",
    "Arm",
    "ArmResult",
    "Candidate",
    "TuneResult",
    "aa_band",
    "combination_arm",
    "combinations",
    "drop_flag",
    "flag_key",
    "load_result",
    "merge_flags",
    "run_arms",
    "split_flags",
    "tune_flags",
    "verdict",
    "version_tuple",
]

#: Smallest median speedup worth carrying a flag for.  It is a second gate
#: and not a restatement of the verdict: a flag can be faster in every round
#: and still not be worth writing down forever.
MIN_GAIN = 0.03

#: Layout of what :meth:`jax2exec.TuneResult.save` writes.  :func:`load_result`
#: refuses anything newer, because a field it does not know about is a field
#: it would silently drop.
RESULT_SCHEMA = 1

#: One-minute load average above which a child's timings are flagged.  A
#: busy machine does not add noise to a latency, it invalidates it.  The
#: measurement's own child holds one core, so the limit allows one and flags
#: half a core more; a limit below one marks every row of every campaign.
LOADAVG_LIMIT = 1.5

#: Busy fraction of the measured core's SMT sibling above which a child is
#: flagged: a sibling shares the physical core's execution resources.
SIBLING_BUSY_LIMIT = 0.02

#: Relative spread of an arm's compile seconds across rounds above which its
#: row is marked.  Some flags move compile time far more than run time, and
#: a drifting compile is the first sign that the machine changed underneath.
COMPILE_DRIFT_LIMIT = 0.30

#: Rounds a verdict needs before it will say anything.  ``all()`` over one
#: surviving ratio is trivially true, and a one-round "faster" would be
#: typographically identical to a five-round one.
MIN_VERDICT_ROUNDS = 2

#: What a jaxlib prints when it refuses a flag.  The first three are the
#: current spellings -- an unknown flag, a bad enum value, a bad value for a
#: validated flag -- and the rest are older ones, kept so that an older
#: interpreter is still recognised.
REJECT_PATTERNS = (
    "Unknown flag in XLA_FLAGS",
    "Flag parsing failed",
    "Illegal value for --xla",
    "Unknown command line flag",
    "unknown command line flag",
    "Illegal value for flag",
    "ERROR: Illegal value",
)

#: The tiny jitted program the smoke probe compiles.  Small enough that what
#: it measures is start-up and flag acceptance, which is all it claims.
SMOKE_CODE = (
    "import jax, jax.numpy as jnp\n"
    "jax.config.update('jax_enable_x64', True)\n"
    "f = jax.jit(lambda x: (x @ x.T).sum())\n"
    "jax.block_until_ready(f(jnp.ones((16, 16))))\n"
    "print('ok')\n"
)

#: Characters of a captured stream kept in a record.
_STREAM_CAP = 8000

_BASELINE = "baseline"
_BASELINE_AA = "baseline-aa"


@dataclass(frozen=True)
class Arm:
    """One configuration under comparison.

    Parameters
    ----------
    label : str
        Name of the arm; it names the per-child record and the table row.
    flags : tuple of str
        ``XLA_FLAGS`` tokens this arm adds on top of ``base_flags``.
    members : tuple of str
        Catalog labels the arm stands for, when it is a combination.
    role : str
        ``"baseline"``, ``"candidate"``, ``"control"`` or ``"ceiling"``.
        Only a ``"candidate"`` can win; a ``"control"`` that does not read
        ``slower`` invalidates the whole session.
    numerics : str
        ``"exact"`` or ``"inexact"``, carried through from the catalog.
    aa_of : str or None
        Label of the arm this one duplicates, when it is the A/A arm.  An
        A/A arm is the ruler and is never a winner.
    python : str or None
        Interpreter the child runs under; ``None`` uses the caller's.
    expected_jax : str or None
        Version that interpreter must resolve to.  ``None`` skips the check.
    note : str
        Free text carried into the table.
    """

    label: str
    flags: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    role: str = "candidate"
    numerics: str = "exact"
    aa_of: str | None = None
    python: str | None = None
    expected_jax: str | None = None
    note: str = ""


@dataclass(frozen=True)
class ArmResult:
    """What one arm's rounds came to.

    Parameters
    ----------
    label, flags, members, role, numerics, aa_of : Any
        Carried over from the :class:`jax2exec.Arm`.
    status : str
        ``"ok"``, ``"rejected"`` (this jaxlib refused a flag), ``"failed"``,
        ``"timeout"``, ``"mismatch"`` (its outputs disagreed with the
        reference's in shape or dtype) or ``"partial"`` (some rounds ran and
        some did not).
    median_ms, p95_ms, p99_ms, max_ms, compile_s : dict
        Round number -> what the child reported.  ``p99_ms`` is absent for a
        child that timed fewer than 100 calls, where a p99 would be the
        maximum wearing another name.
    ratio : dict
        Round number -> this arm's median over the reference's median *in
        that round*, which is the only comparison the interleaving licenses.
    median_ratio : float or None
        Median of those ratios.
    verdict : str
        ``"faster"``, ``"slower"``, ``"within A/A"``,
        ``"insufficient rounds"``, ``"reference"`` or ``"no A/A band"``.
    span : tuple of float or None
        ``(min, max)`` of the ratios.
    rounds_ok : int
        Rounds that produced a timing.
    max_abs_dev, max_rel_dev : float or None
        Largest deviation of this arm's outputs from the reference's, over
        every leaf of the first round's call.
    hygiene : tuple of str
        ``"loadavg"``, ``"sibling"``, ``"affinity"``, ``"compile-drift"`` --
        each present when any round tripped that limit.
    note : str
        Free text from the arm.
    """

    label: str
    flags: tuple[str, ...]
    members: tuple[str, ...]
    role: str
    numerics: str
    aa_of: str | None
    status: str
    median_ms: dict[int, float]
    p95_ms: dict[int, float]
    p99_ms: dict[int, float]
    max_ms: dict[int, float]
    compile_s: dict[int, float]
    ratio: dict[int, float]
    median_ratio: float | None
    verdict: str
    span: tuple[float, float] | None
    rounds_ok: int
    max_abs_dev: float | None
    max_rel_dev: float | None
    hygiene: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class TuneResult:
    """Everything one campaign measured, and whether it may be believed.

    Parameters
    ----------
    schema : int
        :data:`jax2exec.tune.RESULT_SCHEMA` as written.
    host : dict
        the host facts this package reads of the machine that measured.
    jax, jaxlib : str or None
        Versions the children resolved to.
    reference, aa : str
        Labels of the arm every ratio is taken against and of the A/A arm.
    band : tuple of float or None
        ``(low, high)`` the A/A arm's own ratios imply; ``None`` when it
        produced none, and then no arm gets a verdict.
    rounds, reps, warmup : int
        The protocol that ran.
    min_gain : float
        The winner gate that was applied.
    base_flags : tuple of str
        Flags every arm carried, the baseline included.
    target : str
        ``"module:qualname"`` of what was timed.
    started_at : str
        UTC start, ISO 8601.
    wall_s : float
        Wall clock of the whole campaign.
    arms : tuple
        One :class:`jax2exec.ArmResult` per arm, in declared order, the
        reference first.
    dropped : tuple
        ``(label, reason)`` per candidate that was never measured.
    winners : tuple of str
        Candidate arms that read ``faster`` and beat `min_gain`.
    valid : bool
        False when the control arm did not read ``slower`` or the A/A arm
        failed: a protocol that cannot see a known loss, or has no ruler,
        cannot see a small win either.
    invalid_reason : str or None
        Why, in words.
    records_dir : str or None
        Where the per-child records were kept, or ``None`` when they went to
        a temporary directory that has been removed.
    """

    schema: int
    host: dict[str, Any]
    jax: str | None
    jaxlib: str | None
    reference: str
    aa: str
    band: tuple[float, float] | None
    rounds: int
    reps: int
    warmup: int
    min_gain: float
    base_flags: tuple[str, ...]
    target: str
    started_at: str
    wall_s: float
    arms: tuple[ArmResult, ...]
    dropped: tuple[tuple[str, str], ...]
    winners: tuple[str, ...]
    valid: bool
    invalid_reason: str | None
    records_dir: str | None

    def table(self) -> str:
        """Render the campaign as a markdown table.

        Returns
        -------
        str
            One row per arm, then the A/A band the verdicts were judged
            against, then the caveats the rows carry.
        """
        return _render_table(self)

    def save(self, path: str | Path) -> Path:
        """Write the result as JSON.

        Parameters
        ----------
        path : str or Path
            Destination file.

        Returns
        -------
        Path
            The path written, so a caller can log it.
        """
        destination = Path(path)
        destination.write_text(
            json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8"
        )
        return destination


def load_result(path: str | Path) -> TuneResult:
    """Read back a :meth:`jax2exec.TuneResult.save` file.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    TuneResult
        The result, with the round-keyed dictionaries keyed by int again.

    Raises
    ------
    ValueError
        If the file was written by a newer layout.  A field this package
        does not know about is a field it would silently drop.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = int(data.get("schema", 0))
    if schema > RESULT_SCHEMA:
        raise ValueError(
            f"{path} is schema {schema}; this package reads up to schema "
            f"{RESULT_SCHEMA}"
        )
    arms = tuple(_arm_result_from_json(row) for row in data["arms"])
    return TuneResult(
        schema=schema,
        host=data["host"],
        jax=data["jax"],
        jaxlib=data["jaxlib"],
        reference=data["reference"],
        aa=data["aa"],
        band=_as_pair(data["band"]),
        rounds=int(data["rounds"]),
        reps=int(data["reps"]),
        warmup=int(data["warmup"]),
        min_gain=float(data["min_gain"]),
        base_flags=tuple(data["base_flags"]),
        target=data["target"],
        started_at=data["started_at"],
        wall_s=float(data["wall_s"]),
        arms=arms,
        dropped=tuple((label, reason) for label, reason in data["dropped"]),
        winners=tuple(data["winners"]),
        valid=bool(data["valid"]),
        invalid_reason=data["invalid_reason"],
        records_dir=data["records_dir"],
    )


def _as_pair(value: Any) -> tuple[float, float] | None:
    """Return a two-element list read from JSON as a tuple."""
    if value is None:
        return None
    low, high = value
    return (float(low), float(high))


def _int_keys(mapping: dict[Any, Any]) -> dict[int, float]:
    """Return a JSON object keyed by round number rather than by string."""
    return {int(key): float(value) for key, value in mapping.items()}


def _arm_result_from_json(row: dict[str, Any]) -> ArmResult:
    """Rebuild one :class:`jax2exec.ArmResult` from its JSON form."""
    return ArmResult(
        label=row["label"],
        flags=tuple(row["flags"]),
        members=tuple(row["members"]),
        role=row["role"],
        numerics=row["numerics"],
        aa_of=row["aa_of"],
        status=row["status"],
        median_ms=_int_keys(row["median_ms"]),
        p95_ms=_int_keys(row["p95_ms"]),
        p99_ms=_int_keys(row["p99_ms"]),
        max_ms=_int_keys(row["max_ms"]),
        compile_s=_int_keys(row["compile_s"]),
        ratio=_int_keys(row["ratio"]),
        median_ratio=row["median_ratio"],
        verdict=row["verdict"],
        span=_as_pair(row["span"]),
        rounds_ok=int(row["rounds_ok"]),
        max_abs_dev=row["max_abs_dev"],
        max_rel_dev=row["max_rel_dev"],
        hygiene=tuple(row["hygiene"]),
        note=row["note"],
    )


def aa_band(ratios: Sequence[float]) -> tuple[float, float]:
    """The noise band an A/A arm's per-round ratios imply.

    Symmetric about 1: a one-sided ``min .. max`` band leaves no allowance
    on whichever side the A/A ratios happened to miss, and a campaign that
    exists to find speedups must not have its tighter edge on the fast side.

    Parameters
    ----------
    ratios : Sequence of float
        The A/A arm's per-round ratios.

    Returns
    -------
    tuple of float
        ``(1 - w, 1 + w)``, where ``w`` is the largest deviation from 1 seen
        in either direction, so that a ratio and its reciprocal give the
        same width.

    Raises
    ------
    ValueError
        If no ratio is positive.  The filter is applied before the maximum
        rather than inside it, so a list of zeros raises this rather than
        ``max() arg is an empty sequence``.
    """
    widths = [
        max(abs(ratio - 1.0), abs(1.0 / ratio - 1.0))
        for ratio in ratios
        if ratio > 0
    ]
    if not widths:
        raise ValueError("no positive A/A ratios")
    return (1.0 - max(widths), 1.0 + max(widths))


def verdict(
    ratios: Sequence[float], low: float, high: float, rounds: int
) -> str:
    """Decide whether an arm sits outside the A/A band in every round.

    Parameters
    ----------
    ratios : Sequence of float
        The arm's per-round ratios to the reference.
    low, high : float
        The A/A band.
    rounds : int
        Rounds the campaign intended to run.

    Returns
    -------
    str
        ``"faster"``, ``"slower"``, ``"within A/A"`` or
        ``"insufficient rounds"`` -- the last from fewer than half the
        rounds, and never from fewer than ``MIN_VERDICT_ROUNDS``.
    """
    need = max(MIN_VERDICT_ROUNDS, (rounds + 1) // 2)
    if len(ratios) < need:
        return "insufficient rounds"
    if all(ratio < low for ratio in ratios):
        return "faster"
    if all(ratio > high for ratio in ratios):
        return "slower"
    return "within A/A"


########################
#      COMBINATIONS    #
########################


def _conflicting(
    members: Sequence[str], by_label: dict[str, Candidate]
) -> bool:
    """Whether a set holds two settings of one switch."""
    keys = [flag_key(by_label[label].flag) for label in members]
    return len(set(keys)) < len(keys)


def combinations(
    winners: Sequence[str],
    *,
    max_size: int = 3,
    candidates: Iterable = CATALOG,
) -> list[tuple[str, ...]]:
    """Enumerate the sets of winning switches worth measuring together.

    Two flags that each won on their own need not win together, so a set is
    a hypothesis and has to be measured as its own arm.

    Parameters
    ----------
    winners : Sequence of str
        Catalog labels, best first, as ``winners`` orders
        them.
    max_size : int
        Most winners for which every non-empty subset is enumerated.  Beyond
        it the enumeration is the first pass of greedy forward selection:
        ``2**k - 1`` arms at several rounds each is more wall clock than the
        answer is worth.
    candidates : Iterable
        The catalog of :class:`jax2exec.Candidate` the labels are resolved
        against.

    Returns
    -------
    list of tuple of str
        Member tuples, increasing in size and, within a size, in the
        winners' own order.  A set that would set one switch twice is
        omitted: merging it would keep one value and the arm would then be
        measuring something its own label does not say.
    """
    by_label = {entry.label: entry for entry in candidates}
    if not winners:
        return []
    if len(winners) > max_size:
        return [(label,) for label in winners]
    sets: list[tuple[str, ...]] = []
    for size in range(1, len(winners) + 1):
        for members in itertools.combinations(winners, size):
            if not _conflicting(members, by_label):
                sets.append(members)
    return sets


def combination_arm(
    members: Sequence[str],
    candidates: Iterable = CATALOG,
) -> Arm:
    """Build the arm that measures one set of switches together.

    Parameters
    ----------
    members : Sequence of str
        Catalog labels.
    candidates : Iterable
        The catalog of :class:`jax2exec.Candidate` the labels are resolved
        against.

    Returns
    -------
    Arm
        Labelled with the members joined by ``+`` and carrying their flags
        in catalog order, so that the same set always produces the same arm.
    """
    entries = list(candidates)
    order = {entry.label: index for index, entry in enumerate(entries)}
    by_label = {entry.label: entry for entry in entries}
    chosen = sorted(members, key=lambda label: order.get(label, len(order)))
    inexact = any(by_label[label].numerics == "inexact" for label in chosen)
    return Arm(
        label="+".join(members),
        flags=tuple(by_label[label].flag for label in chosen),
        members=tuple(members),
        numerics="inexact" if inexact else "exact",
        note=f"{len(chosen)} switch(es): {', '.join(chosen)}",
    )


########################
#       THE DRIVER     #
########################


def tune_flags(
    target: Callable[..., Any] | str,
    args: Sequence[Any] | None = None,
    *,
    candidates: Sequence | None = None,
    rounds: int = 5,
    reps: int = 200,
    warmup: int = 3,
    cpu: int | None = None,
    base_flags: Sequence[str] = (),
    python: str = sys.executable,
    enable_x64: bool = False,
    burn_in: bool = True,
    idle_gap_s: float = 2.0,
    timeout_s: float = 900.0,
    min_gain: float = MIN_GAIN,
    smoke: bool = True,
    out_dir: str | Path | None = None,
    child_env: dict[str, str] | None = None,
    log: Callable[[str], Any] = print,
) -> TuneResult:
    """Measure every applicable catalog switch against an unflagged baseline.

    Parameters
    ----------
    target : callable or str
        The function to time, or ``"module:qualname"`` naming it.  It must
        be importable by name, because the child re-imports it.  With `args`
        left out it is a *factory*: called with no arguments, it returns
        ``(fn, args)``.
    args : Sequence, optional
        Positional arguments, each an array or a Python scalar.  They are
        pickled as NumPy and converted in the child.
    candidates : Sequence, optional
        The :class:`jax2exec.Candidate` entries to measure.  The default is
        every :data:`jax2exec.tune.CATALOG` entry this host and this JAX
        accept.
    rounds : int
        Interleaved rounds.  Arms run in declared order on odd rounds and
        reversed on even ones, so a position effect changes sign between
        rounds instead of biasing the same arm every time.
    reps : int
        Timed calls per child.  Fewer than 100 leaves no p99.
    warmup : int
        Blocked calls after the compile and before the timed ones.
    cpu : int or None
        Logical CPU to pin every child to with ``taskset``; ``None`` leaves
        the scheduler alone, which is fine for a smoke run and not for a
        number anyone acts on.
    base_flags : Sequence of str
        Flags every arm carries, the baseline included.
    python : str
        Interpreter the children run under.
    enable_x64 : bool
        Set ``jax_enable_x64`` in the child before anything is built.
    burn_in : bool
        Run one discarded child first.  From cold a package sits in its
        turbo window for about the length of one child, so whichever arm
        ran first would otherwise be measured at a clock the rest never
        sees.
    idle_gap_s : float
        Seconds slept before every child.
    timeout_s : float
        Wall-clock limit of one child.
    min_gain : float
        See :data:`jax2exec.tune.MIN_GAIN`.
    smoke : bool
        Compile ``SMOKE_CODE`` under each candidate's flags first, and
        drop the ones this jaxlib refuses.  Seconds per candidate against
        minutes per round.
    out_dir : str or Path, optional
        Directory to keep the per-child records in.  Without one they go to
        a temporary directory that is removed at the end.
    child_env : dict, optional
        Environment variables set on top of every child's.
    log : callable
        Where progress goes; pass a no-op for silence.

    Returns
    -------
    TuneResult
        Read the ``valid`` field before anything else in it.
    """
    versions = _probe_versions(python)
    dropped: list[tuple[str, str]] = []
    if candidates is None:
        kept, refused = applicable(
            CATALOG, jax_version=versions.get("jax") or ""
        )
        dropped += [(entry.label, reason) for entry, reason in refused]
    else:
        kept = list(candidates)
    if smoke:
        kept, refused = _smoke(
            kept, python, cpu, base_flags, timeout_s, child_env, log
        )
        dropped += refused

    base = tuple(merge_flags(base_flags))
    arms = [Arm(label=_BASELINE, flags=base, role="baseline")]
    arms += [
        Arm(
            label=entry.label,
            flags=tuple(merge_flags(base, (entry.flag,))),
            members=(entry.label,),
            role=entry.role,
            numerics=entry.numerics,
            note=entry.note,
        )
        for entry in kept
    ]
    arms.append(
        Arm(
            label=_BASELINE_AA,
            flags=base,
            aa_of=_BASELINE,
            note="the ruler: the same configuration as the baseline",
        )
    )

    result = run_arms(
        tuple(arms),
        target,
        args,
        reference=_BASELINE,
        aa=_BASELINE_AA,
        rounds=rounds,
        reps=reps,
        warmup=warmup,
        cpu=cpu,
        base_flags=base_flags,
        python=python,
        enable_x64=enable_x64,
        burn_in=burn_in,
        idle_gap_s=idle_gap_s,
        timeout_s=timeout_s,
        min_gain=min_gain,
        out_dir=out_dir,
        child_env=child_env,
        log=log,
    )
    return dataclasses.replace(result, dropped=tuple(dropped))


def run_arms(
    arms: Sequence,
    target: Callable[..., Any] | str,
    args: Sequence[Any] | None = None,
    *,
    reference: str = _BASELINE,
    aa: str = _BASELINE_AA,
    rounds: int = 5,
    reps: int = 200,
    warmup: int = 3,
    cpu: int | None = None,
    base_flags: Sequence[str] = (),
    python: str = sys.executable,
    enable_x64: bool = False,
    burn_in: bool = True,
    idle_gap_s: float = 2.0,
    timeout_s: float = 900.0,
    min_gain: float = MIN_GAIN,
    out_dir: str | Path | None = None,
    child_env: dict[str, str] | None = None,
    log: Callable[[str], Any] = print,
) -> TuneResult:
    """Run a set of arms against each other, interleaved in rounds.

    :func:`jax2exec.tune_flags` is this with the catalog for its arms; call this one
    directly to compare arms of your own -- combination arms, for instance.

    Parameters
    ----------
    arms : Sequence
        The :class:`jax2exec.Arm` entries, in the order one round runs
        them.  One of them must be `reference`.
    target, args : Any
        As in :func:`jax2exec.tune_flags`.
    reference : str
        Label of the arm every ratio is taken against.
    aa : str
        Label of the A/A arm, whose own spread is the band.
    rounds, reps, warmup, cpu, base_flags, python, enable_x64, burn_in, \
idle_gap_s, timeout_s, min_gain, out_dir, child_env, log : Any
        As in :func:`jax2exec.tune_flags`.

    Returns
    -------
    TuneResult
        The same object :func:`jax2exec.tune_flags` returns, with ``dropped`` empty.

    Raises
    ------
    ValueError
        If the target cannot be named in a way the child could import, if an
        argument is not array-like, if two arms share a label, or if
        `reference` is not among the arms.
    """
    started = time.time()
    monotonic = time.monotonic()
    arms = tuple(arms)
    _check_arms(arms, reference)
    name = _target_name(target)
    _import_target(name)

    scratch = Path(tempfile.mkdtemp(prefix="jax2exec-tune-"))
    records_dir = (
        Path(out_dir)
        if out_dir is not None
        else Path(tempfile.mkdtemp(prefix="jax2exec-records-"))
    )
    records_dir.mkdir(parents=True, exist_ok=True)
    try:
        args_path: Path | None = None
        if args is not None:
            args_path = scratch / "args.pkl"
            _pickle_args(args, args_path)

        resolved = _resolve_versions(arms, python, log)
        driver_affinity = None if cpu is None else confine_driver(cpu)
        host = cpu_facts(cpu)
        host["driver_affinity"] = driver_affinity
        watched = list(host["siblings"] or [])

        run = _Run(
            target=name,
            factory=args is None,
            args_path=args_path,
            enable_x64=enable_x64,
            reps=reps,
            warmup=warmup,
            rounds=rounds,
            cpu=cpu,
            watched=watched,
            base_flags=tuple(base_flags),
            python=python,
            idle_gap_s=idle_gap_s,
            timeout_s=timeout_s,
            scratch=scratch,
            records_dir=records_dir,
            child_env=child_env,
            log=log,
        )

        records: list[dict[str, Any]] = []
        if burn_in:
            first = next(arm for arm in arms if arm.label == reference)
            _run_child(first, 0, 0, run, save_outputs=False)
        for index in range(1, rounds + 1):
            order = arms if index % 2 else tuple(reversed(arms))
            for slot, arm in enumerate(order, 1):
                records.append(
                    _run_child(arm, index, slot, run, save_outputs=index == 1)
                )
        deviations = _deviations(arms, reference, scratch, rounds)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if out_dir is None:
            shutil.rmtree(records_dir, ignore_errors=True)

    return _summarize(
        arms=arms,
        records=records,
        deviations=deviations,
        reference=reference,
        aa=aa,
        rounds=rounds,
        reps=reps,
        warmup=warmup,
        min_gain=min_gain,
        base_flags=tuple(base_flags),
        target=name,
        host=host,
        versions=resolved.get(python, {"jax": None, "jaxlib": None}),
        started=started,
        wall_s=time.monotonic() - monotonic,
        records_dir=str(records_dir) if out_dir is not None else None,
    )


@dataclass(frozen=True)
class _Run:
    """Everything every child of one campaign shares."""

    target: str
    factory: bool
    args_path: Path | None
    enable_x64: bool
    reps: int
    warmup: int
    rounds: int
    cpu: int | None
    watched: list[int]
    base_flags: tuple[str, ...]
    python: str
    idle_gap_s: float
    timeout_s: float
    scratch: Path
    records_dir: Path
    child_env: dict[str, str] | None
    log: Callable[[str], Any]


def _check_arms(arms: tuple[Arm, ...], reference: str) -> None:
    """Refuse an arm list that cannot be reduced to a table."""
    if not arms:
        raise ValueError("no arms to run")
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise ValueError("two arms share a label; every label names a file")
    if reference not in labels:
        raise ValueError(f"the reference arm {reference!r} is not among them")


def _target_name(target: Callable[..., Any] | str) -> str:
    """Return ``"module:qualname"`` for something the child can import."""
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(f"{target!r} has no importable name")
    # The child is a fresh interpreter: it can import a module but it cannot
    # reach a closure, and its `__main__` is this package's child module.
    if qualname.endswith("<lambda>"):
        raise ValueError("a lambda cannot be named for the child to import")
    if "<locals>" in qualname:
        raise ValueError(
            f"{qualname} is defined inside a function, so the child cannot "
            "import it; move it to module level"
        )
    if module == "__main__":
        raise ValueError(
            f"{qualname} lives in __main__, which means something different "
            "in the child; import it from a module instead"
        )
    return f"{module}:{qualname}"


def _import_target(name: str) -> Any:
    """Import a target by name here, so the child's failure happens now."""
    from ._tune_child import _resolve

    try:
        return _resolve(name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"the child could not import {name!r}: {exc}") from exc


def _pickle_args(args: Sequence[Any], path: Path) -> None:
    """Write the call arguments as NumPy arrays."""
    prepared = []
    for index, value in enumerate(args):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"argument {index} is neither array-like nor a scalar: {exc}"
            ) from exc
        if array.dtype == object:
            raise ValueError(
                f"argument {index} is neither array-like nor a scalar: "
                f"{type(value).__name__}"
            )
        prepared.append(array)
    with open(path, "wb") as handle:
        pickle.dump(tuple(prepared), handle)


def _probe_versions(python: str) -> dict[str, str | None]:
    """Ask one interpreter which JAX and jaxlib it resolved to."""
    code = (
        "import json, jax, jaxlib; "
        "print(json.dumps({'jax': jax.__version__, "
        "'jaxlib': jaxlib.__version__}))"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"jax": None, "jaxlib": None}


def _resolve_versions(
    arms: tuple[Arm, ...], python: str, log: Callable[[str], Any]
) -> dict[str, dict[str, str | None]]:
    """Probe every distinct interpreter once and hold the arms to it."""
    interpreters = sorted({str(arm.python or python) for arm in arms})
    resolved = {name: _probe_versions(name) for name in interpreters}
    for name in interpreters:
        log(
            f"# {name}: jax {resolved[name]['jax']}, "
            f"jaxlib {resolved[name]['jaxlib']}"
        )
    for arm in arms:
        if arm.expected_jax is None:
            continue
        found = resolved[str(arm.python or python)]["jax"]
        if found != arm.expected_jax:
            raise ValueError(
                f"arm {arm.label!r} expects jax {arm.expected_jax} but its "
                f"interpreter resolved {found}"
            )
    return resolved


def _child_argv(python: str, cpu: int | None, spec_path: Path) -> list[str]:
    """Build the command line of one child.

    Parameters
    ----------
    python : str
        Interpreter to run.
    cpu : int or None
        Logical CPU to pin to, or ``None`` for no pinning.
    spec_path : Path
        The JSON spec the child reads.

    Returns
    -------
    list of str
        ``taskset -c <cpu>`` in front when there is one.  The driver's own
        affinity excludes that core and does not stop the child reaching it:
        Linux affinity is not hierarchical.
    """
    prefix = [] if cpu is None else ["taskset", "-c", str(cpu)]
    return [
        *prefix,
        str(python),
        "-m",
        "jax2exec._tune_child",
        str(spec_path),
    ]


def _child_environment(
    arm: Arm, base_flags: Sequence[str], child_env: dict[str, str] | None
) -> dict[str, str]:
    """Return the full environment one child runs under."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = " ".join(merge_flags(base_flags, arm.flags))
    if child_env:
        env.update(child_env)
    return env


def _classify(returncode: int, stderr: str, flagged: bool) -> str:
    """Name what happened to a child."""
    if returncode == 0:
        return "ok"
    named = any(pattern in stderr for pattern in REJECT_PATTERNS)
    # A refused flag kills the process before any Python frame exists, by
    # SIGABRT on older jaxlibs and exit 1 on newer ones, so exit 1 needs the
    # stderr line to tell a bad flag from a failed call.  Guarded on the
    # arm carrying flags at all: an unflagged baseline exiting 1 is a
    # failure and naming it "rejected" would hide a broken target.
    rejected = returncode == -signal.SIGABRT or (returncode == 1 and named)
    return "rejected" if flagged and rejected else "failed"


def _cap(text: str) -> str:
    """Cut a captured stream, saying how much was cut."""
    if len(text) <= _STREAM_CAP:
        return text
    dropped = len(text) - _STREAM_CAP
    return f"[... {dropped} earlier characters cut ...]\n" + text[-_STREAM_CAP:]


def _run_child(
    arm: Arm,
    round_index: int,
    slot: int,
    run: _Run,
    *,
    save_outputs: bool,
) -> dict[str, Any]:
    """Run one arm once and record everything about the run."""
    stem = f"{arm.label}_r{round_index}"
    spec_path = run.scratch / f"{stem}.spec.json"
    child_json = run.scratch / f"{stem}.child.json"
    npz_path = run.scratch / f"{stem}.npz" if save_outputs else None
    spec = {
        "target": run.target,
        "factory": run.factory,
        "args": None if run.args_path is None else str(run.args_path),
        "enable_x64": run.enable_x64,
        "reps": run.reps,
        "warmup": run.warmup,
        "out": str(child_json),
        "save_outputs": None if npz_path is None else str(npz_path),
        "label": arm.label,
        "round": round_index,
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    python = str(arm.python or run.python)
    argv = _child_argv(python, run.cpu, spec_path)
    env = _child_environment(arm, run.base_flags, run.child_env)
    tag = "burn-in" if round_index == 0 else f"round {round_index}"
    run.log(f"[{tag}] slot {slot} {arm.label}: sleeping {run.idle_gap_s:g} s")
    time.sleep(run.idle_gap_s)

    load = loadavg_1min()
    before = cpu_jiffies(run.watched)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=run.timeout_s,
            check=False,
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        status = _classify(returncode, stderr, bool(arm.flags))
    except subprocess.TimeoutExpired:
        returncode, stdout, stderr, status = -1, "", "timed out", "timeout"
    except OSError as exc:
        # a missing taskset or a deleted interpreter is one failed arm, not
        # a dead campaign
        returncode, stdout, status = -1, "", "failed"
        stderr = f"could not spawn the child: {exc}"
    wall_s = time.monotonic() - started
    occupancy = cpu_occupancy(before, cpu_jiffies(run.watched), wall_s)

    bench: dict[str, Any] | None = None
    try:
        if child_json.exists():
            bench = json.loads(child_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        stderr += f"\ncould not read {child_json}: {exc}"
    if isinstance(bench, dict) and bench.get("error"):
        status = "failed"
        stderr += f"\n{bench['error']}"
    elif status == "ok" and bench is None:
        status = "failed"
        stderr += "\nthe child exited 0 but wrote no readable JSON"

    affinity = child_affinity(bench)
    sibling = max(
        (
            busy
            for cpu, busy in occupancy.items()
            if run.cpu is None or int(cpu) != run.cpu
        ),
        default=None,
    )
    record = {
        "arm": arm.label,
        "round": round_index,
        "slot": slot,
        "status": status,
        "returncode": returncode,
        "wall_s": wall_s,
        "argv": argv,
        "xla_flags": env["XLA_FLAGS"],
        "loadavg_1min": load,
        "loadavg_flagged": load is not None and load > LOADAVG_LIMIT,
        "cpu_busy_fraction": occupancy,
        "sibling_busy_fraction": sibling,
        "sibling_flagged": (
            sibling is not None and sibling > SIBLING_BUSY_LIMIT
        ),
        "child_affinity": affinity,
        "pin_flagged": (
            run.cpu is not None
            and affinity is not None
            and affinity != [run.cpu]
        ),
        "stdout": _cap(stdout),
        "stderr": _cap(stderr),
        "bench": bench,
    }
    (run.records_dir / f"{stem}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    median = (bench or {}).get("median_ms")
    shown = f", p50 {median:.3f} ms" if isinstance(median, float) else ""
    run.log(f"    -> {status} in {wall_s:.1f} s, load {load}{shown}")
    return record


def _smoke(
    candidates: Sequence,
    python: str,
    cpu: int | None,
    base_flags: Sequence[str],
    timeout_s: float,
    child_env: dict[str, str] | None,
    log: Callable[[str], Any],
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Drop the candidates this jaxlib refuses, before any round is run."""
    kept: list[Candidate] = []
    dropped: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate.kind == "extra-option":
            # XLA parses the outer flag and hands the map to the backend
            # untouched, so a pass here would not be evidence of anything.
            log(f"# {candidate.label}: extra-option, cannot be probed")
            kept.append(candidate)
            continue
        status, tail = _smoke_one(
            candidate, python, cpu, base_flags, timeout_s, child_env
        )
        if status == "ok":
            kept.append(candidate)
        else:
            dropped.append((candidate.label, f"smoke {status}: {tail}"))
            log(f"# {candidate.label}: dropped, smoke {status}")
    return kept, dropped


def _smoke_one(
    candidate: Candidate,
    python: str,
    cpu: int | None,
    base_flags: Sequence[str],
    timeout_s: float,
    child_env: dict[str, str] | None,
) -> tuple[str, str]:
    """Compile one tiny jitted program under a candidate's flags."""
    arm = Arm(label=candidate.label, flags=(candidate.flag,))
    env = _child_environment(arm, base_flags, child_env)
    prefix = [] if cpu is None else ["taskset", "-c", str(cpu)]
    try:
        proc = subprocess.run(
            [*prefix, str(python), "-c", SMOKE_CODE],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        returncode, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        returncode, stderr = -1, "timed out"
    except OSError as exc:
        returncode, stderr = -1, str(exc)
    status = _classify(returncode, stderr, True)
    lines = stderr.strip().splitlines()
    return status, lines[-1][:120] if lines else ""


def _deviations(
    arms: tuple[Arm, ...],
    reference: str,
    scratch: Path,
    rounds: int,
) -> dict[str, tuple[float | None, float | None, bool]]:
    """Compare every arm's first-round outputs with the reference's.

    A flag that makes a program faster by computing something else is the
    one failure this protocol could otherwise report as a win.
    """
    out: dict[str, tuple[float | None, float | None, bool]] = {}
    if rounds < 1:
        return out
    base = _load_leaves(scratch / f"{reference}_r1.npz")
    for arm in arms:
        leaves = _load_leaves(scratch / f"{arm.label}_r1.npz")
        if base is None or leaves is None:
            out[arm.label] = (None, None, False)
            continue
        if len(leaves) != len(base) or any(
            a.shape != b.shape or a.dtype != b.dtype
            for a, b in zip(leaves, base, strict=True)
        ):
            out[arm.label] = (None, None, True)
            continue
        abs_dev = 0.0
        rel_dev = 0.0
        for a, b in zip(leaves, base, strict=True):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            abs_dev = max(abs_dev, float(diff.max(initial=0.0)))
            scaled = diff / (np.abs(b.astype(np.float64)) + 1e-300)
            rel_dev = max(rel_dev, float(scaled.max(initial=0.0)))
        out[arm.label] = (abs_dev, rel_dev, False)
    for path in scratch.glob("*.npz"):
        path.unlink(missing_ok=True)
    return out


def _load_leaves(path: Path) -> list[np.ndarray] | None:
    """Read the leaves a child saved, or ``None`` when it saved none."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            names = sorted(
                (key for key in data.files if key.startswith("leaf_")),
                key=lambda key: int(key.partition("_")[2]),
            )
            return [np.asarray(data[key]) for key in names]
    except (OSError, ValueError):
        return None


def _summarize(
    *,
    arms: tuple[Arm, ...],
    records: list[dict[str, Any]],
    deviations: dict[str, tuple[float | None, float | None, bool]],
    reference: str,
    aa: str,
    rounds: int,
    reps: int,
    warmup: int,
    min_gain: float,
    base_flags: tuple[str, ...],
    target: str,
    host: dict[str, Any],
    versions: dict[str, str | None],
    started: float,
    wall_s: float,
    records_dir: str | None,
) -> TuneResult:
    """Reduce the per-child records to a band, a verdict per arm, winners."""
    by_arm: dict[str, list[dict[str, Any]]] = {arm.label: [] for arm in arms}
    for record in records:
        by_arm[record["arm"]].append(record)
    ref_median = _series(by_arm[reference], "median_ms")

    ratios: dict[str, dict[int, float]] = {}
    for arm in arms:
        own = _series(by_arm[arm.label], "median_ms")
        ratios[arm.label] = {
            index: own[index] / ref_median[index]
            for index in sorted(own)
            if ref_median.get(index)
        }
    band: tuple[float, float] | None
    try:
        band = aa_band(list(ratios.get(aa, {}).values()))
    except ValueError:
        band = None

    results: list[ArmResult] = []
    for arm in arms:
        rows = by_arm[arm.label]
        abs_dev, rel_dev, mismatched = deviations.get(
            arm.label, (None, None, False)
        )
        compile_s = _series(rows, "compile_s")
        values = list(ratios[arm.label].values())
        if arm.label == reference:
            decision = "reference"
        elif band is None:
            decision = "no A/A band"
        else:
            decision = verdict(values, band[0], band[1], rounds)
        results.append(
            ArmResult(
                label=arm.label,
                flags=tuple(arm.flags),
                members=tuple(arm.members),
                role=arm.role,
                numerics=arm.numerics,
                aa_of=arm.aa_of,
                status=_arm_status(rows, mismatched),
                median_ms=_series(rows, "median_ms"),
                p95_ms=_series(rows, "p95_ms"),
                p99_ms=_series(rows, "p99_ms"),
                max_ms=_series(rows, "max_ms"),
                compile_s=compile_s,
                ratio=ratios[arm.label],
                median_ratio=statistics.median(values) if values else None,
                verdict=decision,
                span=(min(values), max(values)) if values else None,
                rounds_ok=sum(1 for row in rows if row["status"] == "ok"),
                max_abs_dev=abs_dev,
                max_rel_dev=rel_dev,
                hygiene=_hygiene(rows, compile_s),
                note=arm.note,
            )
        )

    winners = tuple(
        result.label
        for result in sorted(
            (
                result
                for result in results
                if result.role == "candidate"
                and result.aa_of is None
                and result.verdict == "faster"
                and result.median_ratio is not None
                and result.median_ratio <= 1.0 - min_gain
                and result.status == "ok"
            ),
            key=lambda result: result.median_ratio or 1.0,
        )
    )
    valid, reason = _validity(results, aa)
    return TuneResult(
        schema=RESULT_SCHEMA,
        host=host,
        jax=versions.get("jax"),
        jaxlib=versions.get("jaxlib"),
        reference=reference,
        aa=aa,
        band=band,
        rounds=rounds,
        reps=reps,
        warmup=warmup,
        min_gain=min_gain,
        base_flags=base_flags,
        target=target,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        wall_s=wall_s,
        arms=tuple(results),
        dropped=(),
        winners=winners,
        valid=valid,
        invalid_reason=reason,
        records_dir=records_dir,
    )


def _series(rows: list[dict[str, Any]], field: str) -> dict[int, float]:
    """Round number -> one field of that round's child JSON."""
    out: dict[int, float] = {}
    for row in rows:
        bench = row.get("bench")
        value = bench.get(field) if isinstance(bench, dict) else None
        if isinstance(value, (int, float)):
            out[int(row["round"])] = float(value)
    return out


def _arm_status(rows: list[dict[str, Any]], mismatched: bool) -> str:
    """Reduce an arm's per-round statuses to one word."""
    if mismatched:
        return "mismatch"
    kinds = {row["status"] for row in rows}
    if not kinds:
        return "failed"
    if kinds == {"ok"}:
        return "ok"
    if "ok" in kinds:
        return "partial"
    for kind in ("rejected", "timeout", "failed"):
        if kind in kinds:
            return kind
    return "failed"


def _hygiene(
    rows: list[dict[str, Any]], compile_s: dict[int, float]
) -> tuple[str, ...]:
    """Name every limit any of an arm's rounds tripped."""
    flags = []
    if any(row["loadavg_flagged"] for row in rows):
        flags.append("loadavg")
    if any(row["sibling_flagged"] for row in rows):
        flags.append("sibling")
    if any(row["pin_flagged"] for row in rows):
        flags.append("affinity")
    values = list(compile_s.values())
    if len(values) > 1:
        middle = statistics.median(values)
        if middle > 0 and any(
            abs(value - middle) / middle > COMPILE_DRIFT_LIMIT
            for value in values
        ):
            flags.append("compile-drift")
    return tuple(flags)


def _validity(results: Sequence[ArmResult], aa: str) -> tuple[bool, str | None]:
    """Whether this campaign's own instruments worked."""
    for result in results:
        if result.label == aa and result.status != "ok":
            return False, (
                f"the A/A arm `{aa}` read `{result.status}`, so the campaign "
                "has no ruler"
            )
    for result in results:
        if result.role == "control" and result.verdict != "slower":
            return False, (
                f"the control arm `{result.label}` read `{result.verdict}` "
                "and not `slower`; a protocol that cannot see a known loss "
                "cannot see a small win either"
            )
    return True, None


########################
#       RENDERING      #
########################

_COLUMNS = (
    "arm",
    "flags",
    "p50 ratio",
    "span",
    "verdict",
    "p50 ms",
    "p95 ms",
    "p99 ms",
    "max ms",
    "compile s",
    "dev",
    "notes",
)


def _num(value: float | None, digits: int = 3) -> str:
    """Format a number for a table cell, or ``-`` for nothing."""
    return "-" if value is None else f"{value:.{digits}f}"


def _mid(series: dict[int, float]) -> float | None:
    """Median over the rounds that produced a value."""
    values = list(series.values())
    return statistics.median(values) if values else None


def _tail_disagrees(
    result: ArmResult, ref_p99: dict[int, float], band: tuple[float, float]
) -> bool:
    """Whether the p99 ratio and the p50 ratio point opposite ways.

    The figure of merit for a real-time runtime is the tail, so a flag whose
    median improves while its p99 gets worse is the one result that must not
    be read off the p50 column alone.
    """
    if result.median_ratio is None:
        return False
    tails = [
        result.p99_ms[index] / ref_p99[index]
        for index in sorted(result.p99_ms)
        if ref_p99.get(index)
    ]
    if not tails:
        return False
    tail = statistics.median(tails)
    width = band[1] - 1.0
    opposed = (result.median_ratio - 1.0) * (tail - 1.0) < 0
    return opposed and abs(tail - result.median_ratio) > width


def _render_table(result: TuneResult) -> str:
    """Render a whole campaign as markdown."""
    ref_p99 = {
        index: value
        for arm in result.arms
        if arm.label == result.reference
        for index, value in arm.p99_ms.items()
    }
    caveat = False
    rows = []
    for arm in result.arms:
        notes = list(arm.hygiene)
        if any(token.startswith(EXTRA_OPTIONS) for token in arm.flags):
            notes.append("dagger")
            caveat = True
        if result.band is not None and _tail_disagrees(
            arm, ref_p99, result.band
        ):
            notes.append("tail^")
        if arm.status != "ok":
            notes.append(arm.status)
        if arm.note:
            notes.append(arm.note)
        span = (
            "-" if arm.span is None else f"{arm.span[0]:.3f}-{arm.span[1]:.3f}"
        )
        dev = (
            "-"
            if arm.max_abs_dev is None
            else f"{arm.max_abs_dev:.2e}/{arm.max_rel_dev:.2e}"
        )
        rows.append(
            (
                arm.label,
                " ".join(arm.flags) or "-",
                _num(arm.median_ratio),
                span,
                arm.verdict,
                _num(_mid(arm.median_ms)),
                _num(_mid(arm.p95_ms)),
                _num(_mid(arm.p99_ms)),
                _num(_mid(arm.max_ms)),
                _num(_mid(arm.compile_s), 1),
                dev,
                "; ".join(notes).replace("dagger", "†"),
            )
        )

    widths = [
        max(len(_COLUMNS[index]), *(len(row[index]) for row in rows))
        if rows
        else len(_COLUMNS[index])
        for index in range(len(_COLUMNS))
    ]
    header = zip(_COLUMNS, widths, strict=True)
    lines = [
        "| " + " | ".join(name.ljust(w) for name, w in header) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                cell.ljust(width)
                for cell, width in zip(row, widths, strict=True)
            )
            + " |"
        )
    lines.append("")
    if result.band is None:
        lines.append(
            f"A/A band: none -- the arm `{result.aa}` produced no ratio, so "
            "no verdict was taken."
        )
    else:
        lines.append(
            f"A/A band: {result.band[0]:.4f} - {result.band[1]:.4f}, from "
            f"the arm `{result.aa}` over {result.rounds} rounds."
        )
    if caveat:
        lines.append(
            "† an `--xla_backend_extra_options` key cannot be validated: "
            "XLA parses the outer flag and hands the map to the backend "
            "untouched, so acceptance is not evidence that it took effect."
        )
    if not result.valid:
        lines.append(f"INVALID: {result.invalid_reason}")
    return "\n".join(lines)
