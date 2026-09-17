"""The XLA code-generation switches worth measuring, and flag arithmetic.

Pure data and string handling: no JAX, no subprocess, so the catalog can be
read and filtered on a machine that has neither.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ._isa import host_isa_level, isa_supports

__all__ = [
    "CATALOG",
    "Candidate",
    "applicable",
    "drop_flag",
    "flag_key",
    "merge_flags",
    "split_flags",
    "version_tuple",
]

#: The one XLA flag whose value is a map rather than a scalar.  Two arms
#: setting different inner keys of it are not in conflict, so it is the one
#: flag :func:`jax2exec.tune.merge_flags` merges key-wise instead of replacing.
EXTRA_OPTIONS = "--xla_backend_extra_options"


def version_tuple(text: str) -> tuple[int, ...]:
    """Leading integer components of a version string.

    ``packaging`` is not a dependency of this package and a JAX nightly is
    spelled ``0.11.2.dev20260905``, so the comparison keeps the leading
    integers and stops at the first component that is not one.

    Parameters
    ----------
    text : str
        A version, e.g. ``"0.11.1"``.

    Returns
    -------
    tuple of int
        ``(0, 11, 1)``.  Empty when nothing leads with a number, which
        compares below every real release.
    """
    parts: list[int] = []
    for part in str(text).split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


@dataclass(frozen=True)
class Candidate:
    """One switch the tuner is willing to measure.

    Parameters
    ----------
    label : str
        Arm name.  It names the child's record file, the table row and,
        joined with ``+``, a combination arm.
    flag : str
        The ``XLA_FLAGS`` token, written exactly as it must be passed.  A
        plain flag and an ``--xla_backend_extra_options`` inner key are not
        interchangeable: one spelled as the other is either refused outright
        or ignored in silence, so nothing here ever re-spells a flag.
    kind : str
        ``"flag"`` or ``"extra-option"``.  The second kind cannot be
        validated -- XLA parses the outer flag and hands the map to the
        backend untouched -- so its rows carry a caveat.
    numerics : str
        ``"exact"`` if the switch only changes how the same arithmetic is
        emitted, ``"inexact"`` if it changes the arithmetic itself.
    role : str
        ``"candidate"`` to tune with, ``"control"`` to prove the protocol
        can see a loss on this host, ``"ceiling"`` to bound what any amount
        of arithmetic conservatism could buy.
    min_isa : str or None
        psABI level the host must implement, e.g. ``"x86-64-v4"``.
    since : str or None
        First JAX version that accepts the flag, inclusive.
    until : str or None
        First JAX version that no longer accepts it, exclusive.
    note : str
        One line of why the entry is here.
    """

    label: str
    flag: str
    kind: str = "flag"
    numerics: str = "exact"
    role: str = "candidate"
    min_isa: str | None = None
    since: str | None = None
    until: str | None = None
    note: str = ""

    @property
    def requires(self) -> str:
        """Return what this candidate needs, in words.

        Returns
        -------
        str
            One line naming every constraint, or ``"any host, any pin"``
            when there is none.  A table prints this beside a dropped
            candidate, which is why the constraints are data rather than a
            predicate: a function cannot be printed.
        """
        parts = []
        if self.min_isa:
            parts.append(f"host at {self.min_isa} or better")
        if self.since:
            parts.append(f"jax >= {self.since}")
        if self.until:
            parts.append(f"jax < {self.until}")
        return "; ".join(parts) if parts else "any host, any pin"


#: Every switch worth an arm, with the reason that put it there.  Every
#: string was smoke-tested against a real jaxlib rather than read off a
#: header.  Order matters in one place only: a combination arm's flags are
#: merged in catalog order, so the same set always produces the same arm.
CATALOG: tuple[Candidate, ...] = (
    Candidate(
        label="vector-width-128",
        flag="--xla_cpu_prefer_vector_width=128",
        note=(
            "The largest single win found on the hosts tried so far, and "
            "the reason the vector-width ladder is walked at all."
        ),
    ),
    Candidate(
        label="vector-width-64",
        flag="--xla_cpu_prefer_vector_width=64",
        note=(
            "The next step down; the flag is an unvalidated int32, so the "
            "ladder has to be walked rather than reasoned about."
        ),
    ),
    Candidate(
        label="vector-width-512",
        flag="--xla_cpu_prefer_vector_width=512",
        min_isa="x86-64-v4",
        note=(
            "Preferring wider than the machine has is a lie to LLVM, not a "
            "speedup: without AVX512 there are no 512-bit registers to "
            "prefer.  The flag is accepted on an AVX2 host anyway, so "
            "min_isa and not the smoke probe has to drop it."
        ),
    ),
    Candidate(
        label="max-isa-avx2",
        flag="--xla_cpu_max_isa=AVX2",
        min_isa="x86-64-v4",
        note=(
            "Capping code generation at AVX2 is a no-op by construction on "
            "a host whose ceiling is AVX2 already."
        ),
    ),
    Candidate(
        label="max-isa-sse4",
        flag="--xla_cpu_max_isa=SSE4_2",
        numerics="inexact",
        note=(
            "Caps code generation at SSE4.2, which is not a width knob.  It "
            "is inexact because SSE4.2 predates FMA: capping there stops "
            "the multiply-add contraction and changes the rounding.  Use "
            "the vector-width arms for the width effect on its own; they "
            "keep FMA."
        ),
    ),
    Candidate(
        label="legacy-emitters",
        flag="--xla_cpu_use_fusion_emitters=false",
        until="0.11.1",
        note=(
            "Removed at 0.11.1 (measured: `Unknown flag in XLA_FLAGS`, rc "
            "1); accepted at 0.11.0."
        ),
    ),
    Candidate(
        label="new-xtile",
        flag="--xla_cpu_use_new_xtile_lowering=true",
        since="0.11.1",
        note="Added at 0.11.1 (measured: rejected at 0.11.0).",
    ),
    Candidate(
        label="no-tiling-propagation",
        flag="--xla_cpu_experimental_enable_tiling_propagation=false",
        note=(
            "Accepted at both 0.11.0 and 0.11.1, and present in 0.11.1's "
            "own --help: it gets no version constraint."
        ),
    ),
    Candidate(
        label="ynn-dot-only",
        flag="--xla_cpu_experimental_ynn_fusion_type=INDIVIDUAL_DOT",
        note=(
            "Restricts ynn fusion to individual dots; the HLO census shows "
            "fewer library fusions, so the flag reaches code generation."
        ),
    ),
    Candidate(
        label="ynn-none",
        flag="--xla_cpu_experimental_ynn_fusion_type=",
        note=(
            "Switches the family off entirely.  0.11.1's help calls the "
            "default list empty, which would make this a duplicate "
            "baseline; the HLO census disagrees, so the help text is not "
            "authoritative about defaults."
        ),
    ),
    Candidate(
        label="no-xnnpack",
        flag="--xla_cpu_use_xnnpack=false",
        note=(
            "XNNPACK is on by default, and a program of small kernels may "
            "pay for a dispatch it cannot amortize."
        ),
    ),
    Candidate(
        label="xnn-graph-greedy",
        flag=(
            "--xla_cpu_experimental_xnn_graph_fusion_mode="
            "XNN_GRAPH_FUSION_MODE_GREEDY"
        ),
        note=(
            "The pass is disabled by default; greedy extraction is the "
            "only untried direction in the XNNPACK family."
        ),
    ),
    Candidate(
        label="sched-memory",
        flag="--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED",
        note=(
            "Measured slower rather than null on the hosts tried so far; "
            "kept because that is a result, and because it is a second "
            "de-facto control."
        ),
    ),
    Candidate(
        label="sched-concurrency",
        flag=(
            "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED"
        ),
        note=(
            "The third enum value.  Its own flag "
            "--xla_cpu_enable_concurrency_optimized_scheduler is deprecated "
            "and redundant with this, so only this spelling is listed."
        ),
    ),
    Candidate(
        label="codegen-split-1",
        flag="--xla_cpu_parallel_codegen_split_count=1",
        note=(
            "Default 32; it changes compile time far more than run time, "
            "so read its row with the compile-drift marker."
        ),
    ),
    Candidate(
        label="region-copy",
        flag="--xla_cpu_copy_insertion_use_region_analysis=true",
        note=(
            "Fewer copies around a program's `while` loops is the one "
            "plausible story for this switch."
        ),
    ),
    Candidate(
        label="opt-level-2",
        flag="--xla_backend_optimization_level=2",
        note="Default 3; a cheaper pipeline sometimes emits better loops.",
    ),
    Candidate(
        label="no-expensive-passes",
        flag="--xla_llvm_disable_expensive_passes=true",
        note="The same hypothesis, from the LLVM side.",
    ),
    Candidate(
        label="opt-preset-fast-runtime",
        flag="--xla_cpu_opt_preset=CPU_OPT_PRESET_FAST_RUNTIME",
        note=(
            "The preset XLA itself advertises for runtime.  Spelled from "
            "the binary: --xla_cpu_experimental_optimization_preset does "
            "not exist at either pin."
        ),
    ),
    Candidate(
        label="no-slp",
        flag="--xla_backend_extra_options=xla_cpu_disable_slp_vectorizer=1",
        kind="extra-option",
        note=(
            "Measured null once; kept because the vector-width result says "
            "vectorization is where the money is."
        ),
    ),
    Candidate(
        label="no-unroll",
        flag="--xla_backend_extra_options=xla_cpu_disable_loop_unrolling=1",
        kind="extra-option",
        role="control",
        note=(
            "The control, and the one switch whose direction is not in "
            "question: refusing to unroll a tight loop does not make it "
            "faster on any machine.  A session where it does not read "
            "slower is invalid, not null -- a protocol that cannot see "
            "unrolling being switched off cannot see a small win either.  "
            "It doubles as the proof that an extra-option key reaches code "
            "generation, which the outer flag's acceptance never shows."
        ),
    ),
    Candidate(
        label="disable-tiled-emitter",
        flag="--xla_backend_extra_options=xla_cpu_disable_tiled_emitter=1",
        kind="extra-option",
        note=(
            "The HLO census shows fewer fusions, so it reaches code "
            "generation; the time it bought was null."
        ),
    ),
    Candidate(
        label="disable-new-fusion",
        flag=(
            "--xla_backend_extra_options=xla_cpu_disable_new_fusion_emitters=1"
        ),
        kind="extra-option",
        note=(
            "The HLO census shows far fewer fusions, so it reaches code "
            "generation; the time it bought was null."
        ),
    ),
    Candidate(
        label="optimize-for-size",
        flag="--xla_backend_extra_options=xla_cpu_optimize_for_size=1",
        kind="extra-option",
        note=(
            "A kernel that fits in L1i is a different machine from one "
            "that does not."
        ),
    ),
    Candidate(
        label="small-while-0",
        flag=(
            "--xla_backend_extra_options="
            "xla_cpu_small_while_loop_byte_threshold=0"
        ),
        kind="extra-option",
        note=(
            "Directly on the mechanism of jax#40101: this threshold decides "
            "how a small-carry `while` is emitted."
        ),
    ),
    Candidate(
        label="small-while-64k",
        flag=(
            "--xla_backend_extra_options="
            "xla_cpu_small_while_loop_byte_threshold=65536"
        ),
        kind="extra-option",
        note="The other end of the same knob.",
    ),
    Candidate(
        label="fast-math",
        flag="--xla_cpu_enable_fast_math=true",
        numerics="inexact",
        role="ceiling",
        note=(
            "Bounds what arithmetic conservatism costs.  It changes the "
            "results, so it is a ceiling and never a setting to ship."
        ),
    ),
    Candidate(
        label="no-fast-min-max",
        flag="--xla_cpu_enable_fast_min_max=false",
        numerics="inexact",
        note=(
            "A numerics knob listed so a reader does not have to wonder "
            "whether it was considered."
        ),
    ),
    Candidate(
        label="no-platform-math",
        flag="--xla_cpu_enable_platform_dependent_math=false",
        numerics="inexact",
        note=(
            "The same; its own help says it trades speed for cross-CPU "
            "consistency."
        ),
    ),
)


def split_flags(text: str | None) -> tuple[str, ...]:
    """Split an ``XLA_FLAGS`` string into tokens.

    Parameters
    ----------
    text : str or None
        The variable's value, or ``None`` when it is unset.

    Returns
    -------
    tuple of str
        The whitespace-separated tokens; empty for ``None`` or ``""``.
    """
    return tuple(text.split()) if text else ()


def flag_key(flag: str) -> str:
    """Return the name a flag token sets.

    Parameters
    ----------
    flag : str
        An ``XLA_FLAGS`` token.

    Returns
    -------
    str
        ``"--xla_cpu_prefer_vector_width"``, or the *inner* key for an
        ``--xla_backend_extra_options`` token, because that flag is a map
        and two arms setting different keys of it are not in conflict.
    """
    prefix = f"{EXTRA_OPTIONS}="
    if flag.startswith(prefix):
        return flag[len(prefix) :].partition("=")[0]
    return flag.partition("=")[0]


def _split_extra(token: str) -> dict[str, str | None]:
    """Parse ``--xla_backend_extra_options=k=v,k2`` into its inner map."""
    _, _, value = token.partition("=")
    inner: dict[str, str | None] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        inner[key] = val if sep else None
    return inner


def _join_extra(inner: dict[str, str | None]) -> str:
    """Render an inner map back into one ``--xla_backend_extra_options``."""
    parts = [k if v is None else f"{k}={v}" for k, v in inner.items()]
    return f"{EXTRA_OPTIONS}={','.join(parts)}"


def merge_flags(*groups: str | Iterable[str] | None) -> tuple[str, ...]:
    """Combine XLA flag groups, a later group winning per flag.

    Parameters
    ----------
    *groups : str or Iterable of str or None
        Each an ``XLA_FLAGS``-style string or an iterable of tokens.
        ``None`` and empty groups are skipped.

    Returns
    -------
    tuple of str
        The merged tokens, each flag appearing once, in the order it was
        first seen.  Two groups that set the same flag to different values
        leave the later value; two groups that set *different inner keys*
        of ``--xla_backend_extra_options`` leave both, because replacing
        that map wholesale would silently drop a key.

    Notes
    -----
    A token with no ``=`` is a flag in its own right and is kept as it
    stands, and a token that is not a flag at all is passed through rather
    than dropped: this is not a validator, and XLA will say so itself.
    """
    merged: dict[str, str] = {}
    extra: dict[str, str | None] = {}
    for group in groups:
        if group is None:
            continue
        tokens = split_flags(group) if isinstance(group, str) else tuple(group)
        for token in tokens:
            if token.startswith(f"{EXTRA_OPTIONS}="):
                extra.update(_split_extra(token))
                merged.setdefault(EXTRA_OPTIONS, "")
            else:
                merged[token.partition("=")[0]] = token
    return tuple(
        _join_extra(extra) if name == EXTRA_OPTIONS else token
        for name, token in merged.items()
    )


def drop_flag(flags: Iterable[str], key: str) -> tuple[str, ...]:
    """Remove one flag by name, whatever value it was given.

    Parameters
    ----------
    flags : Iterable of str
        Flag tokens.
    key : str
        The name to remove, as :func:`jax2exec.tune.flag_key` spells it.

    Returns
    -------
    tuple of str
        The tokens without that flag.
    """
    return tuple(token for token in flags if flag_key(token) != key)


def applicable(
    candidates: Iterable,
    *,
    isa_level: str | None = None,
    jax_version: str = "",
) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Split a catalog into what this host may measure and what it may not.

    Parameters
    ----------
    candidates : Iterable
        The :class:`Candidate` entries to consider.
    isa_level : str or None
        This host's psABI level; ``None`` asks
        this package's ISA detection.
    jax_version : str
        The JAX release that will run, as the child reports it.  An empty
        string compares below every release, so a version-constrained
        candidate is dropped rather than guessed at.

    Returns
    -------
    kept : list of Candidate
        Candidates a child will be spawned for.
    dropped : list of tuple
        Each refused candidate with the reason, ready to print.
    """
    host = host_isa_level() if isa_level is None else isa_level
    pin = version_tuple(jax_version)
    kept: list[Candidate] = []
    dropped: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        reason = _refusal(candidate, host, pin, jax_version)
        if reason is None:
            kept.append(candidate)
        else:
            dropped.append((candidate, reason))
    return kept, dropped


def _refusal(
    candidate: Candidate,
    host: str,
    pin: tuple[int, ...],
    jax_version: str,
) -> str | None:
    """Return why this host may not measure a candidate, or ``None``."""
    if candidate.min_isa is not None:
        supported = isa_supports(host, candidate.min_isa)
        if supported is None:
            return (
                f"needs {candidate.min_isa}, which cannot be compared with "
                f"this host's {host}"
            )
        if not supported:
            return f"needs {candidate.min_isa}; this host is {host}"
    if candidate.since is not None and pin < version_tuple(candidate.since):
        return (
            f"needs jax >= {candidate.since}; this run has "
            f"{jax_version or 'an unknown version'}"
        )
    if candidate.until is not None and pin >= version_tuple(candidate.until):
        return (
            f"gone at jax {candidate.until}; this run has "
            f"{jax_version or 'an unknown version'}"
        )
    return None


#: Label -> candidate, for the callers that resolve a label back to a flag.
BY_LABEL: dict[str, Candidate] = {entry.label: entry for entry in CATALOG}
