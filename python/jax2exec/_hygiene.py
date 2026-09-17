"""What the driver reads about the machine it is measuring on.

Linux-first and tolerant: every reader returns ``None`` for a file this
kernel does not have, because a missing ``/sys`` entry is a fact to record
rather than a reason to abandon a campaign.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from ._isa import cpu_model, host_isa_level

__all__ = [
    "child_affinity",
    "confine_driver",
    "cpu_facts",
    "cpu_jiffies",
    "cpu_occupancy",
    "cpu_siblings",
    "loadavg_1min",
    "read_first",
]


def read_first(path: str) -> str | None:
    """Return the first line of a file, stripped.

    Parameters
    ----------
    path : str
        File to read.

    Returns
    -------
    str or None
        The first line, or ``None`` when the file cannot be read.
    """
    try:
        return Path(path).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def loadavg_1min() -> float | None:
    """Return the one-minute load average.

    Returns
    -------
    float or None
        First field of ``/proc/loadavg``, or ``None`` if it cannot be read.
        A busy machine does not add noise to a latency measurement, it
        invalidates it, so this is read before every child.
    """
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def cpu_siblings(cpu: int) -> list[int]:
    """Return the logical CPUs sharing a physical core with `cpu`.

    Parameters
    ----------
    cpu : int
        Logical CPU.

    Returns
    -------
    list of int
        `cpu` and its SMT siblings, sorted; ``[cpu]`` alone when the
        topology cannot be read.
    """
    line = read_first(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )
    siblings = {cpu}
    for part in (line or "").split(","):
        start, _, end = part.partition("-")
        try:
            low = int(start)
            high = int(end) if end else low
        except ValueError:
            continue
        siblings.update(range(low, high + 1))
    return sorted(siblings)


def cpu_jiffies(cpus: list[int]) -> dict[str, int]:
    """Read the busy jiffy counter of some logical CPUs.

    Parameters
    ----------
    cpus : list of int
        Logical CPUs to read.

    Returns
    -------
    dict
        ``str(cpu) -> busy jiffies``, everything but idle and iowait.  A CPU
        whose ``/proc/stat`` line could not be parsed is absent.
    """
    wanted = {f"cpu{cpu}": str(cpu) for cpu in cpus}
    counters: dict[str, int] = {}
    try:
        text = Path("/proc/stat").read_text()
    except OSError:
        return counters
    for line in text.splitlines():
        name, _, rest = line.partition(" ")
        if name not in wanted:
            continue
        try:
            fields = [int(value) for value in rest.split()]
        except ValueError:
            continue
        if len(fields) >= 5:
            counters[wanted[name]] = sum(fields) - fields[3] - fields[4]
    return counters


def cpu_occupancy(
    before: dict[str, int], after: dict[str, int], wall_s: float
) -> dict[str, float]:
    """Busy fraction of each watched CPU over one child.

    The load average is damped over 60 s and is machine-wide: it cannot see
    an indexer that wakes up inside a short child, and says nothing about
    *which* CPU the work landed on.  These deltas do.  The measured core
    should be busy for about the whole child and its SMT sibling for none of
    it.

    Parameters
    ----------
    before, after : dict
        :func:`cpu_jiffies` readings taken around the child.
    wall_s : float
        The child's wall clock.

    Returns
    -------
    dict
        ``str(cpu) -> busy seconds / wall_s``; empty when `wall_s` is not
        positive.
    """
    busy: dict[str, float] = {}
    if wall_s <= 0:
        return busy
    ticks = os.sysconf("SC_CLK_TCK")
    for cpu, counter in after.items():
        if cpu in before:
            busy[cpu] = (counter - before[cpu]) / ticks / wall_s
    return busy


def confine_driver(cpu: int) -> list[int] | None:
    """Keep the driver off the measured core and its SMT siblings.

    The scheduler may put the driver on the measured core or -- worse,
    because nothing would show it -- on that core's SMT sibling, which
    shares the physical core's execution resources.

    Linux affinity is not hierarchical: a child's own ``taskset -c <cpu>``
    still reaches a CPU the parent excluded, so this costs the children
    nothing.

    Parameters
    ----------
    cpu : int
        The logical CPU the children are pinned to.

    Returns
    -------
    list of int or None
        The driver's affinity after the change, or ``None`` when it was left
        alone because excluding those CPUs would leave it nowhere to run.
    """
    try:
        keep = set(os.sched_getaffinity(0)) - set(cpu_siblings(cpu))
    except (AttributeError, OSError):
        return None
    if not keep:
        return None
    try:
        os.sched_setaffinity(0, keep)
    except OSError:
        return None
    return sorted(keep)


def child_affinity(spec: dict[str, Any] | None) -> list[int] | None:
    """Return the CPUs a child reported it was allowed to run on.

    Reading it back is the only check that ``taskset`` took effect: a bad
    CPU number or a cgroup cpuset would otherwise produce a whole table of
    unpinned timings that looks exactly like a pinned one.

    Parameters
    ----------
    spec : dict or None
        The child's own JSON, or ``None`` when it produced none.

    Returns
    -------
    list of int or None
        The sorted affinity, or ``None`` when the child did not record one.
    """
    if not isinstance(spec, dict):
        return None
    affinity = spec.get("affinity")
    if not isinstance(affinity, list):
        return None
    try:
        return sorted(int(cpu) for cpu in affinity)
    except (TypeError, ValueError):
        return None


def cpu_facts(cpu: int | None) -> dict[str, Any]:
    """Describe the host once, for the header of a result.

    Parameters
    ----------
    cpu : int or None
        The logical CPU the children will be pinned to, or ``None`` when
        they are not pinned; the per-CPU fields are then ``None``.

    Returns
    -------
    dict
        The CPU model and ISA level, the logical CPU count, the kernel
        release, and for the measured CPU its governor, its minimum and
        maximum frequency in kHz, the ``no_turbo`` setting where the driver
        exposes one, and its SMT siblings.  A field this kernel does not
        expose is ``None``.
    """
    base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
    return {
        "cpu_model": cpu_model(),
        "isa_level": host_isa_level(),
        "nproc": os.cpu_count(),
        "kernel": platform.release(),
        "pinned_cpu": cpu,
        "siblings": None if cpu is None else cpu_siblings(cpu),
        "governor": (
            None if cpu is None else read_first(f"{base}/scaling_governor")
        ),
        "cpuinfo_min_freq": (
            None if cpu is None else read_first(f"{base}/cpuinfo_min_freq")
        ),
        "cpuinfo_max_freq": (
            None if cpu is None else read_first(f"{base}/cpuinfo_max_freq")
        ),
        "no_turbo": read_first("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    }
