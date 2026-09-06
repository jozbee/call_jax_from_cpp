# Latency and tails

*Assumes nothing. The words the reports use, and why the worst call matters
more than the average. Producing a number worth quoting is
{doc}`/guides/measuring`; the traps are {doc}`/developer/measurement`.*

## Percentiles and the two ratios

Sort a run's samples. `p50` is the median: what a typical call costs, which is
capacity, not a deadline. `p99.9` is the one-in-a-thousand call — on a
one-kilohertz loop, once a second. `max` is the worst call in this run, and a
run too short to contain the rare event flatters it. The ratios `max/p50` and
`p99.9/p50` are the {term}`tail` this project optimizes: how much worse than
typical the bad calls are. A change that lowers the mean and raises `max/p50`
is a worse result.
[Percentile](https://en.wikipedia.org/wiki/Percentile).

## The four quantities of a periodic loop

```text
release        wake         call start      call end   ready      next release
   |------------|---------------|---------------|---------|--------------|
   |<- wake-up ->|               |<-   call    ->|         |
   |<-------------------- cycle time --------------------->|
   |<------------------------- period ------------------------------------>|
```

{term}`Wake-up latency <wake-up latency>` is how late the sleep returned: the
scheduler's and the idle state's contribution, nothing to do with the
computation. *Call latency* is the computation. {term}`Cycle time <cycle time>`
is everything between waking and being ready to sleep again.
{term}`Period jitter <period jitter>` is this wake-up minus the previous one,
minus the period; it is signed, because waking early is as much a scheduling
defect as waking late, and a report shows both tails. A
{term}`deadline miss` is a cycle whose work ended after the next release; on a
loop that never skips a period, the next cycle starts late and catches up.

## Where the spread comes from

Inside the process: the allocator, page faults, and XLA's own pool threads.
Outside it: the scheduler, the timer tick, interrupts, and the idle and
frequency states of the core. {doc}`/guides/realtime` maps each to what
removes it.

## Measuring honestly

An idle machine, because a concurrent build does not add noise to a run, it
invalidates it. A campaign long enough to contain the event being claimed — a
rare spike hides in a short run. Configurations under comparison interleaved
rather than run one after the other, because temperature drifts by the same
order as the effect. {doc}`/guides/measuring` has the commands.

## The floor

`cyclictest` measures wake-up latency alone, with no computation, and its
worst case is the floor under any loop on that host: a loop cannot beat it,
and a loop far above it is losing time somewhere this page names.
{doc}`/developer/realtime-notes` has this project's figures.
