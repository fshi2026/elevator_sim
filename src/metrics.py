"""Aggregate stats for comparing schedulers.

Three levels of aggregation, all derived from the per-passenger raw
data in `passengers.csv`:

  1. **Per-passenger** (raw) — `Passenger.wait_time`, `total_time`,
     `overhead(load_time)`. The source of truth.

  2. **Per-simulation** — `RunStats`: one seed's worth of passengers
     reduced to mean / p50 / p95 / max of the per-passenger metrics,
     plus the run's `makespan` (final tick) and the per-passenger
     `ideal` baseline. One row per seed in `summary.csv`.

  3. **Per-scheduler** — `AggregateStats`: many seeds rolled up into
     the headline numbers we use to say "scheduler B is X% better than
     A on metric Y across the corpus." Two roll-up strategies are
     emitted side-by-side, because they answer different questions:

     - **Macro-average** = mean of the per-seed values. Each seed
       counted equally. Tells you "how does this scheduler perform
       on a typical run?" — small bursts and big bursts contribute
       the same weight.

     - **Micro-average** = pool every passenger across every seed
       into one bag and compute the stats once. Each passenger
       counted equally. Tells you "across every passenger this
       scheduler ever served, what was the experience?" — runs with
       more passengers dominate.

     Both are useful. The divergence between macro and micro is itself
     informative: if macro_wait_mean is much lower than micro_wait_mean,
     the scheduler is better at small workloads than large ones.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from passenger import Passenger, PassengerState


@dataclass(frozen=True)
class RunStats:
    """Stats for a single simulation run (one seed's passengers).

    All "per-passenger metric" fields are computed only over
    `delivered` passengers (those with non-None `dropoff_time`).
    Empty / undelivered runs are unusable for comparison.
    """

    num_passengers: int        # total in the input, delivered or not
    num_delivered: int         # got a dropoff_time
    makespan: int              # final tick of the simulation

    # Wait time: request → pickup. Trip-length-INdependent — the
    # cleanest measure of "how long did the scheduler make me wait?"
    wait_mean: float
    wait_p50: float
    wait_p95: float
    wait_max: int

    # Total time: request → dropoff. User-facing — what the passenger
    # actually experienced. Confounded with trip length.
    total_mean: float
    total_p50: float
    total_p95: float
    total_max: int

    # Overhead: total - ideal_total(load_time). The time the scheduler
    # added on top of each passenger's unavoidable trip cost. Trip-
    # length-NORMALIZED — fair to mean-average across mixed workloads.
    overhead_mean: float
    overhead_p50: float
    overhead_p95: float
    overhead_max: int

    # Per-passenger ideal baseline (mean of ideal_total_time over
    # delivered passengers). Useful as the floor that `total_mean`
    # could reach with a perfect scheduler + waiting elevator.
    ideal_mean: float


@dataclass(frozen=True)
class AggregateStats:
    """Per-scheduler stats across multiple runs.

    Both macro (mean-of-per-seed) and micro (pool-all-passengers)
    aggregations are emitted; see module docstring for when each is
    appropriate.
    """

    num_seeds: int
    num_passengers_total: int    # sum across all seeds (delivered only)

    # ---- macro: mean across per-seed values --------------------------
    macro_wait_mean: float
    macro_wait_p95: float        # mean of per-seed p95s
    macro_wait_max: float        # mean of per-seed maxes
    macro_total_mean: float
    macro_total_p95: float
    macro_overhead_mean: float
    macro_overhead_p95: float
    macro_overhead_max: float
    macro_makespan: float

    # ---- micro: pool all passengers, compute stats once -------------
    micro_wait_mean: float
    micro_wait_p95: float
    micro_wait_max: int
    micro_total_mean: float
    micro_total_p95: float
    micro_overhead_mean: float
    micro_overhead_p95: float
    micro_overhead_max: int


def compute_run_stats(
    passengers: Sequence[Passenger],
    load_time: int,
    makespan: int,
) -> RunStats:
    """Reduce one run's per-passenger data to a `RunStats` record.

    `passengers` is the simulator's output — `delivered` is the subset
    with `dropoff_time` populated. Raises ValueError if zero delivered
    passengers (nothing to summarize).
    """
    delivered = [p for p in passengers if p.state == PassengerState.DELIVERED]
    if not delivered:
        raise ValueError("compute_run_stats: no delivered passengers to summarize")

    waits = [p.wait_time for p in delivered]
    totals = [p.total_time for p in delivered]
    overheads = [p.overhead(load_time) for p in delivered]
    ideals = [p.ideal_total_time(load_time) for p in delivered]

    return RunStats(
        num_passengers=len(passengers),
        num_delivered=len(delivered),
        makespan=makespan,
        wait_mean=statistics.mean(waits),
        wait_p50=_p50(waits),
        wait_p95=_p95(waits),
        wait_max=max(waits),
        total_mean=statistics.mean(totals),
        total_p50=_p50(totals),
        total_p95=_p95(totals),
        total_max=max(totals),
        overhead_mean=statistics.mean(overheads),
        overhead_p50=_p50(overheads),
        overhead_p95=_p95(overheads),
        overhead_max=max(overheads),
        ideal_mean=statistics.mean(ideals),
    )


def aggregate_runs(
    runs: Sequence[tuple[RunStats, Sequence[Passenger]]],
    load_time: int,
) -> AggregateStats:
    """Roll up many runs into per-scheduler `AggregateStats`.

    Pass a sequence of `(run_stats, passengers)` pairs — `run_stats`
    is the precomputed per-seed summary (used for macro aggregation)
    and `passengers` is the raw list (used for micro aggregation).
    Keeping both inputs makes the dependence explicit; the function
    has no implicit reach back into per-passenger data.
    """
    if not runs:
        raise ValueError("aggregate_runs: empty input")

    per_seed = [s for s, _ in runs]
    pooled = [
        p for _, ps in runs
        for p in ps
        if p.state == PassengerState.DELIVERED
    ]
    if not pooled:
        raise ValueError("aggregate_runs: no delivered passengers across any run")

    # Pool all passengers for micro stats.
    waits = [p.wait_time for p in pooled]
    totals = [p.total_time for p in pooled]
    overheads = [p.overhead(load_time) for p in pooled]

    return AggregateStats(
        num_seeds=len(per_seed),
        num_passengers_total=len(pooled),
        macro_wait_mean=statistics.mean(s.wait_mean for s in per_seed),
        macro_wait_p95=statistics.mean(s.wait_p95 for s in per_seed),
        macro_wait_max=statistics.mean(s.wait_max for s in per_seed),
        macro_total_mean=statistics.mean(s.total_mean for s in per_seed),
        macro_total_p95=statistics.mean(s.total_p95 for s in per_seed),
        macro_overhead_mean=statistics.mean(s.overhead_mean for s in per_seed),
        macro_overhead_p95=statistics.mean(s.overhead_p95 for s in per_seed),
        macro_overhead_max=statistics.mean(s.overhead_max for s in per_seed),
        macro_makespan=statistics.mean(s.makespan for s in per_seed),
        micro_wait_mean=statistics.mean(waits),
        micro_wait_p95=_p95(waits),
        micro_wait_max=max(waits),
        micro_total_mean=statistics.mean(totals),
        micro_total_p95=_p95(totals),
        micro_overhead_mean=statistics.mean(overheads),
        micro_overhead_p95=_p95(overheads),
        micro_overhead_max=max(overheads),
    )


def _p50(xs: Sequence[int]) -> float:
    """Population median; uses linear interpolation for even-length
    samples (matches numpy default). Safe on small lists where
    `statistics.quantiles(..., n=2)` would raise."""
    return _quantile(xs, 0.5)


def _p95(xs: Sequence[int]) -> float:
    """Population p95 with linear interpolation between order
    statistics (matches numpy default). Safe on small samples."""
    return _quantile(xs, 0.95)


def _quantile(xs: Sequence[int], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac
