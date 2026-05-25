"""Unit tests for metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from metrics import (
    RunStats,
    aggregate_runs,
    compute_run_stats,
)
from passenger import Passenger


def delivered(
    pid: int, origin: int, dest: int, request_time: int,
    pickup_time: int, dropoff_time: int,
) -> Passenger:
    """Construct a fully-delivered Passenger for stat fixtures."""
    p = Passenger(id=pid, origin=origin, destination=dest, request_time=request_time)
    p.pickup_time = pickup_time
    p.dropoff_time = dropoff_time
    return p


# ---- Passenger.overhead / ideal_total_time --------------------------------
# (Living in test_metrics.py because they're metric-derivation; the basic
# Passenger lifecycle tests are in test_passenger.py.)

def test_ideal_total_time_is_2load_plus_distance():
    p = Passenger(id=1, origin=3, destination=8, request_time=0)
    assert p.ideal_total_time(load_time=3) == 2 * 3 + 5


def test_overhead_is_total_minus_ideal():
    p = delivered(pid=1, origin=3, dest=8, request_time=0,
                  pickup_time=5, dropoff_time=20)
    # total_time = 20 - 0 = 20; ideal = 2*3 + 5 = 11; overhead = 9
    assert p.overhead(load_time=3) == 9


def test_overhead_none_when_not_delivered():
    p = Passenger(id=1, origin=3, destination=8, request_time=0)
    assert p.overhead(load_time=3) is None


# ---- compute_run_stats ----------------------------------------------------

def test_compute_run_stats_single_passenger():
    """One delivered passenger; all percentiles equal the single value."""
    p = delivered(pid=1, origin=1, dest=6, request_time=0,
                  pickup_time=3, dropoff_time=15)
    # wait=3, total=15, ideal=2*1+5=7, overhead=8
    stats = compute_run_stats([p], load_time=1, makespan=15)
    assert stats.num_passengers == 1
    assert stats.num_delivered == 1
    assert stats.makespan == 15
    assert stats.wait_mean == 3 and stats.wait_p50 == 3 and stats.wait_p95 == 3
    assert stats.wait_max == 3
    assert stats.total_mean == 15
    assert stats.overhead_mean == 8 and stats.overhead_max == 8
    assert stats.ideal_mean == 7


def test_compute_run_stats_distribution():
    """Three passengers with different waits — percentiles + mean correct."""
    pax = [
        delivered(pid=1, origin=1, dest=2, request_time=0,
                  pickup_time=5, dropoff_time=10),    # wait=5, total=10
        delivered(pid=2, origin=1, dest=2, request_time=0,
                  pickup_time=10, dropoff_time=15),   # wait=10, total=15
        delivered(pid=3, origin=1, dest=2, request_time=0,
                  pickup_time=15, dropoff_time=20),   # wait=15, total=20
    ]
    stats = compute_run_stats(pax, load_time=1, makespan=20)
    assert stats.wait_mean == 10
    assert stats.wait_p50 == 10                       # middle value
    assert stats.wait_max == 15
    assert stats.total_mean == 15


def test_compute_run_stats_skips_undelivered():
    """Undelivered passengers don't pollute the stats."""
    delivered_p = delivered(pid=1, origin=1, dest=5, request_time=0,
                            pickup_time=2, dropoff_time=10)
    undelivered = Passenger(id=2, origin=1, destination=5, request_time=0)
    stats = compute_run_stats([delivered_p, undelivered], load_time=1, makespan=10)
    assert stats.num_passengers == 2
    assert stats.num_delivered == 1
    assert stats.wait_mean == 2


def test_compute_run_stats_raises_on_no_delivered():
    undelivered = Passenger(id=1, origin=1, destination=5, request_time=0)
    with pytest.raises(ValueError, match="no delivered"):
        compute_run_stats([undelivered], load_time=1, makespan=100)


# ---- aggregate_runs ------------------------------------------------------

def _run(load_time: int, makespan: int, *pax: Passenger) -> tuple[RunStats, list[Passenger]]:
    """Helper: build (RunStats, passengers) tuple."""
    pax_list = list(pax)
    return compute_run_stats(pax_list, load_time, makespan), pax_list


def test_aggregate_macro_equals_mean_of_per_seed_means():
    """Macro stats are mean-of-per-seed-means, regardless of seed sizes."""
    # Seed 1: 1 passenger, wait=10
    r1 = _run(1, 10, delivered(pid=1, origin=1, dest=2, request_time=0,
                                pickup_time=10, dropoff_time=11))
    # Seed 2: 1 passenger, wait=20
    r2 = _run(1, 20, delivered(pid=2, origin=1, dest=2, request_time=0,
                                pickup_time=20, dropoff_time=21))
    agg = aggregate_runs([r1, r2], load_time=1)
    # Macro = mean(10, 20) = 15
    assert agg.macro_wait_mean == 15
    # Two seeds, two delivered passengers total
    assert agg.num_seeds == 2 and agg.num_passengers_total == 2


def test_aggregate_micro_weights_each_passenger_equally():
    """Micro pools every passenger — a seed with more passengers dominates.

    Seed 1: 4 passengers all wait=10 (so per-seed mean = 10).
    Seed 2: 1 passenger wait=50 (so per-seed mean = 50).
    Macro = mean(10, 50) = 30 (each seed equally).
    Micro = mean(10, 10, 10, 10, 50) = 18 (each passenger equally).
    """
    seed1 = [
        delivered(pid=i, origin=1, dest=2, request_time=0,
                  pickup_time=10, dropoff_time=11)
        for i in range(4)
    ]
    seed2 = [
        delivered(pid=10, origin=1, dest=2, request_time=0,
                  pickup_time=50, dropoff_time=51),
    ]
    r1 = (compute_run_stats(seed1, 1, 11), seed1)
    r2 = (compute_run_stats(seed2, 1, 51), seed2)
    agg = aggregate_runs([r1, r2], load_time=1)
    assert agg.macro_wait_mean == 30
    assert agg.micro_wait_mean == 18


def test_aggregate_raises_on_empty():
    with pytest.raises(ValueError, match="empty input"):
        aggregate_runs([], load_time=1)


def test_aggregate_makespan_is_macro_only():
    """Makespan is per-sim, so micro doesn't apply. Aggregate reports
    macro_makespan = mean of per-seed makespans."""
    r1 = _run(1, 100, delivered(pid=1, origin=1, dest=2, request_time=0,
                                pickup_time=1, dropoff_time=2))
    r2 = _run(1, 200, delivered(pid=2, origin=1, dest=2, request_time=0,
                                pickup_time=1, dropoff_time=2))
    agg = aggregate_runs([r1, r2], load_time=1)
    assert agg.macro_makespan == 150
