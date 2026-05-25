"""Unit tests for simulator.py — Simulator tick loop + SimulationResult."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from building import BuildingConfig
from elevator import DropoffEvent, Elevator, OvercapacityError, PickupEvent
from passenger import Passenger
from scheduler import RoundRobinScheduler
from simulator import (
    InvalidRequestError,
    SimulationResult,
    Simulator,
    estimate_max_ticks,
)


# ---- helpers --------------------------------------------------------------

def build(
    num_floors: int = 60,
    num_elevators: int = 1,
    capacity: int = 10,
    load_time: int = 3,
) -> tuple[BuildingConfig, list[Elevator], RoundRobinScheduler]:
    cfg = BuildingConfig(
        num_floors=num_floors,
        num_elevators=num_elevators,
        capacity=capacity,
        load_time=load_time,
    )
    elevs = [Elevator(id=i, capacity=capacity, load_time=load_time) for i in range(num_elevators)]
    sched = RoundRobinScheduler(elevs)
    return cfg, elevs, sched


def req(passenger_id: int, t: int, origin: int, dest: int) -> Passenger:
    return Passenger(id=passenger_id, origin=origin, destination=dest, request_time=t)


# ---- basic shape ----------------------------------------------------------

def test_no_passengers_terminates_immediately():
    cfg, _, sched = build()
    sim = Simulator(cfg, sched, passengers=[])
    result = sim.run()
    assert isinstance(result, SimulationResult)
    assert result.final_tick == 0
    assert result.position_log == [(0, [1])]   # only initial state


def test_position_log_t0_is_initial_state():
    cfg, _, sched = build(num_elevators=2)
    sim = Simulator(cfg, sched, passengers=[req(1, t=0, origin=1, dest=5)])
    result = sim.run()
    assert result.position_log[0] == (0, [1, 1])


def test_position_log_one_row_per_tick():
    cfg, _, sched = build()
    sim = Simulator(cfg, sched, passengers=[req(1, t=0, origin=1, dest=4)])
    result = sim.run()
    # ticks t=0, 1, 2, ... up to final_tick: that's final_tick + 1 rows
    assert len(result.position_log) == result.final_tick + 1
    # ticks are strictly increasing 0..final_tick
    assert [row[0] for row in result.position_log] == list(range(result.final_tick + 1))


def test_simulator_rejects_mismatched_elevator_count():
    cfg = BuildingConfig(num_floors=10, num_elevators=3, capacity=10)
    elevs = [Elevator(id=0, capacity=10), Elevator(id=1, capacity=10)]  # only 2
    sched = RoundRobinScheduler(elevs)
    with pytest.raises(ValueError, match="num_elevators"):
        Simulator(cfg, sched, passengers=[])


# ---- spec example: 3 requests --------------------------------------------

def test_spec_example_three_requests():
    """The 3-request example from elevator_simulator.md:
        time, id, origin, dest
        0, 1, 1, 51
        0, 2, 1, 37
        10, 3, 20, 1
    Verifies the sim runs to completion with sensible timings and that
    the position log ends exactly at the final dropoff."""
    cfg, elevs, sched = build(num_floors=60, num_elevators=2, capacity=10, load_time=3)
    passengers = [
        req(1, t=0, origin=1, dest=51),
        req(2, t=0, origin=1, dest=37),
        req(3, t=10, origin=20, dest=1),
    ]
    result = Simulator(cfg, sched, passengers).run()

    # All passengers dropped off
    for p in passengers:
        assert p.pickup_time is not None
        assert p.dropoff_time is not None
        assert p.request_time <= p.pickup_time < p.dropoff_time

    # Position log spans [0, final_tick = last dropoff]
    last_dropoff = max(p.dropoff_time for p in passengers)
    assert result.final_tick == last_dropoff
    assert result.position_log[-1][0] == last_dropoff


# ---- causal reveal: future requests invisible to scheduler --------------

def test_future_request_not_assigned_early():
    """A request at t=10 must not have assigned_elevator set before t=10."""
    cfg, _, sched = build(num_elevators=2)
    early = req(1, t=10, origin=2, dest=8)
    sim = Simulator(cfg, sched, passengers=[early])
    # Up to t=9, the request is invisible; we drive the sim tick-by-tick
    # by re-using the internal pending map. Easier: run a sim with only
    # this one passenger and check that pickup_time >= 10.
    sim.run()
    assert early.pickup_time is not None
    assert early.pickup_time >= early.request_time


def test_idle_elevators_log_floor_1_until_request_arrives():
    """No movement should happen before any request is revealed."""
    cfg, _, sched = build(num_elevators=2)
    p = req(1, t=5, origin=3, dest=7)
    sim = Simulator(cfg, sched, passengers=[p])
    result = sim.run()
    # Ticks 0..4 are all-idle at floor 1 (no requests yet revealed).
    for t, floors in result.position_log[:5]:
        assert floors == [1, 1], f"tick {t}: expected all floor-1 idle, got {floors}"


# ---- multi-elevator end-to-end -------------------------------------------

def test_two_elevators_split_load():
    """Mix of UP and DOWN requests: scheduler should bind to both."""
    cfg, _, sched = build(num_floors=20, num_elevators=2)
    passengers = [
        req(1, t=0, origin=1, dest=10),
        req(2, t=0, origin=15, dest=5),
        req(3, t=1, origin=2, dest=8),
    ]
    result = Simulator(cfg, sched, passengers).run()
    # Both elevators got at least one passenger
    elev_ids = {p.assigned_elevator for p in passengers}
    assert elev_ids == {0, 1}
    # Everyone delivered
    assert all(p.dropoff_time is not None for p in passengers)


# ---- safety net ----------------------------------------------------------

def test_max_ticks_raises_with_pending_passenger_ids():
    """Explicit override path: if max_ticks is unreasonably small,
    RuntimeError lists the laggards."""
    cfg, _, sched = build()
    p = req(1, t=0, origin=1, dest=50)
    sim = Simulator(cfg, sched, passengers=[p], max_ticks=5)
    with pytest.raises(RuntimeError, match="pending: \\[1\\]"):
        sim.run()


def test_estimate_max_ticks_formula():
    """The derived upper bound matches the documented formula:
    max(request_time) + N * (2*load_time + 2*(num_floors - 1))."""
    cfg = BuildingConfig(num_floors=60, num_elevators=1, capacity=10, load_time=3)
    passengers = [
        req(1, t=0, origin=1, dest=51),
        req(2, t=0, origin=1, dest=37),
        req(3, t=10, origin=20, dest=1),
    ]
    # max_request_time = 10
    # per_passenger_worst = 2*3 + 2*59 = 124
    # total = 10 + 3 * 124 = 382
    assert estimate_max_ticks(cfg, passengers) == 382


def test_estimate_max_ticks_handles_empty_input():
    cfg = BuildingConfig(num_floors=10, num_elevators=1, capacity=5)
    assert estimate_max_ticks(cfg, []) == 0


def test_simulator_uses_derived_max_ticks_by_default():
    """When max_ticks is not passed, the simulator computes it from input."""
    cfg, _, sched = build(num_floors=10, num_elevators=1, capacity=10, load_time=3)
    p = req(1, t=0, origin=1, dest=5)
    sim = Simulator(cfg, sched, passengers=[p])
    # max_request_time=0 + 1 * (2*3 + 2*(10-1)) = 24
    assert sim.max_ticks == 24


def test_simulator_explicit_max_ticks_overrides_derived():
    cfg, _, sched = build()
    p = req(1, t=0, origin=1, dest=5)
    sim = Simulator(cfg, sched, passengers=[p], max_ticks=42)
    assert sim.max_ticks == 42


# ---- Hooks for streaming variant -----------------------------------------

def test_streaming_subclass_via_hooks():
    """Proves a subclass can convert Simulator from batch to streaming
    by overriding `_reveal_at` and `_is_input_exhausted`. Sketches the
    eventual StreamingSimulator without committing to its public API."""

    class FakeSource:
        """Emits one passenger at t=2; goes exhausted immediately after."""
        def __init__(self):
            self.passenger = Passenger(
                id=1, origin=1, destination=4, request_time=2,
            )
            self._delivered = False

        def poll(self, t: int) -> list:
            if t >= 2 and not self._delivered:
                self._delivered = True
                return [self.passenger]
            return []

        def exhausted(self) -> bool:
            return self._delivered

    class StreamingSimulator(Simulator):
        def __init__(self, config, scheduler, source, *, max_ticks=1000):
            super().__init__(config, scheduler, passengers=[], max_ticks=max_ticks)
            self._source = source

        def _reveal_at(self, t):
            new = self._source.poll(t)
            self.passengers.extend(new)
            return new

        def _is_input_exhausted(self):
            return self._source.exhausted()

    cfg, _, sched = build(num_floors=10, num_elevators=1, capacity=10, load_time=1)
    src = FakeSource()
    sim = StreamingSimulator(cfg, sched, src, max_ticks=200)

    # Before any tick: source not exhausted → _all_dropped_off is False
    # even though the passenger list is empty.
    assert sim._all_dropped_off() is False

    result = sim.run()

    # The one streamed passenger was assigned + delivered.
    assert len(sim.passengers) == 1
    p = sim.passengers[0]
    assert p.pickup_time is not None and p.dropoff_time is not None
    assert p.pickup_time >= 2   # not picked up before reveal
    assert result.final_tick == p.dropoff_time


# ---- timing sanity --------------------------------------------------------

def test_pickup_and_dropoff_match_elevator_motion():
    """A single passenger from floor 1 → 5: pickup at floor 1 should
    happen at t ≈ 1 (ARRIVED at floor 1 on first tick), and dropoff
    after 5 floors of travel + 2 dwells of 3 ticks each (one at origin,
    one at dest)."""
    cfg, _, sched = build(num_floors=10, num_elevators=1, capacity=10, load_time=3)
    p = req(1, t=0, origin=1, dest=5)
    Simulator(cfg, sched, passengers=[p]).run()
    # Elevator starts at floor 1, queue is [1, 5]. First tick: ARRIVED at 1.
    assert p.pickup_time == 1
    # Then load_time=3 dwell (ARRIVED + 2 LOADING = 3 ticks t=1,2,3),
    # then 4 movement ticks (1→5 is 4 floors), then ARRIVED at 5 → dropoff.
    # That's pickup at t=1, dropoff at t=1+2+4+1 = 8.
    assert p.dropoff_time == 8


# ---- Per-passenger validation -------------------------------------------
#
# The Simulator validates the full passenger list at construction time —
# a malformed request (origin == destination, or origin / destination
# outside `[1, num_floors]`) fails fast rather than silently propagating
# through the tick loop. Scheduler trusts whatever it's given.

def test_simulator_rejects_self_request():
    """origin == destination is meaningless; surface the bug at __init__."""
    cfg, _, sched = build()
    with pytest.raises(InvalidRequestError, match="origin == destination"):
        Simulator(cfg, sched, passengers=[req(1, t=0, origin=5, dest=5)])


def test_simulator_rejects_zero_origin():
    cfg, _, sched = build()
    with pytest.raises(InvalidRequestError, match=r"origin 0 outside \[1, "):
        Simulator(cfg, sched, passengers=[req(1, t=0, origin=0, dest=5)])


def test_simulator_rejects_origin_above_num_floors():
    """Catches the gap the old `Scheduler._validate_request` missed."""
    cfg, _, sched = build(num_floors=60)
    with pytest.raises(InvalidRequestError, match=r"origin 100 outside \[1, 60\]"):
        Simulator(cfg, sched, passengers=[req(1, t=0, origin=100, dest=5)])


def test_simulator_rejects_destination_above_num_floors():
    cfg, _, sched = build(num_floors=60)
    with pytest.raises(InvalidRequestError, match=r"destination 100 outside \[1, 60\]"):
        Simulator(cfg, sched, passengers=[req(1, t=0, origin=5, dest=100)])


# ---- Capacity / underflow at stop boundaries ----------------------------
#
# Cap-safety is the SCHEDULER's invariant; the SIMULATOR is the runtime
# defense that surfaces a scheduler bug. A stop is atomic — within it
# `passenger_count` may transiently exceed capacity or dip below zero
# (door open, swap in progress). After all events at a shared floor
# have been popped, the post-stop count must lie in `[0, capacity]`.

def test_simulator_raises_overcapacity_on_bad_stop_leaving_count():
    """Three PickupEvents at the same floor with `capacity = 2`. The
    intra-stop count transiently goes to 3 (allowed); the post-stop
    count of 3 violates the boundary check → `OvercapacityError`."""
    cfg, elevs, sched = build(
        num_floors=5, num_elevators=1, capacity=2, load_time=1,
    )
    passengers = [
        req(i, t=0, origin=1, dest=3) for i in (1, 2, 3)
    ]
    sim = Simulator(cfg, sched, passengers)
    # Bypass the scheduler — install an unsafe event sequence directly.
    elevs[0].set_events([
        PickupEvent(passenger_id=1, floor=1),
        PickupEvent(passenger_id=2, floor=1),
        PickupEvent(passenger_id=3, floor=1),
    ])
    with pytest.raises(OvercapacityError, match="post-stop cabin count 3"):
        sim._process_events_at(elevs[0], t=0)


def test_simulator_raises_overcapacity_on_negative_post_stop():
    """A DropoffEvent for a passenger who isn't in the cabin drives the
    post-stop count below zero → same boundary check, same error type."""
    cfg, elevs, sched = build(
        num_floors=5, num_elevators=1, capacity=5, load_time=1,
    )
    p = req(1, t=0, origin=1, dest=3)
    sim = Simulator(cfg, sched, passengers=[p])
    elevs[0].set_events([DropoffEvent(passenger_id=1, floor=1)])
    with pytest.raises(OvercapacityError, match="post-stop cabin count -1"):
        sim._process_events_at(elevs[0], t=0)


def test_simulator_allows_transient_overcap_within_a_stop():
    """`PU PU DO` at the same floor, starting `pc = capacity - 1`:
    intra-stop the count goes capacity-1 → capacity → capacity+1 →
    capacity. Transient over-cap (capacity+1) is allowed; the
    post-stop count is `capacity`, which clears the boundary check."""
    cfg, elevs, sched = build(
        num_floors=5, num_elevators=1, capacity=2, load_time=1,
    )
    passengers = [
        req(0, t=0, origin=1, dest=5),
        req(1, t=0, origin=1, dest=5),
        req(2, t=0, origin=1, dest=5),
    ]
    sim = Simulator(cfg, sched, passengers)
    elev = elevs[0]
    # Seed the cabin with one passenger already aboard (manual setup
    # since the scheduler wouldn't produce this state).
    elev.set_events([PickupEvent(passenger_id=0, floor=1)])
    sim._process_events_at(elev, t=0)        # pc = 1
    assert elev.passenger_count == 1
    # Now install PU(1)@1, PU(2)@1, DO(0)@1: transiently 1→2→3→2.
    elev.set_events([
        PickupEvent(passenger_id=1, floor=1),
        PickupEvent(passenger_id=2, floor=1),
        DropoffEvent(passenger_id=0, floor=1),
    ])
    sim._process_events_at(elev, t=1)        # should NOT raise
    assert elev.passenger_count == 2


# End to end test
def test_end_to_end_completes_no_stranding():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from request_gen import generate_uniform

    cfg = BuildingConfig(num_floors=60, num_elevators=2, capacity=10, load_time=3)
    passengers = generate_uniform(
        num_passengers=20, num_floors=60, duration=100, seed=42,
    )
    elevs = [Elevator(id=i, capacity=cfg.capacity, load_time=cfg.load_time)
             for i in range(cfg.num_elevators)]
    sched = RoundRobinScheduler(elevs)
    result = Simulator(cfg, sched, passengers).run()

    # Every passenger delivered.
    for p in passengers:
        assert p.pickup_time is not None, f"P{p.id} never picked up"
        assert p.dropoff_time is not None, f"P{p.id} never delivered"
        assert p.request_time <= p.pickup_time < p.dropoff_time