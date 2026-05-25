"""Unit tests for the Elevator state machine.

These tests exercise the elevator in isolation — no scheduler, no simulator.
The test pattern is: install an `event_queue` (the scheduler's job in
real runs), drive `tick()` in a loop, and assert on the event-stream and
the final / mid-trip state. After each ARRIVED, the test pops events at
the current floor (the simulator's job in real runs).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from elevator import (
    Direction,
    DropoffEvent,
    Elevator,
    Event,
    PickupEvent,
)


def drain_at_current_floor(elev: Elevator) -> list:
    """Pop and return all front events whose floor equals current floor.

    Simulates what the Simulator does on ARRIVED/LOADING.
    """
    popped = []
    while elev.event_queue and elev.event_queue[0].floor == elev.floor:
        popped.append(elev.pop_front())
    return popped


def run(elevator: Elevator, max_ticks: int = 100) -> list[tuple[Event, int]]:
    """Tick until IDLE (or `max_ticks` exceeded), draining events on each
    ARRIVED/LOADING. Returns `(event, floor)` per tick."""
    log: list[tuple[Event, int]] = []
    for _ in range(max_ticks):
        ev = elevator.tick()
        log.append((ev, elevator.floor))
        if ev in (Event.ARRIVED, Event.LOADING):
            drain_at_current_floor(elevator)
        if ev == Event.IDLE:
            return log
    raise AssertionError(
        f"elevator did not idle within {max_ticks} ticks; log={log}"
    )


# ---- Construction & defaults ----------------------------------------------

def test_idle_when_no_events():
    e = Elevator(id=0, capacity=10)
    assert e.tick() == Event.IDLE
    assert e.direction == Direction.IDLE
    assert e.floor == 1
    assert e.passenger_count == 0
    assert e.event_queue == ()


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        Elevator(id=0, capacity=0)
    with pytest.raises(ValueError):
        Elevator(id=0, capacity=5, load_time=0)
    with pytest.raises(ValueError):
        Elevator(id=0, capacity=5, floor=0)


# ---- Movement & dwell -----------------------------------------------------

def test_single_passenger_up_trip():
    """Passenger boards at floor 3, exits at floor 5. Elevator starts at 1.

    Direction updates only on MOVED ticks; it stays IDLE until the
    first move, becomes UP, and reverts to IDLE only when an IDLE
    event fires."""
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([
        PickupEvent(passenger_id=1, floor=3),
        DropoffEvent(passenger_id=1, floor=5),
    ])
    assert e.direction == Direction.IDLE

    assert e.tick() == Event.MOVED      # 1 -> 2
    assert e.floor == 2
    assert e.direction == Direction.UP
    assert e.tick() == Event.MOVED      # 2 -> 3
    assert e.tick() == Event.ARRIVED    # dwell at 3
    # Simulator-equivalent: drain events at floor 3 (one PickupEvent).
    drained = drain_at_current_floor(e)
    assert len(drained) == 1 and isinstance(drained[0], PickupEvent)
    assert e.passenger_count == 1

    assert e.tick() == Event.MOVED      # 3 -> 4
    assert e.tick() == Event.MOVED      # 4 -> 5
    assert e.tick() == Event.ARRIVED    # dwell at 5
    drained = drain_at_current_floor(e)
    assert len(drained) == 1 and isinstance(drained[0], DropoffEvent)
    assert e.passenger_count == 0

    assert e.tick() == Event.IDLE
    assert e.direction == Direction.IDLE


def test_multi_passenger_same_direction_pickup():
    """Three pickups at 2/4/6, all bound for 8."""
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([
        PickupEvent(passenger_id=1, floor=2),
        PickupEvent(passenger_id=2, floor=4),
        PickupEvent(passenger_id=3, floor=6),
        DropoffEvent(passenger_id=1, floor=8),
        DropoffEvent(passenger_id=2, floor=8),
        DropoffEvent(passenger_id=3, floor=8),
    ])
    log = run(e)

    expected = [
        (Event.MOVED, 2),
        (Event.ARRIVED, 2),
        (Event.MOVED, 3),
        (Event.MOVED, 4),
        (Event.ARRIVED, 4),
        (Event.MOVED, 5),
        (Event.MOVED, 6),
        (Event.ARRIVED, 6),
        (Event.MOVED, 7),
        (Event.MOVED, 8),
        (Event.ARRIVED, 8),
        (Event.IDLE, 8),
    ]
    assert log == expected
    assert e.passenger_count == 0   # all dropped off in the floor-8 dwell


def test_direction_reversal_at_queue_boundary():
    """Scheduler orders events to take the elevator UP to 8 then DOWN to 1.
    Direction tracks the most-recent MOVED step; flips happen at sweep
    boundaries without any invariant check."""
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([
        # Drive purely as floor-visits: pickup an imaginary passenger
        # to anchor each visit so the elevator has something to do.
        # (Tests at this level use synthetic ids only the elevator sees.)
        PickupEvent(passenger_id=10, floor=5),
        PickupEvent(passenger_id=11, floor=8),
        DropoffEvent(passenger_id=10, floor=3),
        DropoffEvent(passenger_id=11, floor=1),
    ])
    log = run(e)
    floors = [floor for _, floor in log]
    assert 5 in floors and 8 in floors and 3 in floors and 1 in floors
    assert max(floors) == 8
    assert e.floor == 1
    assert e.direction == Direction.IDLE
    assert e.passenger_count == 0


def test_load_time_accounting():
    """load_time=3 → 1 ARRIVED + 2 LOADING ticks per stop."""
    e = Elevator(id=0, capacity=10, load_time=3)
    e.set_events([PickupEvent(passenger_id=1, floor=2)])
    assert e.tick() == Event.MOVED      # 1 -> 2
    assert e.tick() == Event.ARRIVED    # first dwell tick
    # In real runs the simulator drains here; do it manually.
    drain_at_current_floor(e)
    assert e.tick() == Event.LOADING
    assert e.tick() == Event.LOADING
    assert e.tick() == Event.IDLE


def test_load_time_one_is_single_dwell_tick():
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([
        PickupEvent(passenger_id=1, floor=2),
        DropoffEvent(passenger_id=1, floor=4),
    ])
    assert e.tick() == Event.MOVED
    assert e.tick() == Event.ARRIVED
    drain_at_current_floor(e)
    assert e.tick() == Event.MOVED
    assert e.tick() == Event.MOVED
    assert e.tick() == Event.ARRIVED
    drain_at_current_floor(e)
    assert e.tick() == Event.IDLE


# ---- set_events semantics --------------------------------------------------

def test_set_events_does_not_change_direction():
    """set_events doesn't recompute direction (direction tracks last MOVED)."""
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([
        PickupEvent(passenger_id=1, floor=3),
        DropoffEvent(passenger_id=1, floor=1),
    ])
    assert e.direction == Direction.IDLE   # no MOVED yet
    e.tick()                                # MOVED 5 -> 4
    assert e.direction == Direction.DOWN


def test_set_events_replaces_queue_completely():
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([PickupEvent(passenger_id=1, floor=5)])
    e.tick()  # 1 -> 2
    # Scheduler reroutes mid-trip.
    e.set_events([PickupEvent(passenger_id=2, floor=3)])
    assert len(e.event_queue) == 1
    assert e.event_queue[0].floor == 3
    e.tick()                            # 2 -> 3
    assert e.tick() == Event.ARRIVED
    drain_at_current_floor(e)
    assert e.tick() == Event.IDLE


def test_consecutive_same_floor_events_share_one_dwell():
    """Two events at the same floor (e.g. two pickups) drain in one dwell;
    the elevator does NOT take a second `load_time` cycle for the second
    event — it matches real elevators where one door cycle serves all
    boarders at a floor."""
    e = Elevator(id=0, capacity=10, load_time=1)
    e.set_events([
        PickupEvent(passenger_id=1, floor=3),
        PickupEvent(passenger_id=2, floor=3),
    ])
    log = run(e)
    arrived = [ev for ev, _ in log if ev == Event.ARRIVED]
    assert len(arrived) == 1
    assert e.passenger_count == 2


# ---- Capacity / underflow ---------------------------------------------
#
# Cap-safety is enforced at stop boundaries by the Simulator, not by
# the Elevator. `pop_front` no longer checks per-event — transient
# intra-stop values may exceed `capacity` or dip below zero (the door
# is open, people swap through it). The boundary check lives in
# `tests/test_simulator.py` under `test_simulator_raises_overcapacity_*`.


def test_pop_front_on_empty_queue_asserts():
    e = Elevator(id=0, capacity=5)
    with pytest.raises(AssertionError):
        e.pop_front()


# ---- Permissive direction — no flip-with-passengers assertion -------------

def test_direction_can_flip_with_passengers_aboard():
    """The elevator does not enforce 'direction flip requires empty cabin.'
    A passenger in cabin during a sweep boundary just takes a detour."""
    e = Elevator(id=0, capacity=10, load_time=1, floor=1)
    e.set_events([
        PickupEvent(passenger_id=1, floor=1),
        DropoffEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=5),
        DropoffEvent(passenger_id=2, floor=3),
    ])
    log = run(e)
    floors = [floor for _, floor in log]
    assert 5 in floors and 3 in floors
    assert e.passenger_count == 0


# ---- Derived views: heading --------------------------------

def test_heading_idle_when_queue_empty():
    e = Elevator(id=0, capacity=10, floor=5)
    assert e.heading == Direction.IDLE


def test_heading_up_when_next_event_above():
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([PickupEvent(passenger_id=1, floor=8)])
    assert e.heading == Direction.UP


def test_heading_down_when_next_event_below():
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([PickupEvent(passenger_id=1, floor=2)])
    assert e.heading == Direction.DOWN


def test_heading_skips_current_floor_events():
    """If the front events are at the current floor, heading looks
    past them to find the first event at a different floor."""
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([
        PickupEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=5),
        PickupEvent(passenger_id=3, floor=8),
    ])
    assert e.heading == Direction.UP


def test_heading_all_current_floor_is_idle():
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([
        PickupEvent(passenger_id=1, floor=5),
        DropoffEvent(passenger_id=2, floor=5),
    ])
    assert e.heading == Direction.IDLE


def test_set_events_does_not_validate_contents():
    """set_events does no validation — caller (scheduler) is the source of
    truth. Even nonsensical sequences install without error; the elevator
    will execute them and only the hard invariants (cap / underflow) will
    catch genuine corruption."""
    e = Elevator(id=0, capacity=10, floor=5)
    e.set_events([
        PickupEvent(passenger_id=1, floor=8),
        DropoffEvent(passenger_id=1, floor=10),
    ])
    e.set_events([])  # legal; elevator goes IDLE
    assert e.event_queue == ()
    assert e.tick() == Event.IDLE
