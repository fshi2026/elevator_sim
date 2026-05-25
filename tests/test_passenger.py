"""Unit tests for passenger.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from elevator import Direction
from passenger import Passenger, PassengerState


def make(pid: int = 1, origin: int = 2, dest: int = 8, t: int = 0) -> Passenger:
    return Passenger(id=pid, origin=origin, destination=dest, request_time=t)


# ---- direction (computed at __post_init__) -------------------------------

def test_direction_up_when_destination_above_origin():
    assert make(origin=2, dest=8).direction == Direction.UP


def test_direction_down_when_destination_below_origin():
    assert make(origin=8, dest=2).direction == Direction.DOWN


# ---- state property ------------------------------------------------------

def test_state_initial_is_pending_assign():
    p = make()
    assert p.state == PassengerState.PENDING_ASSIGN


def test_state_after_assignment_is_waiting():
    p = make()
    p.assigned_elevator = 0
    assert p.state == PassengerState.WAITING


def test_state_after_pickup_is_in_cabin():
    p = make()
    p.assigned_elevator = 0
    p.pickup_time = 5
    assert p.state == PassengerState.IN_CABIN


def test_state_after_dropoff_is_delivered():
    p = make()
    p.assigned_elevator = 0
    p.pickup_time = 5
    p.dropoff_time = 12
    assert p.state == PassengerState.DELIVERED


def test_state_treats_dropoff_as_terminal_even_if_other_fields_partial():
    """`state` reads in reverse-chronological order — once dropped off,
    that's the answer regardless of earlier fields' contents."""
    p = make()
    p.dropoff_time = 12
    # Even with no pickup_time / assigned_elevator, dropoff wins.
    assert p.state == PassengerState.DELIVERED


# ---- travel_distance -----------------------------------------------------

def test_travel_distance_up():
    assert make(origin=2, dest=8).travel_distance == 6


def test_travel_distance_down():
    assert make(origin=8, dest=2).travel_distance == 6


def test_travel_distance_adjacent_floors():
    assert make(origin=4, dest=5).travel_distance == 1


# ---- wait_time / total_time (existing properties; spot-check) ------------

def test_wait_and_total_time_none_until_stamped():
    p = make(t=10)
    assert p.wait_time is None
    assert p.total_time is None
    p.pickup_time = 15
    assert p.wait_time == 5
    assert p.total_time is None
    p.dropoff_time = 28
    assert p.total_time == 18
