"""Elevator state machine for the simulator.

The Elevator holds `event_queue` handed to it by the scheduler.
Each event is either a `PickupEvent` or a `DropoffEvent` — it names a
specific passenger AND the floor where that passenger boards / alights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class OvercapacityError(Exception):
    """Raised by the simulator at a stop boundary when the cabin count
    leaving the stop exceeds `capacity` (or falls below 0).

    The check fires AFTER the simulator pops every front event whose
    floor matches the elevator's current floor — i.e. once the door
    has "closed." Transient values within a stop (cabin briefly above
    capacity mid-swap, or briefly negative if dropoffs are queued
    before their corresponding same-stop pickups) are allowed; only
    the post-stop count is constrained.

    Under a scheduler that walks its proposed queue with
    `routing.walk_capacity` and verifies `max_pc <= capacity`, this
    exception is unreachable in normal operation — reaching it means
    the scheduler produced an invalid event sequence and is a real
    bug. Survives `python -O` (unlike an assertion).
    """


class Direction(Enum):
    UP = 1
    DOWN = -1
    IDLE = 0


@dataclass(frozen=True)
class PickupEvent:
    """The elevator picks up this passenger at this floor."""

    passenger_id: int
    floor: int


@dataclass(frozen=True)
class DropoffEvent:
    """The elevator drops off this passenger at this floor."""

    passenger_id: int
    floor: int


ElevatorEvent = Union[PickupEvent, DropoffEvent]


@dataclass
class Elevator:
    id: int
    capacity: int
    load_time: int = 3
    floor: int = 1      # current floor

    _passenger_count: int = field(default=0, init=False, repr=False)
    _event_queue: list[ElevatorEvent] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive (got {self.capacity})")
        if self.load_time < 1:
            raise ValueError(f"load_time must be >= 1 (got {self.load_time})")
        if self.floor < 1:
            raise ValueError(f"floor must be >= 1 (got {self.floor})")

    # ---- Read-only state -----------------------------------------

    @property
    def passenger_count(self) -> int:
        return self._passenger_count

    @property
    def event_queue(self) -> tuple[ElevatorEvent, ...]:
        """The remaining ordered event list. Tuple = immutable from outside."""
        return tuple(self._event_queue)

    @property
    def heading(self) -> Direction:
        """Direction the elevator will move on its next non-dwell tick."""
        for e in self._event_queue:
            if e.floor > self.floor:
                return Direction.UP
            if e.floor < self.floor:
                return Direction.DOWN
        return Direction.IDLE

    @property
    def free_seats(self) -> int:
        return self.capacity - self._passenger_count

    @property
    def is_full(self) -> bool:
        return self._passenger_count >= self.capacity

    # ---- Mutators (Scheduler side) --------------------------------------

    def set_events(self, events: list[ElevatorEvent]) -> None:
        """Replace the event queue with `events` (scheduler-decided order)."""
        self._event_queue = list(events)