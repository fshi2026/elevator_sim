"""Elevator state machine for the simulator.

The Elevator executes an `event_queue` handed to it by the scheduler.
Each event is either a `PickupEvent` or a `DropoffEvent` — it names a
specific passenger AND the floor where that passenger boards / alights.
The elevator moves toward `event_queue[0].floor`, dwells `load_time`
ticks on arrival, and the simulator (during dwell) calls
`board()` / `alight()` for each event at the arrived floor. On the
last dwell tick the elevator pops all front events whose floor matches
its current floor.
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


class Event(Enum):
    """What happened during one `tick()`. Returned by `Elevator.tick()`."""

    MOVED = "moved"      # moved one floor toward the next event
    ARRIVED = "arrived"  # first dwell tick at the next event's floor
    LOADING = "loading"  # additional dwell tick (only when load_time > 1)
    IDLE = "idle"        # no events queued; sitting still


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
    floor: int = 1       # current floor

    _passenger_count: int = field(default=0, init=False, repr=False)
    _event_queue: list[ElevatorEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _direction: Direction = field(default=Direction.IDLE, init=False, repr=False)
    _load_remaining: int = field(default=0, init=False, repr=False)

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

    # ---- Mutators (Simulator side) --------------------------------------

    def pop_front(self) -> ElevatorEvent:
        """Pop the front event and adjust `passenger_count`; return the event."""
        assert self._event_queue, "pop_front on empty event_queue"
        event = self._event_queue.pop(0)
        if isinstance(event, PickupEvent):
            self._passenger_count += 1
        else:
            self._passenger_count -= 1
        return event

    # ---- Mutators (Scheduler side) --------------------------------------

    def set_events(self, events: list[ElevatorEvent]) -> None:
        """Replace the event queue with `events` (scheduler-decided order)."""
        self._event_queue = list(events)

    # ---- The tick --------------------------------------------------------

    def tick(self) -> Event:
        # Mid-dwell: tick down. No popping here — the simulator owns
        # event processing AND popping (see pop_front docstring).
        if self._load_remaining > 0:
            self._load_remaining -= 1
            return Event.LOADING

        if not self._event_queue:
            self._direction = Direction.IDLE
            return Event.IDLE

        target = self._event_queue[0].floor

        # At the target floor: this is the first dwell tick. The
        # simulator will (post-tick) process and pop all front events
        # whose floor matches. We just track the dwell timing.
        if self.floor == target:
            self._load_remaining = self.load_time - 1
            return Event.ARRIVED

        # Move one floor toward the target.
        if target > self.floor:
            self.floor += 1
            self._direction = Direction.UP
        else:
            self.floor -= 1
            self._direction = Direction.DOWN
        return Event.MOVED
