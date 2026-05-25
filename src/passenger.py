"""Passenger model for the elevator simulator.

A Passenger represents a single Destination Dispatch request: origin
and destination are both known at request time.

`direction` is fixed at construction (UP if destination > origin, else
DOWN — the degenerate origin == destination case is rejected by the
scheduler at assign time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from elevator import Direction


class PassengerState(Enum):

    PENDING_ASSIGN = "pending_assign"    # no scheduler has bound them yet
    WAITING = "waiting"                  # bound to an elevator, not yet boarded
    IN_CABIN = "in_cabin"                # boarded, not yet dropped off
    DELIVERED = "delivered"              # done


@dataclass
class Passenger:

    id: int
    origin: int
    destination: int
    request_time: int

    direction: Direction = field(init=False)
    assigned_elevator: Optional[int] = None
    pickup_time: Optional[int] = None
    dropoff_time: Optional[int] = None

    def __post_init__(self) -> None:
        self.direction = (
            Direction.UP if self.destination > self.origin else Direction.DOWN
        )

    @property
    def state(self) -> PassengerState:
        if self.dropoff_time is not None:
            return PassengerState.DELIVERED
        if self.pickup_time is not None:
            return PassengerState.IN_CABIN
        if self.assigned_elevator is not None:
            return PassengerState.WAITING
        return PassengerState.PENDING_ASSIGN
    
    @property
    def travel_distance(self) -> int:
        return abs(self.destination - self.origin)

    @property
    def wait_time(self) -> Optional[int]:
        if self.pickup_time is None:
            return None
        return self.pickup_time - self.request_time

    @property
    def total_time(self) -> Optional[int]:
        if self.dropoff_time is None:
            return None
        return self.dropoff_time - self.request_time

    def ideal_total_time(self, load_time: int) -> int:
        return 2 * load_time + self.travel_distance

    def overhead(self, load_time: int) -> Optional[int]:
        """`total_time - ideal_total_time(load_time)` — the time the
        scheduler / contention added on top of this passenger's
        unavoidable trip cost. None if not yet delivered.
        """
        if self.total_time is None:
            return None
        return self.total_time - self.ideal_total_time(load_time)