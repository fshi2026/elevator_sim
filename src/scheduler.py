"""Scheduler base class.

Two-step assignment, used by every scheduler:

  **Step 1** — pick which elevator should serve this passenger.
  Subclasses implement `_pick_elevator(passenger, current_time) -> int`.

  **Step 2** — insert the new request into the chosen elevator's
  event queue. SAME for all schedulers; lives on the `Scheduler`
  base class.

"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import routing
from elevator import Direction, DropoffEvent, PickupEvent
from passenger import Passenger

if TYPE_CHECKING:
    from elevator import Elevator

log = logging.getLogger(__name__)


class Scheduler(ABC):
    """Abstract base for assignment policies.

    Bound to a fixed list of elevators at construction. Owns the
    ledger, the simulator-facing query methods, and the step-2
    queue-insertion primitive that every scheduler needs. Subclasses
    implement only `assign()`.
    """

    def __init__(self, elevators: list["Elevator"]) -> None:
        if not elevators:
            raise ValueError("Scheduler requires at least one elevator")
        self._elevators: list["Elevator"] = elevators
        self._n: int = len(elevators)
        self._bound: dict[int, list[Passenger]] = {}

    @property
    def elevators(self) -> list["Elevator"]:
        return self._elevators

    # ---- STEP 1: subclass contract ---------------------------------

    @abstractmethod
    def _pick_elevator(
        self, passenger: Passenger, current_time: int
    ) -> int:
        """Pick which elevator should serve `passenger`.

        Subclass responsibility — this is the only thing concrete
        schedulers must implement. Return the elevator's index in
        `self._elevators`.
        """

    # ---- assignment workflow (template method) --------------------------

    def assign(self, passenger: Passenger, current_time: int) -> int:
        idx = self._pick_elevator(passenger, current_time)
        elev = self._elevators[idx]
        elev.set_events(self._optimize_event_queue(elev, passenger))
        self._bind(idx, passenger)
        return idx

    # ---- ledger ----------------------------------------------------------

    def _bind(self, elev_idx: int, passenger: Passenger) -> None:
        passenger.assigned_elevator = elev_idx
        self._bound.setdefault(elev_idx, []).append(passenger)

    def bound_to(self, elev_idx: int) -> list[Passenger]:
        return list(self._bound.get(elev_idx, []))

    # ---- STEP 2: build the new event queue for one passenger -------------

    def _optimize_event_queue(
        self, elev: "Elevator", passenger: Passenger
    ) -> list:
        """Build the new `event_queue` for `elev` taking `passenger`.

        Constructs `PickupEvent` and `DropoffEvent` for the request and
        calls `routing.try_insert_pair`, which brute-force searches
        all direction-natural insertion positions and returns the
        cap-safe candidate with the lowest added travel cost. End-of-
        queue is always a valid fallback, so this returns a usable
        queue for any legal request when the existing queue is itself
        cap-safe (an invariant the scheduler maintains).
        """
        pickup = PickupEvent(passenger.id, passenger.origin)
        dropoff = DropoffEvent(passenger.id, passenger.destination)
        result = routing.try_insert_pair(
            elev.event_queue,
            elev.floor,
            elev.passenger_count,
            elev.capacity,
            pickup,
            dropoff,
        )
        if result is None:
            # Should be unreachable: end-of-queue insertion is always
            # direction-natural, and the existing queue is cap-safe by
            # induction (every prior assignment passed the same check).
            # Reaching here means an invariant was violated upstream.
            log.error(
                "try_insert_pair returned None for passenger %d on elev %d "
                "(floor=%d pc=%d, |queue|=%d). Falling back to raw append.",
                passenger.id, elev.id, elev.floor,
                elev.passenger_count, len(elev.event_queue),
            )
            return list(elev.event_queue) + [pickup, dropoff]
        return result


class RoundRobinScheduler(Scheduler):
    """Round-robin elevator pick.

    rotate cursor over elevators; pick the first
    that satisfies `_can_take` — projected direction matches the
    passenger's request AND pickup is reachable without backtracking.

    If no elev passes `_can_take`, fall back to the cursor's elev
    """

    def __init__(self, elevators: list["Elevator"]) -> None:
        super().__init__(elevators)
        self._cursor = 0

    def _pick_elevator(self, passenger: Passenger, current_time: int) -> int:
        for offset in range(self._n):
            idx = (self._cursor + offset) % self._n
            if _can_take(self._elevators[idx], passenger):
                self._cursor = (idx + 1) % self._n
                return idx

        # No elev's projected direction matched. Fall back to current elev
        idx = self._cursor
        log.debug(
            "t=%d: no _can_take match for passenger %d (%d->%d %s); "
            "step-2 falls to elev %d",
            current_time, passenger.id, passenger.origin, passenger.destination,
            passenger.direction.name, idx,
        )
        self._cursor = (idx + 1) % self._n
        return idx

def _can_take(elev: "Elevator", passenger: Passenger) -> bool:
    """direction match + pickup in reach."""
    elev_dir = elev.heading
    if elev_dir == Direction.IDLE:
        return True
    if elev_dir != passenger.direction:
        return False
    if elev_dir == Direction.UP:
        return passenger.origin >= elev.floor
    return passenger.origin <= elev.floor