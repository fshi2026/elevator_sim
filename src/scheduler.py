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

    # ---- subclass contract ---------------------------------

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
        # TODO: Constructs `PickupEvent` and `DropoffEvent` for the request
        # and optimize the elevator's event queue
        pass
