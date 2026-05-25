"""Discrete-time tick loop that drives elevators + scheduler + passengers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from building import BuildingConfig
from elevator import (
    DropoffEvent,
    Elevator,
    Event,
    OvercapacityError,
)
from passenger import Passenger, PassengerState
from scheduler import Scheduler


class InvalidRequestError(ValueError):
    """A passenger request is malformed — origin == destination, or
    origin / destination outside `[1, num_floors]`. Raised by the
    Simulator at construction time so a bad workload fails before
    any tick fires."""


@dataclass(frozen=True)
class SimulationResult:
    """
    Fields:
        position_log:   one row per tick, including initial `t=0`.
                        Each row is `(t, [elev.floor for elev in elevators])`.
        passengers:     the same list passed in, now with timing fields
                        populated.
        final_tick:     the last `t` in the position log; equal to the
                        tick where the final passenger was dropped off.
    """

    position_log: list[tuple[int, list[int]]]
    passengers: list[Passenger]
    final_tick: int


class Simulator:
    def __init__(
        self,
        config: BuildingConfig,
        scheduler: Scheduler,
        passengers: list[Passenger],
        *,
        max_ticks: Optional[int] = None,   # upper bound on simulation length
    ) -> None:
        if len(scheduler.elevators) != config.num_elevators:
            raise ValueError(
                f"scheduler manages {len(scheduler.elevators)} elevators but "
                f"config.num_elevators = {config.num_elevators}"
            )
        for p in passengers:
            _validate_passenger(p, config.num_floors)
        self.config = config
        self.scheduler = scheduler
        self.passengers = passengers
        self.max_ticks = (
            max_ticks if max_ticks is not None
            else estimate_max_ticks(config, passengers)
        )

        # Group requests by reveal time. Each tick we drain the list at
        # that key in order — preserving input-file order for same-tick ties.
        self._pending_by_time: dict[int, list[Passenger]] = {}
        for p in passengers:
            self._pending_by_time.setdefault(p.request_time, []).append(p)

        # Lookup for event processing: events name passengers by id.
        self._by_id: dict[int, Passenger] = {p.id: p for p in passengers}

    def run(self) -> SimulationResult:
        elevators = self.scheduler.elevators

        # t=0: initial pre-tick state.
        position_log: list[tuple[int, list[int]]] = [
            (0, [e.floor for e in elevators])
        ]
        for p in self._reveal_at(0):
            self._by_id[p.id] = p
            self.scheduler.assign(p, current_time=0)

        t = 0
        while not self._all_dropped_off():
            t += 1
            if t > self.max_ticks:
                pending = [
                    p.id for p in self.passengers
                    if p.state != PassengerState.DELIVERED
                ]
                raise RuntimeError(
                    f"simulation exceeded max_ticks={self.max_ticks}; "
                    f"passengers still pending: {pending}"
                )

            # 1. Reveal new requests for this tick. Update _by_id so
            # events emitted in this tick (or later) can resolve the
            # passenger — this is the only path streaming variants have
            # to register new arrivals.
            for p in self._reveal_at(t):
                self._by_id[p.id] = p
                self.scheduler.assign(p, current_time=t)

            # 2. Step each elevator; process events on ARRIVED / LOADING.
            for elev in elevators:
                ev = elev.tick()
                if ev in (Event.ARRIVED, Event.LOADING):
                    self._process_events_at(elev, t)

            # 3. Log positions.
            position_log.append((t, [e.floor for e in elevators]))

        return SimulationResult(
            position_log=position_log,
            passengers=self.passengers,
            final_tick=t,
        )

    # ---- Event processing -----------------------------------------------

    def _process_events_at(self, elev: Elevator, t: int) -> None:
        """Pop and apply every front event whose `.floor == elev.floor`,
        then validate cap-safety at the stop boundary.
        """
        while elev.event_queue and elev.event_queue[0].floor == elev.floor:
            event = elev.pop_front()
            p = self._by_id[event.passenger_id]
            if isinstance(event, DropoffEvent):
                p.dropoff_time = t
            else:
                p.pickup_time = t

        pc = elev.passenger_count
        if pc > elev.capacity or pc < 0:
            raise OvercapacityError(
                f"Elevator {elev.id} at floor {elev.floor} (t={t}): "
                f"post-stop cabin count {pc} outside [0, {elev.capacity}]. "
                f"Scheduler produced an invalid event sequence."
            )

    def _reveal_at(self, t: int) -> Iterable[Passenger]:
        """Passengers whose `request_time` becomes visible at tick `t`."""
        return self._pending_by_time.get(t, ())

    def _all_dropped_off(self) -> bool:
        return all(p.state == PassengerState.DELIVERED for p in self.passengers)


def _validate_passenger(passenger: Passenger, num_floors: int) -> None:
    """Origin and destination must differ and lie in `[1, num_floors]`."""
    if passenger.origin == passenger.destination:
        raise InvalidRequestError(
            f"passenger {passenger.id}: origin == destination == {passenger.origin}"
        )
    if not 1 <= passenger.origin <= num_floors:
        raise InvalidRequestError(
            f"passenger {passenger.id}: origin {passenger.origin} outside "
            f"[1, {num_floors}]"
        )
    if not 1 <= passenger.destination <= num_floors:
        raise InvalidRequestError(
            f"passenger {passenger.id}: destination {passenger.destination} "
            f"outside [1, {num_floors}]"
        )


def estimate_max_ticks(
    config: BuildingConfig, passengers: list[Passenger]
) -> int:
    """Upper bound on simulation length given this workload.

    Assumes the worst case: a single elevator serves every passenger
    sequentially, each passenger costs at most:

        2 * load_time            # pickup + dropoff dwell
        + 2 * (num_floors - 1)   # worst-case travel + repositioning
    """
    if not passengers:
        return 0
    max_request = max(p.request_time for p in passengers)
    per_passenger_worst = 2 * config.load_time + 2 * (config.num_floors - 1)
    return max_request + len(passengers) * per_passenger_worst
