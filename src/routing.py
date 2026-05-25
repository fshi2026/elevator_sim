"""Routing primitives for event-based schedulers.

The Elevator stores its plan as a sequence of named events
(`PickupEvent` / `DropoffEvent`). The scheduler's job is to
insert a new passenger's `(pickup, dropoff)` event pair into the
chosen elevator's queue such that:

  1. Pickup precedes dropoff (obviously).
  2. Walking the resulting sequence, cabin occupancy never exceeds
     capacity.
  3. The insertion is "direction-natural" — the elevator naturally
     passes through the pickup floor and dropoff floor along its
     existing path, with no backward detour.

A note on "direction-natural"
-----------------------------
Inserting a pickup at position `i` means: between
`events[i-1].floor` (or `elev_floor` if `i == 0`) and
`events[i].floor`, the elevator passes through the new floor. To be
direction-natural, the new floor must lie between those two floors in
the same monotonic direction the elevator is already traveling
through that segment. End-of-queue insertions are unconditionally
allowed — they're new sweeps tacked on after everything else, so
"direction-natural" is trivially satisfied.
"""
from __future__ import annotations

from typing import Optional, Sequence

from elevator import Direction, DropoffEvent, ElevatorEvent, PickupEvent


def walk_capacity(
    events: Sequence[ElevatorEvent], start_pc: int
) -> tuple[list[int], int]:
    """Walk `events`; return `(pc_after_each_event, max_pc_at_stop_boundaries)`.

    `pcs[i]` is the cabin count after applying `events[i]` (queue
    order). `max_pc` is the largest cabin count observed at any STOP
    BOUNDARY — i.e. the count entering each contiguous same-floor
    group and the count leaving the final group, plus `start_pc`.
    Transient cabin counts WITHIN a shared-floor stop (e.g. after
    one of three pickups while two dropoffs are still pending) are
    intentionally excluded from `max_pc`.
    """
    pc = start_pc
    max_pc = pc                     # start_pc is the entering count of stop 0
    pcs: list[int] = []
    if not events:
        return pcs, max_pc

    last_floor = events[0].floor
    for e in events:
        if e.floor != last_floor:
            # Stop boundary: pc here is the count LEAVING the prior
            # stop and ENTERING the new one. Both must be ≤ capacity.
            if pc > max_pc:
                max_pc = pc
            last_floor = e.floor
        if isinstance(e, PickupEvent):
            pc += 1
        else:
            pc -= 1
        pcs.append(pc)
    # Final stop's leaving count must also clear capacity.
    if pc > max_pc:
        max_pc = pc
    return pcs, max_pc


def try_insert_pair(
    events: Sequence[ElevatorEvent],
    elev_floor: int,
    elev_pc: int,
    capacity: int,
    pickup: PickupEvent,
    dropoff: DropoffEvent,
) -> Optional[list[ElevatorEvent]]:
    """Insert `(pickup, dropoff)` into `events` at the first cap-safe
    direction-natural position.

    Searches `(pickup_pos, dropoff_pos)` pairs with `dropoff_pos >
    pickup_pos`, smallest first; returns the first one whose walk
    keeps `max_pc <= capacity`. End-of-queue insertions are always
    direction-natural by construction, so a cap-safe existing queue
    always yields at least one valid candidate; `None` signals a
    precondition violation (existing queue isn't cap-safe) — surface
    it as a real bug.
    """

    events_list = list(events)
    n = len(events_list)
    direction = (
        Direction.UP if dropoff.floor > pickup.floor else Direction.DOWN
    )

    # floors[i] = floor the elev is "at" when about to process events[i]:
    # floors[0] = elev_floor, floors[i] = events[i-1].floor for i > 0.
    floors = [elev_floor] + [e.floor for e in events_list]

    for pi in range(n + 1):
        if not _can_insert_at(floors, pi, pickup.floor, direction):
            continue
        events_after_pickup = events_list[:pi] + [pickup] + events_list[pi:]
        floors_after = [elev_floor] + [e.floor for e in events_after_pickup]
        for di in range(pi + 1, n + 2):
            if not _can_insert_at(
                floors_after, di, dropoff.floor, direction
            ):
                continue
            candidate = (
                events_after_pickup[:di] + [dropoff] + events_after_pickup[di:]
            )
            _, max_pc = walk_capacity(candidate, elev_pc)
            if max_pc <= capacity:
                return candidate

    return None


def _can_insert_at(
    floors: list[int], i: int, new_floor: int, direction: Direction
) -> bool:
    """True iff inserting an event at floor `new_floor` at position `i`
    in the event sequence is direction-natural.

    Same-floor matches (prev == new or next == new) are allowed: the
    elevator is already stopping there, no detour to add another event.
    """
    if i >= len(floors) - 1:
        return True  # end-of-queue: always allowed (new sweep)
    prev = floors[i]
    nxt = floors[i + 1]
    if prev == new_floor or nxt == new_floor:
        return True
    if direction == Direction.UP:
        return prev < nxt and prev <= new_floor <= nxt
    return prev > nxt and prev >= new_floor >= nxt

def compute_etas(
    events: Sequence[ElevatorEvent],
    elev_floor: int,
    load_remaining: int,
    load_time: int,
) -> list[int]:
    """Tick offsets from "now" for each event in `events`.

    Returns a list of length `len(events)`.
    """
    etas: list[int] = []
    if not events:
        return etas

    t = 0
    cur_floor = elev_floor
    is_mid_dwell = load_remaining > 0

    i = 0
    while i < len(events):
        target = events[i].floor
        if target != cur_floor:
            # Need to move. If mid-dwell, drain it first.
            if is_mid_dwell:
                t += load_remaining
                is_mid_dwell = False
            t += abs(target - cur_floor)   # MOVED ticks
            t += 1                         # ARRIVED tick (events fire here)
        else:
            # Already at target — events fire on the next tick
            # (ARRIVED if not mid-dwell, LOADING if mid-dwell;
            # either way, 1 tick from now).
            t += 1

        # Same-floor batch.
        while i < len(events) and events[i].floor == target:
            etas.append(t)
            i += 1

        # Post-event dwell before next move, if any events remain.
        if i < len(events):
            if target == cur_floor and is_mid_dwell:
                # Continuing the in-flight dwell: 1 tick was the
                # firing tick; the rest of the dwell is what was
                # already scheduled.
                t += load_remaining - 1
                is_mid_dwell = False
            else:
                # Fresh ARRIVED → standard post-arrival dwell.
                t += load_time - 1

        cur_floor = target

    return etas
