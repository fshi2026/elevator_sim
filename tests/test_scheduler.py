"""Unit tests for scheduler.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from elevator import Direction, DropoffEvent, Elevator, PickupEvent
from passenger import Passenger
from routing import walk_capacity
from scheduler import (
    NearestScheduler,
    RoundRobinScheduler,
)


# ---- helpers --------------------------------------------------------------

def make_elevators(n: int, capacity: int = 10, load_time: int = 3) -> list[Elevator]:
    return [Elevator(id=i, capacity=capacity, load_time=load_time) for i in range(n)]


def req(passenger_id: int, origin: int, destination: int, t: int = 0) -> Passenger:
    return Passenger(id=passenger_id, origin=origin, destination=destination, request_time=t)


def floors_of(events) -> list[int]:
    return [e.floor for e in events]


def event_kinds(events) -> list[str]:
    return ["PU" if isinstance(e, PickupEvent) else "DO" for e in events]


# ---- passenger.direction --------------------------------------------------

def test_passenger_direction_set_at_construction():
    assert req(1, 2, 8).direction == Direction.UP
    assert req(2, 8, 2).direction == Direction.DOWN


# ---- validation -----------------------------------------------------------
#
# Per-request validation (origin != destination, floors in
# `[1, num_floors]`) lives on `Simulator.__init__` — see
# `test_simulator.py::test_simulator_rejects_*`. The Scheduler trusts
# whatever it's given.


def test_constructor_rejects_empty_elevators():
    with pytest.raises(ValueError):
        RoundRobinScheduler([])


# ---- single elevator: event queue grows correctly -------------------------

def test_single_elevator_always_bound_to_zero():
    elevs = make_elevators(1)
    s = RoundRobinScheduler(elevs)
    assert s.assign(req(1, 2, 8), current_time=0) == 0
    assert s.assign(req(2, 7, 3), current_time=1) == 0


def test_single_elevator_first_assign_installs_pickup_dropoff():
    elevs = make_elevators(1)
    s = RoundRobinScheduler(elevs)
    s.assign(req(1, 2, 9), current_time=0)
    assert floors_of(elevs[0].event_queue) == [2, 9]
    assert event_kinds(elevs[0].event_queue) == ["PU", "DO"]


def test_compatible_up_requests_merge_into_one_sweep():
    """Two UP requests with overlapping ranges merge into a single sweep —
    the brute-force prefers low travel cost, so the events naturally
    interleave into ascending floor order."""
    elevs = make_elevators(1)
    s = RoundRobinScheduler(elevs)
    s.assign(req(1, 2, 9), current_time=0)
    s.assign(req(2, 4, 7), current_time=0)
    floors = floors_of(elevs[0].event_queue)
    # Natural ascending order: 2, 4, 7, 9
    assert floors == [2, 4, 7, 9]


# ---- 2-elevator round-robin alternation -----------------------------------

def test_two_elevators_alternate_when_all_compatible():
    elevs = make_elevators(2)
    s = RoundRobinScheduler(elevs)
    assert s.assign(req(1, 2, 8), 0) == 0
    assert s.assign(req(2, 3, 7), 0) == 1
    assert s.assign(req(3, 4, 9), 0) == 0


# ---- "no compatible elevator" fallback to cursor's elev -------------------

def test_no_compat_falls_back_to_cursor_with_end_of_queue_insertion():
    """When neither elev's projected direction matches the new request,
    step 1 falls back to the cursor's elev. Step 2 still produces a
    valid cap-safe queue — the new request lands as an end-of-queue
    sweep after the current path drains."""
    elevs = make_elevators(2)
    s = RoundRobinScheduler(elevs)
    s.assign(req(1, 2, 9), 0)   # elev 0 → UP sweep
    s.assign(req(2, 3, 8), 0)   # elev 1 → UP sweep
    # Both elevs projected UP. New DOWN request 7→4 → no _can_take match.
    # Cursor is 0; fallback to elev 0; step-2 appends as new sweep.
    idx = s.assign(req(3, 7, 4), 0)
    assert idx == 0
    # Resulting queue contains all four floors; the new DOWN sweep is
    # appended after the existing UP sweep's events.
    floors = floors_of(elevs[0].event_queue)
    assert floors[-2:] == [7, 4]
    assert event_kinds(elevs[0].event_queue)[-2:] == ["PU", "DO"]


# ---- multi-trip pattern (the [5, 7, 5, 7] case) --------------------------

def test_stacked_origin_exceeds_capacity_produces_multi_trip_pattern():
    """4 passengers all 5→7, cap=3. The first 3 fit in one sweep; the
    4th can't (cap full) so the brute-force inserts at end-of-queue —
    a second visit to floor 5 and then to floor 7. That's the
    user's '[5, 7, 5, 7]' pattern, emerging naturally from cap-aware
    insertion (no special-case logic)."""
    elevs = make_elevators(n=1, capacity=3, load_time=1)
    s = RoundRobinScheduler(elevs)
    for i in range(4):
        s.assign(req(i, 5, 7), current_time=0)
    floors = floors_of(elevs[0].event_queue)
    # 3 pickups at 5, 3 dropoffs at 7, then one more pickup at 5, dropoff at 7.
    # Exact event order: PU PU PU at 5, then DO DO DO at 7, then PU at 5, DO at 7.
    assert floors == [5, 5, 5, 7, 7, 7, 5, 7]
    kinds = event_kinds(elevs[0].event_queue)
    assert kinds == ["PU", "PU", "PU", "DO", "DO", "DO", "PU", "DO"]
    # Cap-safety: walking from pc=0 never exceeds 3.
    _, max_pc = walk_capacity(elevs[0].event_queue, start_pc=0)
    assert max_pc == 3


def test_cap_aware_insertion_keeps_max_pc_at_or_below_cap():
    """Across an arbitrary mix of assignments, the scheduler maintains
    `max_pc <= capacity` on every elevator's queue — never installs an
    over-cap sequence. This is the invariant that makes
    OvercapacityError unreachable in normal operation."""
    elevs = make_elevators(n=2, capacity=3)
    s = RoundRobinScheduler(elevs)
    for i, (o, d) in enumerate([
        (5, 7), (5, 7), (5, 7), (5, 7), (5, 7),     # 5 pax all 5→7
        (3, 9), (8, 2), (1, 6), (6, 1), (9, 3),     # mixed
    ]):
        s.assign(req(i, o, d), current_time=0)
    for elev in elevs:
        _, max_pc = walk_capacity(elev.event_queue, start_pc=elev.passenger_count)
        assert max_pc <= elev.capacity, (
            f"elev{elev.id} queue has max_pc={max_pc} > cap={elev.capacity}: "
            f"{elev.event_queue}"
        )


# ---- ledger --------------------------------------------------------------

def test_assign_sets_passenger_assigned_elevator():
    elevs = make_elevators(2)
    s = RoundRobinScheduler(elevs)
    p = req(1, 2, 8)
    idx = s.assign(p, 0)
    assert p.assigned_elevator == idx


def test_bound_to_returns_in_bind_order():
    elevs = make_elevators(1)
    s = RoundRobinScheduler(elevs)
    p1 = req(1, 2, 5)
    p2 = req(2, 3, 6)
    s.assign(p1, 0)
    s.assign(p2, 0)
    assert s.bound_to(0) == [p1, p2]


# ---- end-to-end: scheduler + elevator coexist correctly -------------------

def test_end_to_end_two_passengers_one_elevator():
    """Bind two passengers, drive the elevator through its queue popping
    front events at each ARRIVED (the simulator's job in production).
    With event-based smart insertion, P2 (5→9) merges into P1's UP
    sweep: queue becomes [PU3, PU5, DO7, DO9] — single clean UP."""
    from elevator import Event

    elevs = make_elevators(1, capacity=2, load_time=1)
    s = RoundRobinScheduler(elevs)
    p1 = req(1, 3, 7)
    p2 = req(2, 5, 9)
    s.assign(p1, 0)
    s.assign(p2, 0)
    assert floors_of(elevs[0].event_queue) == [3, 5, 7, 9]

    e = elevs[0]
    by_id = {p1.id: p1, p2.id: p2}
    t = 0
    while True:
        t += 1
        ev = e.tick()
        if ev in (Event.ARRIVED, Event.LOADING):
            while e.event_queue and e.event_queue[0].floor == e.floor:
                event = e.pop_front()
                p = by_id[event.passenger_id]
                if isinstance(event, DropoffEvent):
                    p.dropoff_time = t
                else:
                    p.pickup_time = t
        if ev == Event.IDLE:
            break

    for p in (p1, p2):
        assert p.pickup_time is not None and p.dropoff_time is not None
        assert p.pickup_time < p.dropoff_time
    assert e.passenger_count == 0
    assert e.direction == Direction.IDLE


# =====================================================================
# NearestScheduler
# =====================================================================

def test_nearest_single_elevator_always_bound_to_zero():
    elevs = make_elevators(1)
    s = NearestScheduler(elevs)
    assert s.assign(req(1, 2, 8), current_time=0) == 0
    assert s.assign(req(2, 5, 1), current_time=1) == 0


def test_nearest_picks_closer_idle_elevator():
    """Two IDLE elevators at floors 1 and 30. A request at floor 25 UP
    should go to elev 1 (closer)."""
    elev_a = Elevator(id=0, capacity=10, load_time=3, floor=1)
    elev_b = Elevator(id=1, capacity=10, load_time=3, floor=30)
    s = NearestScheduler([elev_a, elev_b])
    # ETA(a) = |25-1| + 1 = 25; ETA(b) = |30-25| + 1 = 6.
    assert s.assign(req(1, 25, 35), current_time=0) == 1


def test_nearest_prefers_idle_over_loaded_even_if_farther():
    """Elev 0 (IDLE) at floor 30 vs elev 1 at floor 1 with a long queue
    going UP to 50. Request: pickup at 5 UP. Elev 0 has ETA = 30→5
    (need to come down) = 25 + 1 + ... wait, elev 0 is IDLE, so it
    just moves to 5 directly: ETA = 25 + 1 = 26.

    Elev 1 has its own UP sweep — pickup at 5 fits in-sweep so ETA
    is short (|5 - 1| + 1 = 5). Elev 1 wins."""
    elev_a = Elevator(id=0, capacity=10, load_time=3, floor=30)  # IDLE
    elev_b = Elevator(id=1, capacity=10, load_time=3, floor=1)
    elev_b.set_events([
        PickupEvent(passenger_id=99, floor=50),
        DropoffEvent(passenger_id=99, floor=55),
    ])
    s = NearestScheduler([elev_a, elev_b])
    # ETA(a) = 25 (move 30 → 5) + 1 (arrive) = 26.
    # ETA(b) = |5 - 1| (move 1 → 5) + 1 (arrive) = 5. Wins.
    assert s.assign(req(1, 5, 8), current_time=0) == 1


def test_nearest_addresses_RR_shortcoming_3_almost_free_elev():
    """RR's binary `_can_take` rejects elevators about to finish a sweep
    that go in the wrong direction; NearestScheduler doesn't — it scores
    them by ETA, which captures "wait for current sweep to drain, then
    reverse" naturally.

    Elev 0: heading UP at floor 25, one stop at 30 (~6 ticks to free
    including arrival).
    Elev 1: heading DOWN at floor 50, big queue (~50 ticks of work).
    Request: pickup at floor 5 UP.

    RR would reject both via `_can_take` (elev 0: 5 < 25; elev 1: DOWN
    mismatches UP) → fallback to cursor. Nearest computes ETAs:
      ETA(0) ≈ 6 (finish UP sweep) + 25 (move 30 → 5) + 1 = 32.
      ETA(1) much higher (process big DOWN queue first).
    Nearest picks elev 0 (correctly).
    """
    elev_a = Elevator(id=0, capacity=10, load_time=1, floor=25)
    elev_a.set_events([
        PickupEvent(passenger_id=99, floor=30),
        DropoffEvent(passenger_id=99, floor=30),
    ])
    elev_b = Elevator(id=1, capacity=10, load_time=1, floor=50)
    for i in range(10):
        elev_b.set_events([
            *elev_b.event_queue,
            PickupEvent(passenger_id=200 + i, floor=40 - i),
            DropoffEvent(passenger_id=200 + i, floor=10),
        ])
    s = NearestScheduler([elev_a, elev_b])
    assert s.assign(req(1, 5, 8), current_time=0) == 0


def test_nearest_load_balances_when_etas_differ():
    """Load balance is *implicit* in NearestScheduler: a busy elevator
    has a later ETA, so an empty elevator wins when the workload makes
    their ETAs differ.

    Sequence: first an UP request commits elev 0 to a long trip. Then
    a DOWN request that elev 0 couldn't reach quickly — elev 1 (still
    IDLE) wins.
    """
    elev_a = Elevator(id=0, capacity=10, load_time=3, floor=1)
    elev_b = Elevator(id=1, capacity=10, load_time=3, floor=1)
    s = NearestScheduler([elev_a, elev_b])
    # First UP request goes to elev 0 (tie, lowest-index wins).
    assert s.assign(req(1, 5, 50), current_time=0) == 0
    # Second request is DOWN from floor 8. Elev 0 is committed to a long
    # UP sweep — wrong direction + ETA includes returning. Elev 1 (IDLE)
    # has a much lower ETA. Elev 1 wins.
    assert s.assign(req(2, 8, 2), current_time=0) == 1


def test_nearest_ties_on_identical_requests_stay_on_first_elev():
    """When two IDLE elevs are at the same floor and the same request
    keeps coming, the new pickup is always at the same floor as the
    existing one (insert at front), so the ETA-from-current-floor is
    identical for both elevs. Tie → min picks the first.

    This is a real degenerate case: if you want strict load balancing
    for identical-pickup-floor bursts, a different scheduler is
    needed (NearestScheduler's ETA can't distinguish). Documented
    here so the behavior isn't surprising."""
    elev_a = Elevator(id=0, capacity=10, load_time=3, floor=1)
    elev_b = Elevator(id=1, capacity=10, load_time=3, floor=1)
    s = NearestScheduler([elev_a, elev_b])
    assignments = [s.assign(req(i, 5, 20), current_time=0) for i in range(4)]
    assert assignments == [0, 0, 0, 0]


def test_nearest_end_to_end_completes_all_passengers():
    """Smoke test: a small workload runs to completion with all
    passengers delivered. End-to-end correctness check."""
    from simulator import Simulator
    from building import BuildingConfig

    cfg = BuildingConfig(num_floors=20, num_elevators=2, capacity=4, load_time=2)
    passengers = [
        req(i, origin=o, destination=d, t=t)
        for i, (t, o, d) in enumerate([
            (0, 1, 10), (0, 5, 15), (1, 8, 2), (3, 12, 3),
            (5, 18, 1), (7, 3, 19), (10, 1, 7), (15, 15, 5),
        ])
    ]
    elevs = make_elevators(2, capacity=4, load_time=2)
    s = NearestScheduler(elevs)
    result = Simulator(cfg, s, passengers).run()
    assert all(p.dropoff_time is not None for p in passengers)
    assert result.final_tick == max(p.dropoff_time for p in passengers)
