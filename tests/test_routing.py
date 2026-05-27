"""Unit tests for routing.py.

`routing` provides the cap-aware walk and the brute-force insertion
search used by all schedulers. These tests cover the primitives in
isolation; scheduler-level integration is in `test_scheduler.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from elevator import Direction, DropoffEvent, PickupEvent
from routing import (
    assert_direction_natural,
    compute_etas,
    is_direction_natural,
    try_insert_pair,
    walk_capacity,
)


# ---- walk_capacity -------------------------------------------------------

def test_walk_capacity_empty():
    pcs, max_pc = walk_capacity([], start_pc=0)
    assert pcs == []
    assert max_pc == 0


def test_walk_capacity_empty_with_nonzero_start():
    """`max_pc` is at least the starting cabin count."""
    _, max_pc = walk_capacity([], start_pc=4)
    assert max_pc == 4


def test_walk_capacity_single_pickup_dropoff():
    events = [
        PickupEvent(passenger_id=1, floor=3),
        DropoffEvent(passenger_id=1, floor=7),
    ]
    pcs, max_pc = walk_capacity(events, start_pc=0)
    assert pcs == [1, 0]
    assert max_pc == 1


def test_walk_capacity_overlapping_pickups():
    """Two pickups before any dropoff stack the cabin count."""
    events = [
        PickupEvent(passenger_id=1, floor=2),
        PickupEvent(passenger_id=2, floor=4),
        DropoffEvent(passenger_id=1, floor=6),
        DropoffEvent(passenger_id=2, floor=8),
    ]
    pcs, max_pc = walk_capacity(events, start_pc=0)
    assert pcs == [1, 2, 1, 0]
    assert max_pc == 2


def test_walk_capacity_intra_stop_peaks_excluded_from_max_pc():
    """Order of events within a single shared-floor stop does NOT
    affect `max_pc`. Both orderings have the same entering count
    (start_pc = 1) and the same leaving count (1 + 1 − 1 = 1), so
    `max_pc` is 1 either way — even though pickup-first transiently
    has 2 inside the cabin.

    Intra-stop transients are allowed to exceed capacity briefly: the
    door is open, people swap through it. Only stop-boundary counts
    bind.
    """
    pickup_first = [
        PickupEvent(passenger_id=2, floor=5),
        DropoffEvent(passenger_id=1, floor=5),
    ]
    dropoff_first = [
        DropoffEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=5),
    ]
    _, peak_pu_first = walk_capacity(pickup_first, start_pc=1)
    _, peak_do_first = walk_capacity(dropoff_first, start_pc=1)
    assert peak_pu_first == 1
    assert peak_do_first == 1


def test_walk_capacity_max_pc_uses_stop_boundary_only():
    """A queue with intra-stop transients above the leaving count
    still reports `max_pc` based on the boundary, not the transient.
    Here, the stop at floor 5 has entering=1, intermediate=3 (after
    two PUs), leaving=1 (after two DOs cancel). `max_pc` = 1, not 3.
    """
    events = [
        PickupEvent(passenger_id=2, floor=5),
        PickupEvent(passenger_id=3, floor=5),
        DropoffEvent(passenger_id=1, floor=5),
        DropoffEvent(passenger_id=2, floor=5),
    ]
    _, max_pc = walk_capacity(events, start_pc=1)
    assert max_pc == 1


# ---- compute_etas --------------------------------------------------------
#
# (`projected_direction` moved off `routing` and onto Elevator as the
# `heading` property — tests for it live in test_elevator.py.)

def test_compute_etas_empty():
    assert compute_etas([], elev_floor=5, load_remaining=0, load_time=3) == []


def test_compute_etas_single_event_from_idle():
    """Elev at floor 1, idle, single PickupEvent at floor 5.

    4 MOVED ticks (1→2, 2→3, 3→4, 4→5) + 1 ARRIVED tick → ETA 5.
    """
    events = [PickupEvent(passenger_id=1, floor=5)]
    assert compute_etas(events, elev_floor=1, load_remaining=0, load_time=3) == [5]


def test_compute_etas_same_floor_events_share_dwell():
    """Three events all at floor 5; elev arrives once, all fire on
    the same ARRIVED tick."""
    events = [
        PickupEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=5),
        DropoffEvent(passenger_id=3, floor=5),
    ]
    etas = compute_etas(events, elev_floor=1, load_remaining=0, load_time=3)
    assert etas == [5, 5, 5]


def test_compute_etas_two_groups_with_post_dwell():
    """Floor 5 first, then floor 10. After the floor-5 ARRIVED tick,
    elev dwells `load_time - 1 = 2` LOADING ticks, then 5 MOVED ticks
    to floor 10, then ARRIVED at 10."""
    events = [
        PickupEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=10),
    ]
    # ETA(5)=5; then +2 (LOADING) + 5 (MOVED) + 1 (ARRIVED) → ETA(10)=13.
    assert compute_etas(
        events, elev_floor=1, load_remaining=0, load_time=3,
    ) == [5, 13]


def test_compute_etas_mid_dwell_at_current_floor():
    """Mid-dwell (load_remaining=2) at floor 5, event at floor 5 inserted
    at front (scheduler mid-dwell binding). Event fires on next tick
    (the next LOADING tick processes it)."""
    events = [PickupEvent(passenger_id=1, floor=5)]
    assert compute_etas(
        events, elev_floor=5, load_remaining=2, load_time=3,
    ) == [1]


def test_compute_etas_mid_dwell_to_next_floor():
    """Mid-dwell at floor 5 (load_remaining=2), next event at floor 10.
    Drain the remaining dwell (2 LOADING ticks), then move 5 floors,
    then ARRIVED → ETA = 2 + 5 + 1 = 8."""
    events = [PickupEvent(passenger_id=1, floor=10)]
    assert compute_etas(
        events, elev_floor=5, load_remaining=2, load_time=3,
    ) == [8]


def test_compute_etas_mid_dwell_same_floor_then_next():
    """Mid-dwell at floor 5 (load_remaining=2), one event at floor 5
    then one at floor 10. First event fires on next LOADING tick
    (ETA=1). Remaining dwell = load_remaining - 1 = 1 tick. Then
    5 MOVED + 1 ARRIVED → ETA(10) = 1 + 1 + 5 + 1 = 8."""
    events = [
        PickupEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=2, floor=10),
    ]
    assert compute_etas(
        events, elev_floor=5, load_remaining=2, load_time=3,
    ) == [1, 8]


# ---- is_direction_natural / assert_direction_natural ---------------------

def test_direction_natural_empty_queue():
    """No events, no trips to check — vacuously direction-natural."""
    assert is_direction_natural([], elev_floor=1) is True


def test_direction_natural_matched_up_pair():
    """PU/DO pair both in queue: direction derived from floor pair.
    Path 1→3→7 is monotonic UP for pid=1."""
    events = [
        PickupEvent(passenger_id=1, floor=3),
        DropoffEvent(passenger_id=1, floor=7),
    ]
    assert is_direction_natural(events, elev_floor=1) is True


def test_direction_natural_matched_down_pair():
    events = [
        PickupEvent(passenger_id=1, floor=8),
        DropoffEvent(passenger_id=1, floor=2),
    ]
    assert is_direction_natural(events, elev_floor=10) is True


def test_direction_natural_catches_new_passenger_backward_detour():
    """The strict-baseline bug: PU mid-queue (in first UP sweep), DO
    at end-of-queue (after a DOWN sweep). New passenger trip crosses
    the DOWN sweep — backward detour."""
    events = [
        PickupEvent(passenger_id=1, floor=5),
        PickupEvent(passenger_id=3, floor=30),  # new passenger PU
        DropoffEvent(passenger_id=1, floor=50),
        PickupEvent(passenger_id=2, floor=40),
        DropoffEvent(passenger_id=2, floor=10),
        DropoffEvent(passenger_id=3, floor=60),  # new passenger DO
    ]
    assert is_direction_natural(events, elev_floor=1) is False
    with pytest.raises(ValueError, match="passenger 3 .UP. backward detour"):
        assert_direction_natural(events, elev_floor=1)


def test_direction_natural_matched_pair_direction_from_floors_not_field():
    """For matched PU/DO pairs the field is intentionally ignored —
    direction comes from `DO.floor vs PU.floor`. So a deliberately
    mislabelled DO field doesn't fool the check."""
    events = [
        PickupEvent(passenger_id=1, floor=3),
        # Field says DOWN but floors say UP — utility uses floors.
        DropoffEvent(passenger_id=1, floor=7, direction=Direction.DOWN),
    ]
    assert is_direction_natural(events, elev_floor=1) is True


def test_direction_natural_orphan_do_up_passes_when_path_monotonic():
    """Orphan DO (passenger in cabin at queue-set time, no PU in
    queue) with direction UP and a monotonic UP path: OK."""
    events = [
        DropoffEvent(passenger_id=99, floor=20, direction=Direction.UP),
    ]
    assert is_direction_natural(events, elev_floor=5) is True


def test_direction_natural_orphan_do_up_fails_when_dragged_down():
    """Orphan DO UP passenger whose plan dips DOWN before reaching
    their DO floor — backward detour caught."""
    events = [
        DropoffEvent(passenger_id=99, floor=20, direction=Direction.UP),
        PickupEvent(passenger_id=1, floor=2),  # elevator dips to 2 — fine for pid=1 only
    ]
    # 5 -> 20 -> 2: pid=99 was in cabin during 5→20 (OK) and would
    # still be in cabin if their DO came later — but here DO at idx
    # 0 ends the trip before the dip. Trip ends at idx 0, so OK.
    assert is_direction_natural(events, elev_floor=5) is True

    # Now flip the order so pid=99 is still in cabin during the dip.
    events_bad = [
        PickupEvent(passenger_id=1, floor=2),
        DropoffEvent(passenger_id=99, floor=20, direction=Direction.UP),
    ]
    # elev=5 → 2 (DOWN, pid=99 still in cabin going UP) → 20.
    assert is_direction_natural(events_bad, elev_floor=5) is False


def test_direction_natural_orphan_do_idle_direction_skipped():
    """Orphan DO with IDLE (default) direction can't be verified —
    utility skips it rather than guess wrong. Other passengers are
    still checked."""
    events = [
        DropoffEvent(passenger_id=99, floor=2),  # default IDLE
        PickupEvent(passenger_id=1, floor=5),
        DropoffEvent(passenger_id=1, floor=10),
    ]
    # pid=99's trip 5→2 would be a DOWN trip but with IDLE we don't
    # check. pid=1 trip 5→10 UP is monotonic. → OK.
    assert is_direction_natural(events, elev_floor=5) is True


def test_direction_natural_multiple_in_cabin_passengers():
    """Multiple in-cabin passengers with different directions: both
    must remain direction-natural simultaneously."""
    events = [
        DropoffEvent(passenger_id=10, floor=15, direction=Direction.UP),
        DropoffEvent(passenger_id=11, floor=20, direction=Direction.UP),
    ]
    # elev=8 → 15 → 20: both UP passengers ride UP. OK.
    assert is_direction_natural(events, elev_floor=8) is True

    bad = [
        DropoffEvent(passenger_id=10, floor=20, direction=Direction.UP),
        DropoffEvent(passenger_id=11, floor=15, direction=Direction.DOWN),
    ]
    # elev=8 → 20 (UP, pid=11 dragged UP though they want DOWN) → 15.
    assert is_direction_natural(bad, elev_floor=8) is False


def test_assert_direction_natural_names_passenger_and_index():
    """Error message identifies the offending passenger id and the
    event index where monotonicity broke — so test/prod failures
    point straight at the bug."""
    events = [
        PickupEvent(passenger_id=7, floor=5),
        DropoffEvent(passenger_id=99, floor=2, direction=Direction.UP),  # orphan UP
        DropoffEvent(passenger_id=7, floor=10),
    ]
    # elev=5 → 5 (PU 7) → 2 (DO 99). pid=99 is UP but dragged from
    # 5 down to 2.
    with pytest.raises(ValueError) as exc_info:
        assert_direction_natural(events, elev_floor=5)
    msg = str(exc_info.value)
    assert "passenger 99" in msg
    assert "index 1" in msg


# ---- try_insert_pair (the workhorse) --------------------------------------

def test_insert_into_empty_queue():
    """Empty queue: insertion at position 0 + 1 — a brand-new sweep."""
    pu = PickupEvent(passenger_id=1, floor=3)
    do = DropoffEvent(passenger_id=1, floor=7)
    result = try_insert_pair(
        events=[], elev_floor=1, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    assert result == [pu, do]


def test_insert_natural_into_existing_up_sweep():
    """elev at 1, queue heads to 9 UP. Insert (3, 7) UP — both fit on the
    way up; the result is the natural in-sweep ordering."""
    existing = [
        PickupEvent(passenger_id=10, floor=9),
        DropoffEvent(passenger_id=10, floor=15),
    ]
    pu = PickupEvent(passenger_id=1, floor=3)
    do = DropoffEvent(passenger_id=1, floor=7)
    result = try_insert_pair(
        events=existing, elev_floor=1, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    floors = [e.floor for e in result]
    # Natural order: 3 (pickup), 7 (dropoff), 9 (existing pickup), 15 (existing dropoff)
    assert floors == [3, 7, 9, 15]


def test_insert_extends_sweep_terminus():
    """Existing UP sweep ends at 9. New dropoff at 15 just extends it."""
    existing = [
        PickupEvent(passenger_id=10, floor=5),
        DropoffEvent(passenger_id=10, floor=9),
    ]
    pu = PickupEvent(passenger_id=1, floor=7)
    do = DropoffEvent(passenger_id=1, floor=15)
    result = try_insert_pair(
        events=existing, elev_floor=1, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    floors = [e.floor for e in result]
    # All UP, ascending: 5, 7, 9, 15 (existing P10 dropoff still at 9
    # since dropoff slots before new dropoff at 15).
    assert floors == [5, 7, 9, 15]


def test_insert_down_request_with_up_queue_appends_as_new_sweep():
    """Existing queue is purely UP. DOWN request can't fit naturally; the
    fallback inserts at end-of-queue as a brand-new DOWN sweep."""
    existing = [
        PickupEvent(passenger_id=10, floor=5),
        DropoffEvent(passenger_id=10, floor=9),
    ]
    pu = PickupEvent(passenger_id=1, floor=8)
    do = DropoffEvent(passenger_id=1, floor=2)
    result = try_insert_pair(
        events=existing, elev_floor=1, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    # Pickup and dropoff end up at the tail (their exact pair position is
    # at the end of the resulting list since no in-sweep slot is direction-natural).
    assert result[-2] == pu
    assert result[-1] == do


def test_insert_into_later_sweep_no_positional_restriction():
    """elev at 10; queue does [UP to 20], [DOWN to 5], [UP to 50]. A new
    UP request at pickup=25 doesn't fit the FIRST UP sweep (end=20),
    but does fit the THIRD UP sweep (range 5 → 50). The "no
    positional restriction for later sweeps" property is what makes
    this work — the intervening DOWN sweep down to 5 doesn't block
    inserting at floor 25 in the next UP sweep.

    Each existing passenger's trip is per-passenger direction-natural
    (pid=10 UP 15→20, pid=11 DOWN 20→5, pid=12 UP 40→50), so the
    direction-natural filter accepts candidates whose ordering keeps
    that invariant.
    """
    existing = [
        PickupEvent(passenger_id=10, floor=15),
        DropoffEvent(passenger_id=10, floor=20),
        PickupEvent(passenger_id=11, floor=20),
        DropoffEvent(passenger_id=11, floor=5),
        PickupEvent(passenger_id=12, floor=40),
        DropoffEvent(passenger_id=12, floor=50),
    ]
    pu = PickupEvent(passenger_id=1, floor=25)
    do = DropoffEvent(passenger_id=1, floor=30)
    result = try_insert_pair(
        events=existing, elev_floor=10, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    assert result is not None
    floors = [e.floor for e in result]
    # PU(25) and DO(30) slot into the third UP sweep — after DO(11)@5,
    # before PU(12)@40. The first UP sweep (ending at 20) could only
    # accept them by extending its top to 30 and shoving PU(11)@20
    # into the DOWN split, but that's more expensive than the
    # within-fit insertion in the third UP sweep.
    assert floors == [15, 20, 20, 5, 25, 30, 40, 50]


def test_insert_respects_capacity():
    """cap=2. Existing queue already has two overlapping pickups. A third
    overlapping pickup cannot fit in the first sweep without exceeding
    cap; brute-force finds an end-of-queue slot instead (new sweep,
    cabin drains first)."""
    existing = [
        PickupEvent(passenger_id=10, floor=2),
        PickupEvent(passenger_id=11, floor=3),
        DropoffEvent(passenger_id=10, floor=8),
        DropoffEvent(passenger_id=11, floor=9),
    ]
    pu = PickupEvent(passenger_id=1, floor=4)
    do = DropoffEvent(passenger_id=1, floor=7)
    result = try_insert_pair(
        events=existing, elev_floor=1, elev_pc=0, capacity=2,
        pickup=pu, dropoff=do,
    )
    assert result is not None
    # The resulting walk must respect cap=2.
    _, max_pc = walk_capacity(result, start_pc=0)
    assert max_pc <= 2


def test_insert_returns_none_only_when_existing_queue_over_cap():
    """The end-of-queue position is always direction-natural, so a cap-safe
    existing queue always yields at least one valid insertion. Returning
    None signals that the existing queue itself isn't cap-safe — a
    precondition violation."""
    # Existing queue with max_pc > cap (a deliberately corrupted state).
    bad = [
        PickupEvent(passenger_id=10, floor=2),
        PickupEvent(passenger_id=11, floor=3),
        PickupEvent(passenger_id=12, floor=4),  # third overlapping pickup
        DropoffEvent(passenger_id=10, floor=8),
        DropoffEvent(passenger_id=11, floor=9),
        DropoffEvent(passenger_id=12, floor=10),
    ]
    _, max_pc = walk_capacity(bad, start_pc=0)
    assert max_pc == 3  # confirms the corruption

    pu = PickupEvent(passenger_id=1, floor=5)
    do = DropoffEvent(passenger_id=1, floor=7)
    result = try_insert_pair(
        events=bad, elev_floor=1, elev_pc=0, capacity=2,  # cap < existing max
        pickup=pu, dropoff=do,
    )
    assert result is None


def test_insert_chooses_lower_travel_cost():
    """When multiple cap-safe insertions exist, brute-force picks the one
    with the lowest added floor-distance. Inserting near the start of
    an UP sweep costs less than tacking it on the end."""
    # elev at 1; existing UP sweep [5, 10].
    existing = [
        PickupEvent(passenger_id=10, floor=5),
        DropoffEvent(passenger_id=10, floor=10),
    ]
    pu = PickupEvent(passenger_id=1, floor=3)
    do = DropoffEvent(passenger_id=1, floor=4)
    result = try_insert_pair(
        events=existing, elev_floor=1, elev_pc=0, capacity=10,
        pickup=pu, dropoff=do,
    )
    floors = [e.floor for e in result]
    # In-sweep insertion: 3, 4, 5, 10 (cost = 9 from floor 1).
    # End-of-queue would be 5, 10, 3, 4 (cost = 9 + 7 + 1 = much higher).
    assert floors == [3, 4, 5, 10]
