"""Unit tests for sim_io.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import csv
import json

import pytest

from building import BuildingConfig
from passenger import Passenger
from sim_io import (
    load_config,
    load_requests,
    load_passenger_stats,
    write_passenger_stats,
    write_position_log,
)


# ---- helpers --------------------------------------------------------------

def write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "requests.csv"
    p.write_text(content)
    return p


# ---- happy path -----------------------------------------------------------

def test_loads_basic_csv(tmp_path):
    csv_path = write(tmp_path, "0,1,1,51\n0,2,1,37\n10,3,20,1\n")
    pax = load_requests(csv_path)
    assert pax == [
        Passenger(id=1, origin=1, destination=51, request_time=0),
        Passenger(id=2, origin=1, destination=37, request_time=0),
        Passenger(id=3, origin=20, destination=1, request_time=10),
    ]


def test_preserves_same_tick_input_order(tmp_path):
    csv_path = write(tmp_path, "5,3,1,9\n5,1,2,8\n5,2,3,7\n")
    pax = load_requests(csv_path)
    assert [p.id for p in pax] == [3, 1, 2]


# ---- tolerant of header / blanks / comments -------------------------------

def test_skips_header_row(tmp_path):
    csv_path = write(tmp_path, "time,id,origin,dest\n0,1,2,8\n")
    pax = load_requests(csv_path)
    assert len(pax) == 1
    assert pax[0].id == 1


def test_skips_blank_lines(tmp_path):
    csv_path = write(tmp_path, "0,1,2,8\n\n5,2,3,7\n")
    pax = load_requests(csv_path)
    assert len(pax) == 2


def test_skips_comment_lines(tmp_path):
    csv_path = write(tmp_path, "# this is a comment\n0,1,2,8\n# another\n5,2,3,7\n")
    pax = load_requests(csv_path)
    assert len(pax) == 2


def test_tolerates_whitespace_around_fields(tmp_path):
    csv_path = write(tmp_path, " 0 , 1 ,  2 , 8 \n")
    pax = load_requests(csv_path)
    assert pax == [Passenger(id=1, origin=2, destination=8, request_time=0)]


# ---- error cases ----------------------------------------------------------

def test_rejects_wrong_field_count(tmp_path):
    csv_path = write(tmp_path, "0,1,2\n")
    with pytest.raises(ValueError, match="expected 4 fields"):
        load_requests(csv_path)


def test_rejects_non_integer_after_first_row_tolerance(tmp_path):
    """First malformed row is treated as header; second one is hard error."""
    csv_path = write(tmp_path, "header,row,goes,here\n0,1,2,8\nbad,row,here,too\n")
    with pytest.raises(ValueError, match="non-integer field"):
        load_requests(csv_path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_requests(tmp_path / "does-not-exist.csv")


def test_empty_file_returns_empty_list(tmp_path):
    csv_path = write(tmp_path, "")
    assert load_requests(csv_path) == []


# ---- load_config ---------------------------------------------------------

def test_load_config_minimal(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "num_floors": 10, "num_elevators": 2, "capacity": 5,
    }))
    cfg = load_config(path)
    assert cfg == BuildingConfig(num_floors=10, num_elevators=2, capacity=5)


def test_load_config_with_load_time(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "num_floors": 60, "num_elevators": 3, "capacity": 10, "load_time": 5,
    }))
    cfg = load_config(path)
    assert cfg.load_time == 5


def test_load_config_missing_required_field(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"num_floors": 10, "num_elevators": 2}))
    with pytest.raises(ValueError, match="missing required field"):
        load_config(path)


def test_load_config_invalid_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{ not valid json")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_config(path)


# ---- write_position_log --------------------------------------------------

def test_write_position_log_header_and_rows(tmp_path):
    out = tmp_path / "positions.csv"
    log = [
        (0, [1, 1]),
        (1, [2, 1]),
        (2, [3, 2]),
    ]
    write_position_log(out, log, num_elevators=2)
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time", "ele0_pos", "ele1_pos"]
    assert rows[1] == ["0", "1", "1"]
    assert rows[2] == ["1", "2", "1"]
    assert rows[3] == ["2", "3", "2"]


def test_write_position_log_empty_log(tmp_path):
    out = tmp_path / "positions.csv"
    write_position_log(out, [], num_elevators=3)
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows == [["time", "ele0_pos", "ele1_pos", "ele2_pos"]]


def test_write_position_log_roundtrip_through_csv_reader(tmp_path):
    """End-to-end: write a log, read it back, verify shape preserved."""
    out = tmp_path / "positions.csv"
    log = [(t, [t % 5 + 1, (t * 2) % 5 + 1]) for t in range(10)]
    write_position_log(out, log, num_elevators=2)
    with open(out) as f:
        reader = csv.reader(f)
        next(reader)  # header
        parsed = [(int(row[0]), [int(x) for x in row[1:]]) for row in reader]
    assert parsed == log


# ---- write_passenger_stats -----------------------------------------------

def _make_passenger(**overrides):
    """Build a Passenger with sensible defaults; overrides can fill in
    pickup/dropoff/etc. to mimic a post-simulation state."""
    defaults = dict(id=1, origin=1, destination=5, request_time=0)
    defaults.update(overrides)
    return Passenger(**defaults)


def test_write_passenger_stats_header_and_completed_row(tmp_path):
    out = tmp_path / "passengers.csv"
    p = _make_passenger(id=1, request_time=0, origin=1, destination=5)
    p.assigned_elevator = 0
    p.pickup_time = 1
    p.dropoff_time = 8

    write_passenger_stats(out, [p])

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "id", "request_time", "origin", "destination",
        "pickup_time", "dropoff_time",
        "wait_time", "total_time", "assigned_elevator",
    ]
    # wait_time = pickup - request = 1, total_time = dropoff - request = 8
    assert rows[1] == ["1", "0", "1", "5", "1", "8", "1", "8", "0"]


def test_write_passenger_stats_preserves_input_order(tmp_path):
    out = tmp_path / "passengers.csv"
    pax = []
    for pid in [3, 1, 2]:
        p = _make_passenger(id=pid)
        p.assigned_elevator = 0
        p.pickup_time = 1
        p.dropoff_time = 8
        pax.append(p)
    write_passenger_stats(out, pax)

    with open(out) as f:
        rows = list(csv.reader(f))[1:]   # drop header
    assert [r[0] for r in rows] == ["3", "1", "2"]


def test_write_passenger_stats_none_fields_become_empty(tmp_path):
    """If the sim didn't get to populate a field (shouldn't happen in
    practice for completed runs, but defensive), None → empty cell."""
    out = tmp_path / "passengers.csv"
    p = _make_passenger(id=1)
    # leave assigned_elevator / pickup_time / dropoff_time as None
    write_passenger_stats(out, [p])

    with open(out) as f:
        rows = list(csv.reader(f))
    # request_time/origin/destination populated; everything else empty.
    assert rows[1] == ["1", "0", "1", "5", "", "", "", "", ""]


def test_write_passenger_stats_roundtrip_via_dictreader(tmp_path):
    """Header column names match Passenger attributes, so DictReader keys
    map cleanly back to the model."""
    out = tmp_path / "passengers.csv"
    p = _make_passenger(id=42, origin=3, destination=7, request_time=10)
    p.assigned_elevator = 1
    p.pickup_time = 12
    p.dropoff_time = 19
    write_passenger_stats(out, [p])

    with open(out) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert int(row["id"]) == 42
    assert int(row["request_time"]) == 10
    assert int(row["origin"]) == 3
    assert int(row["destination"]) == 7
    assert int(row["pickup_time"]) == 12
    assert int(row["dropoff_time"]) == 19
    assert int(row["wait_time"]) == 2
    assert int(row["total_time"]) == 9
    assert int(row["assigned_elevator"]) == 1


def test_load_passenger_stats_roundtrips_via_write(tmp_path):
    """write_passenger_stats → load_passenger_stats → equivalent objects."""
    out = tmp_path / "passengers.csv"
    p = _make_passenger(id=42, origin=3, destination=7, request_time=10)
    p.assigned_elevator = 1
    p.pickup_time = 12
    p.dropoff_time = 19
    write_passenger_stats(out, [p])

    loaded = load_passenger_stats(out)
    assert len(loaded) == 1
    q = loaded[0]
    assert q.id == 42
    assert q.request_time == 10
    assert q.origin == 3
    assert q.destination == 7
    assert q.pickup_time == 12
    assert q.dropoff_time == 19
    assert q.assigned_elevator == 1
    # Derived properties also work after roundtrip.
    assert q.wait_time == 2
    assert q.total_time == 9
    assert q.travel_distance == 4


def test_load_passenger_stats_handles_empty_timing_fields(tmp_path):
    """An in-progress passenger with empty timing fields round-trips as Nones."""
    out = tmp_path / "passengers.csv"
    p = _make_passenger(id=1)
    write_passenger_stats(out, [p])

    loaded = load_passenger_stats(out)
    assert loaded[0].pickup_time is None
    assert loaded[0].dropoff_time is None
    assert loaded[0].assigned_elevator is None
    assert loaded[0].wait_time is None
    assert loaded[0].total_time is None
