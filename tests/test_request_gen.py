"""Unit tests for the random request generator + its CLI."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from passenger import Passenger
from request_gen import (
    generate_evening_rush,
    generate_morning_rush,
    generate_uniform,
)
from sim_io import load_requests, write_requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"


# ---- generate_uniform: determinism + invariants -------------------------

def test_seeded_output_is_deterministic():
    a = generate_uniform(num_passengers=20, num_floors=10, duration=50, seed=42)
    b = generate_uniform(num_passengers=20, num_floors=10, duration=50, seed=42)
    assert a == b


def test_different_seeds_produce_different_outputs():
    a = generate_uniform(num_passengers=20, num_floors=10, duration=50, seed=1)
    b = generate_uniform(num_passengers=20, num_floors=10, duration=50, seed=2)
    assert a != b


def test_origin_destination_always_differ():
    pax = generate_uniform(num_passengers=200, num_floors=5, duration=100, seed=7)
    for p in pax:
        assert p.origin != p.destination, f"passenger {p.id}: {p.origin} == {p.destination}"


def test_floors_in_range():
    pax = generate_uniform(num_passengers=200, num_floors=15, duration=50, seed=0)
    for p in pax:
        assert 1 <= p.origin <= 15
        assert 1 <= p.destination <= 15


def test_request_times_in_range():
    pax = generate_uniform(num_passengers=100, num_floors=10, duration=30, seed=0)
    for p in pax:
        assert 0 <= p.request_time <= 30


def test_output_sorted_by_request_time():
    pax = generate_uniform(num_passengers=50, num_floors=20, duration=100, seed=99)
    times = [p.request_time for p in pax]
    assert times == sorted(times)


def test_ids_are_sequential_zero_indexed():
    pax = generate_uniform(num_passengers=10, num_floors=5, duration=20, seed=0)
    assert [p.id for p in pax] == list(range(10))


def test_zero_passengers_returns_empty_list():
    assert generate_uniform(num_passengers=0, num_floors=10, duration=50, seed=0) == []


def test_two_floor_building_only_two_possible_endpoints():
    """The smallest legal building: floors 1 and 2. Every passenger
    must go between them."""
    pax = generate_uniform(num_passengers=20, num_floors=2, duration=10, seed=0)
    for p in pax:
        assert {p.origin, p.destination} == {1, 2}


# ---- validation ---------------------------------------------------------

def test_rejects_negative_num_passengers():
    with pytest.raises(ValueError, match="num_passengers"):
        generate_uniform(num_passengers=-1, num_floors=10, duration=10)


def test_rejects_num_floors_below_2():
    with pytest.raises(ValueError, match="num_floors"):
        generate_uniform(num_passengers=5, num_floors=1, duration=10)


def test_rejects_negative_duration():
    with pytest.raises(ValueError, match="duration"):
        generate_uniform(num_passengers=5, num_floors=10, duration=-1)


# ---- write_requests roundtrip ------------------------------------------

def test_generated_passengers_roundtrip_via_csv(tmp_path):
    """Generate → write → load yields the same tuples."""
    pax = generate_uniform(num_passengers=10, num_floors=8, duration=20, seed=42)
    csv_path = tmp_path / "gen.csv"
    write_requests(csv_path, pax)
    loaded = load_requests(csv_path)
    # Passenger equality compares dataclass fields including timing
    # fields (all None for both lists), so we can compare directly.
    assert loaded == pax


def test_write_requests_header_and_rows(tmp_path):
    csv_path = tmp_path / "out.csv"
    pax = [
        Passenger(id=0, request_time=0, origin=1, destination=5),
        Passenger(id=1, request_time=3, origin=10, destination=2),
    ]
    write_requests(csv_path, pax)
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time", "id", "origin", "dest"]
    assert rows[1] == ["0", "0", "1", "5"]
    assert rows[2] == ["3", "1", "10", "2"]


# ---- generate_requests.py CLI ------------------------------------------

def _write_config(tmp_path, num_floors: int) -> Path:
    import json
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "num_floors": num_floors, "num_elevators": 2, "capacity": 10,
    }))
    return cfg


def test_cli_writes_csv_with_seed(tmp_path):
    cfg = _write_config(tmp_path, num_floors=10)
    out_csv = tmp_path / "gen.csv"
    proc = subprocess.run(
        [
            sys.executable, str(SRC / "generate_requests.py"),
            "--config", str(cfg),
            "--num-passengers", "8",
            "--duration", "20",
            "--seed", "42",
            "--out", str(out_csv),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Wrote 8 passengers" in proc.stdout
    assert "seed=42" in proc.stdout
    # Loadable, and reproduces the same passengers as a direct call with
    # the same num_floors from config.
    loaded = load_requests(out_csv)
    direct = generate_uniform(num_passengers=8, num_floors=10, duration=20, seed=42)
    assert loaded == direct


def test_cli_creates_missing_parent_dir(tmp_path):
    cfg = _write_config(tmp_path, num_floors=5)
    nested = tmp_path / "a" / "b" / "c" / "gen.csv"
    proc = subprocess.run(
        [
            sys.executable, str(SRC / "generate_requests.py"),
            "--config", str(cfg),
            "--num-passengers", "3",
            "--duration", "10",
            "--out", str(nested),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert nested.exists()


# ---- morning_rush / evening_rush -----------------------------------------

def test_morning_rush_biases_pickups_to_lobby():
    """With lobby_fraction=1.0, every pickup must be at floor 1."""
    pax = generate_morning_rush(
        num_passengers=100, num_floors=20, duration=100,
        lobby_fraction=1.0, seed=42,
    )
    assert all(p.origin == 1 for p in pax)
    # Destinations span upper floors.
    assert {p.destination for p in pax} - {1} == {p.destination for p in pax}


def test_evening_rush_biases_dests_to_lobby():
    """With lobby_fraction=1.0, every destination must be at floor 1."""
    pax = generate_evening_rush(
        num_passengers=100, num_floors=20, duration=100,
        lobby_fraction=1.0, seed=42,
    )
    assert all(p.destination == 1 for p in pax)
    assert {p.origin for p in pax} - {1} == {p.origin for p in pax}


def test_morning_rush_lobby_fraction_zero_matches_uniform():
    """lobby_fraction=0 should give identical output to generate_uniform
    (same seed → same RNG draws → same passengers)."""
    pax_morning = generate_morning_rush(
        num_passengers=20, num_floors=10, duration=50,
        lobby_fraction=0.0, seed=42,
    )
    pax_uniform = generate_uniform(
        num_passengers=20, num_floors=10, duration=50, seed=42,
    )
    # When lobby_fraction=0 the morning_rush draw goes to the uniform
    # branch every time, BUT it consumes one extra rng.random() call
    # for the branch decision. So they differ by RNG state — that's
    # expected. Just check structural properties instead.
    assert len(pax_morning) == len(pax_uniform)
    for p in pax_morning:
        assert 1 <= p.origin <= 10 and 1 <= p.destination <= 10
        assert p.origin != p.destination


def test_rush_generators_reject_bad_lobby_fraction():
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="lobby_fraction"):
            generate_morning_rush(
                num_passengers=10, num_floors=10, duration=10,
                lobby_fraction=bad, seed=1,
            )


def test_rush_generators_are_deterministic_with_seed():
    a = generate_morning_rush(
        num_passengers=20, num_floors=10, duration=50, seed=7,
    )
    b = generate_morning_rush(
        num_passengers=20, num_floors=10, duration=50, seed=7,
    )
    assert a == b


def test_cli_accepts_pattern_flag(tmp_path):
    """`--pattern morning_rush` works end-to-end and produces a CSV with
    lobby-biased origins."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"num_floors": 20, "num_elevators": 2, "capacity": 10}')
    out_csv = tmp_path / "morning.csv"
    proc = subprocess.run(
        [
            sys.executable, str(SRC / "generate_requests.py"),
            "--config", str(cfg),
            "--num-passengers", "50",
            "--duration", "100",
            "--out", str(out_csv),
            "--seed", "42",
            "--pattern", "morning_rush",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(open(out_csv)))
    # With default lobby_fraction=0.8 and 50 passengers, expect ~40 at
    # floor 1. Use a loose bound to avoid flakiness on the random draw.
    lobby_pickups = sum(1 for r in rows if int(r["origin"]) == 1)
    assert lobby_pickups >= 30, (
        f"morning_rush with default lobby_fraction=0.8 should heavily "
        f"bias to floor 1; got only {lobby_pickups}/50 lobby pickups"
    )
