"""Smoke tests for the CLI entry point."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as main_module


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "examples/config.json"
REQUESTS_PATH = PROJECT_ROOT / "examples/sample_requests.csv"

def _load_config(path: Path) -> dict:
    """Read the JSON config used by the tests so assertions about the
    positions-CSV shape stay in sync with the bundled config (changing
    `num_elevators` in config.json shouldn't break these tests)."""
    with open(path) as f:
        return json.load(f)

# ---- in-process: call main() directly --------------------------------------

def test_main_runs_spec_example_end_to_end(tmp_path):
    """Calling main() with the bundled example produces valid CSVs."""
    config = _load_config(CONFIG_PATH)
    n_elev = config["num_elevators"]
    n_floors = config["num_floors"]

    out_dir = tmp_path / "out"
    exit_code = main_module.main([
        "--config", str(CONFIG_PATH),
        "--requests", str(REQUESTS_PATH),
        "--out-dir", str(out_dir),
    ])
    assert exit_code == 0

    # positions.csv
    positions_csv = out_dir / "positions.csv"
    assert positions_csv.exists()
    with open(positions_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time"] + [f"ele{i}_pos" for i in range(n_elev)]
    times = [int(r[0]) for r in rows[1:]]
    assert times == list(range(len(times)))
    for r in rows[1:]:
        for floor_str in r[1:]:
            assert 1 <= int(floor_str) <= n_floors

    # passengers.csv
    passengers_csv = out_dir / "passengers.csv"
    assert passengers_csv.exists()
    with open(passengers_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "id", "request_time", "origin", "destination",
        "pickup_time", "dropoff_time",
        "wait_time", "total_time", "assigned_elevator",
    ]
    # Three passengers, all completed (no empty cells).
    assert len(rows) == 4
    for r in rows[1:]:
        assert all(cell != "" for cell in r), f"unexpected empty cell in {r}"
    # Spot-check first row: id=0, request_time=0, request_time<=pickup<dropoff.
    first = rows[1]
    assert first[0] == "0"
    assert int(first[1]) <= int(first[2]) < int(first[3])


def test_main_creates_missing_out_dir(tmp_path):
    nested = tmp_path / "deep" / "nested" / "out"
    main_module.main([
        "--config", str(CONFIG_PATH),
        "--requests", str(REQUESTS_PATH),
        "--out-dir", str(nested),
    ])
    assert (nested / "positions.csv").exists()


def test_main_with_empty_requests(tmp_path):
    """No requests → terminates immediately, writes a CSV with just the
    initial-state row."""
    config = _load_config(CONFIG_PATH)
    n_elev = config["num_elevators"]

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    out_dir = tmp_path / "out"
    main_module.main([
        "--config", str(CONFIG_PATH),
        "--requests", str(empty),
        "--out-dir", str(out_dir),
    ])
    with open(out_dir / "positions.csv") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["time"] + [f"ele{i}_pos" for i in range(n_elev)],
        ["0"] + ["1"] * n_elev,
    ]


# ---- subprocess: confirms `python src/main.py` works ----------------------

def test_main_as_script(tmp_path):
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, str(SRC / "main.py"),
            "--config", str(CONFIG_PATH),
            "--requests", str(REQUESTS_PATH),
            "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    # stdout contains the summary block.
    assert "Simulation complete" in proc.stdout
    assert "passengers served:  3" in proc.stdout
    assert "wait time:" in proc.stdout
    assert "total time:" in proc.stdout
    assert "positions:" in proc.stdout
    assert "passenger stats:" in proc.stdout
    assert "log:" in proc.stdout
    assert (out_dir / "positions.csv").exists()
    assert (out_dir / "passengers.csv").exists()
    assert (out_dir / "simulation.log").exists()


def test_main_rejects_bad_config(tmp_path):
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text(json.dumps({"num_floors": 10}))   # missing required fields
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="missing required field"):
        main_module.main([
            "--config", str(bad_cfg),
            "--requests", str(REQUESTS_PATH),
            "--out-dir", str(out_dir),
        ])
