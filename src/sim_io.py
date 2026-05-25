"""I/O helpers for the elevator simulator.

Holds the request-CSV loader, the JSON config loader, the position-log
CSV writer, and the per-passenger stats CSV writer.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Union

from building import BuildingConfig
from passenger import Passenger


def load_requests(path: Union[str, Path]) -> list[Passenger]:
    """Read a request CSV and return passengers in input-file order.

    Same-tick ordering is preserved.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: on a row that has the right number of fields but
                    doesn't parse as integers (after the header-row
                    tolerance kicks in once at the start).
    """
    passengers: list[Passenger] = []
    seen_a_data_row = False
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            stripped = [cell.strip() for cell in row]
            if not stripped or not any(stripped):
                continue                                # blank line
            if stripped[0].startswith("#"):
                continue                                # comment line
            if len(stripped) != 4:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 fields (time, id, origin, dest), "
                    f"got {len(stripped)}: {row!r}"
                )
            try:
                t, pid, origin, dest = (int(c) for c in stripped)
            except ValueError:
                # First row that fails to parse is allowed to be a header.
                if not seen_a_data_row:
                    seen_a_data_row = True
                    continue
                raise ValueError(
                    f"{path}:{lineno}: non-integer field in {row!r}"
                )
            seen_a_data_row = True
            passengers.append(Passenger(
                id=pid, origin=origin, destination=dest, request_time=t
            ))
    return passengers


def load_config(path: Union[str, Path]) -> BuildingConfig:
    """Read a JSON config file and return a BuildingConfig.

    Required keys: num_floors, num_elevators, capacity.
    Optional: load_time (defaults to BuildingConfig's default).

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        ValueError: on missing required keys or invalid JSON.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e

    required = ("num_floors", "num_elevators", "capacity")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required field(s): {missing}")

    kwargs = {k: data[k] for k in required}
    if "load_time" in data:
        kwargs["load_time"] = data["load_time"]
    return BuildingConfig(**kwargs)


def write_position_log(
    path: Union[str, Path],
    position_log: list[tuple[int, list[int]]],
    num_elevators: int,
) -> None:
    """Write the per-tick position log to `path` as CSV.

    Header is `time, ele0_pos, ele1_pos, ...` (0-indexed elevator
    names, matching `Elevator.id` and `assigned_elevator` in
    passengers.csv).
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + [f"ele{i}_pos" for i in range(num_elevators)])
        for t, floors in position_log:
            writer.writerow([t, *floors])


def write_requests(
    path: Union[str, Path],
    passengers: list[Passenger],
) -> None:
    """Write `passengers` to a request CSV in the spec format."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "id", "origin", "dest"])
        for p in passengers:
            writer.writerow([p.request_time, p.id, p.origin, p.destination])


def write_passenger_stats(
    path: Union[str, Path],
    passengers: list[Passenger],
) -> None:
    """Write per-passenger trip data + timings to `path` as CSV.

    Columns: `id, request_time, origin, destination, pickup_time,
    dropoff_time, wait_time, total_time, assigned_elevator`. Rows
    preserve the input list's order.
    """
    def cell(v: Optional[int]) -> str:
        return "" if v is None else str(v)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "request_time", "origin", "destination",
            "pickup_time", "dropoff_time",
            "wait_time", "total_time", "assigned_elevator",
        ])
        for p in passengers:
            writer.writerow([
                p.id,
                p.request_time,
                p.origin,
                p.destination,
                cell(p.pickup_time),
                cell(p.dropoff_time),
                cell(p.wait_time),
                cell(p.total_time),
                cell(p.assigned_elevator),
            ])


def load_passenger_stats(path: Union[str, Path]) -> list[Passenger]:
    """Read a per-passenger stats CSV back into `Passenger` objects."""
    def opt_int(s: str) -> Optional[int]:
        return int(s) if s else None

    out: list[Passenger] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            p = Passenger(
                id=int(row["id"]),
                origin=int(row["origin"]),
                destination=int(row["destination"]),
                request_time=int(row["request_time"]),
            )
            p.pickup_time = opt_int(row["pickup_time"])
            p.dropoff_time = opt_int(row["dropoff_time"])
            p.assigned_elevator = opt_int(row["assigned_elevator"])
            out.append(p)
    return out
