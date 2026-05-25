"""CLI entry point for the elevator simulator.

Wires the four moving parts together:
  config (JSON) → Elevator(s) → Scheduler → Simulator
and writes the per-tick position log to disk + a compact summary to
stdout.

Usage:
    python src/main.py \
        --config examples/config.json \
        --requests examples/sample_requests.csv \
        --out-dir out/

Outputs (all under `--out-dir`):
  - `positions.csv`    per-tick elevator floors
  - `passengers.csv`   per-passenger timings (request → pickup → dropoff)
  - `simulation.log`   INFO+ events from the run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from elevator import Elevator
from scheduler import (
    NearestScheduler,
    RoundRobinScheduler,
    Scheduler,
)
from sim_io import (
    load_config,
    load_requests,
    write_passenger_stats,
    write_position_log,
)
from simulator import SimulationResult, Simulator


# Registry of scheduler classes selectable via `--scheduler NAME`.
SCHEDULERS: dict[str, type[Scheduler]] = {
    "round_robin": RoundRobinScheduler,
    "nearest": NearestScheduler,
}

def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "simulation.log"
    _configure_logging(log_path)

    config = load_config(args.config)
    passengers = load_requests(args.requests)

    elevators = [
        Elevator(id=i, capacity=config.capacity, load_time=config.load_time)
        for i in range(config.num_elevators)
    ]
    scheduler = SCHEDULERS[args.scheduler](elevators)
    result = Simulator(config, scheduler, passengers).run()

    positions_csv = args.out_dir / "positions.csv"
    passengers_csv = args.out_dir / "passengers.csv"
    write_position_log(positions_csv, result.position_log, config.num_elevators)
    write_passenger_stats(passengers_csv, result.passengers)

    _print_summary(result, positions_csv, passengers_csv, log_path, args.scheduler)
    return 0


def _configure_logging(log_path: Path) -> None:
    """Send INFO+ events from sim modules to <out-dir>/simulation.log.

    Resets root handlers each run so back-to-back invocations in the
    same Python process don't accumulate output across runs (matters
    for the in-process tests).
    """
    root = logging.getLogger()
    # Clear existing handlers (important for repeated in-process runs).
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the elevator simulator on a request CSV.",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to JSON config file (num_floors, num_elevators, capacity, [load_time]).",
    )
    parser.add_argument(
        "--requests", type=Path, required=True,
        help="Path to request CSV (time, id, origin, dest).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory for output files (created if missing).",
    )
    parser.add_argument(
        "--scheduler", choices=sorted(SCHEDULERS.keys()), default="round_robin",
        help="Which scheduler to run (default: round_robin).",
    )
    return parser.parse_args(argv)


def _print_summary(
    result: SimulationResult,
    positions_csv: Path,
    passengers_csv: Path,
    log_path: Path,
    scheduler_name: str,
) -> None:
    n = len(result.passengers)
    print("Simulation complete.")
    print(f"  scheduler:          {scheduler_name}")
    print(f"  passengers served:  {n}")
    print(f"  final tick:         {result.final_tick}")
    if n > 0:
        waits = [p.wait_time for p in result.passengers if p.wait_time is not None]
        totals = [p.total_time for p in result.passengers if p.total_time is not None]
        if waits:
            print(
                f"  wait time:   min={min(waits)}  max={max(waits)}  "
                f"mean={sum(waits) / len(waits):.1f}"
            )
        if totals:
            print(
                f"  total time:  min={min(totals)}  max={max(totals)}  "
                f"mean={sum(totals) / len(totals):.1f}"
            )
    print(f"  positions:          {positions_csv}")
    print(f"  passenger stats:    {passengers_csv}")
    print(f"  log:                {log_path}")


if __name__ == "__main__":
    sys.exit(main())
