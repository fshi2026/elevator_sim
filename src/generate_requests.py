"""CLI: generate a random request CSV.

Usage:
    python src/generate_requests.py \
        --config examples/config.json \
        --num-passengers 50 \
        --duration 100 \
        --out out/inputs/run01.csv \
        --pattern uniform \
        [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from request_gen import (
    generate_evening_rush,
    generate_morning_rush,
    generate_uniform,
)
from sim_io import load_config, write_requests


PATTERNS = {
    "uniform": generate_uniform,
    "morning_rush": generate_morning_rush,
    "evening_rush": generate_evening_rush,
}


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    passengers = PATTERNS[args.pattern](
        num_passengers=args.num_passengers,
        num_floors=config.num_floors,
        duration=args.duration,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_requests(args.out, passengers)
    print(
        f"Wrote {len(passengers)} passengers to {args.out}"
        + (f" (seed={args.seed})" if args.seed is not None else "")
    )
    return 0


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random request CSV for the elevator simulator.",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to JSON config (num_floors is read from here).",
    )
    parser.add_argument("--num-passengers", type=int, required=True)
    parser.add_argument(
        "--duration", type=int, required=True,
        help="Request times sampled uniformly from [0, duration].",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, makes output deterministic.",
    )
    parser.add_argument(
        "--pattern", choices=sorted(PATTERNS.keys()), default="uniform",
        help="Request pattern (default: uniform). Other patterns model "
             "rush-hour-style workloads with origin or destination clustering.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
