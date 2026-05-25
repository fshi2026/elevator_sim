"""Generate N request CSVs with varying seeds and varying
workload size, for a chosen request pattern.

Output files: `<out-dir>/run_seed<NNN>.csv`.

Usage:
    python scripts/generate_corpus.py
    python scripts/generate_corpus.py --pattern morning_rush
    python scripts/generate_corpus.py --pattern evening_rush --count 30
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
DEFAULT_OUT_BASE = PROJECT_ROOT / "examples" / "generated"
DEFAULT_CONFIG = PROJECT_ROOT / "examples" / "config.json"
PATTERN_CHOICES = ("uniform", "morning_rush", "evening_rush")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"JSON config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--pattern", choices=PATTERN_CHOICES, default="uniform",
        help="Request pattern (default: uniform). Determines the default "
             "--out-dir if not given.",
    )
    parser.add_argument("--count", type=int, default=50, help="Number of files to generate.")
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Seeds will be seed-start .. seed-start + count - 1.",
    )
    parser.add_argument("--min-passengers", type=int, default=50)
    parser.add_argument("--max-passengers", type=int, default=200)
    parser.add_argument("--min-duration", type=int, default=100)
    parser.add_argument("--max-duration", type=int, default=300)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: examples/generated/<pattern>/).",
    )
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_BASE / args.pattern

    if args.min_passengers > args.max_passengers:
        parser.error("--min-passengers must be <= --max-passengers")
    if args.min_duration > args.max_duration:
        parser.error("--min-duration must be <= --max-duration")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen_script = SRC / "generate_requests.py"

    for i in range(args.count):
        seed = args.seed_start + i
        # Meta-RNG (separate from generation RNG) picks per-file params
        # deterministically from the seed.
        meta = random.Random(seed)
        n_pax = meta.randint(args.min_passengers, args.max_passengers)
        duration = meta.randint(args.min_duration, args.max_duration)

        out_csv = args.out_dir / f"run_seed{seed:03d}.csv"
        proc = subprocess.run(
            [
                sys.executable, str(gen_script),
                "--config", str(args.config),
                "--num-passengers", str(n_pax),
                "--duration", str(duration),
                "--seed", str(seed),
                "--out", str(out_csv),
                "--pattern", args.pattern,
            ],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(f"FAILED at seed={seed}: {proc.stderr}", file=sys.stderr)
            return 1
    print(
        f"Generated {args.count} files in {args.out_dir} "
        f"(pattern={args.pattern}, "
        f"passengers in [{args.min_passengers}, {args.max_passengers}], "
        f"duration in [{args.min_duration}, {args.max_duration}])"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
