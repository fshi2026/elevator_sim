"""Run every CSV in examples/generated/ through main.py, persist
per-revision artifacts, and emit per-seed + per-scheduler summaries.

Workflow:
    python scripts/generate_corpus.py
    python scripts/stress_test.py

Output layout (one tree per revision; tagged by short git SHA so
back-to-back runs of different scheduler ideas don't overwrite each
other):

    out/corpus/<revision>/
        meta.json              ← cmdline, git SHA, dirty flag, timestamp
        summary.csv            ← per-seed stats (wait/total/overhead × stats)
        aggregate.csv          ← per-scheduler headline numbers
                                  (macro = mean-of-seed-means,
                                   micro = pooled-passenger stats)
        run_seed000/
            passengers.csv     ← from main.py — per-passenger timings
            positions.csv      ← from main.py — per-tick elev floors
            simulation.log     ← from main.py — INFO+ events
        run_seed001/
            ...

The `<revision>` defaults to `<short-sha>` (plus `-dirty` if the
working tree has uncommitted changes). Pass `--revision NAME` to
override (e.g. while iterating without committing). Future A/B
comparisons read `aggregate.csv` (or compare per-seed `summary.csv`)
— no need to re-run.

For each run that fails, prints the error tail. For each that passes,
prints a one-line summary. Exit code is the number of failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
DEFAULT_CORPUS = PROJECT_ROOT / "examples" / "generated" / "uniform"
DEFAULT_CONFIG = PROJECT_ROOT / "examples" / "config.json"
DEFAULT_OUT_BASE = PROJECT_ROOT / "out" / "corpus"

sys.path.insert(0, str(SRC))
from metrics import (
    RunStats,
    aggregate_runs,
    compute_run_stats,
)
from sim_io import load_config, load_passenger_stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help=f"Directory of request CSVs (default: {DEFAULT_CORPUS.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"JSON config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--out-base", type=Path, default=DEFAULT_OUT_BASE,
        help=f"Per-run output goes to <out-base>/<revision>/<filename-stem>/ "
             f"(default: {DEFAULT_OUT_BASE.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--scheduler", default="round_robin",
        help="Which scheduler to run (passed through to main.py). "
             "Default: round_robin.",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Label this run (default: derived from git short SHA + '-dirty' "
             "suffix if the working tree has uncommitted changes; falls back "
             "to a timestamp if not in a git repo). The scheduler name is "
             "automatically prefixed to the revision dir name so running "
             "two schedulers on the same SHA doesn't collide.",
    )
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"corpus dir not found: {args.corpus}", file=sys.stderr)
        print(f"  (try: python scripts/generate_corpus.py)", file=sys.stderr)
        return 2

    csvs = sorted(args.corpus.glob("*.csv"))
    if not csvs:
        print(f"no CSVs found in {args.corpus}", file=sys.stderr)
        return 2

    revision = args.revision or _derive_revision()
    # Include corpus name (= corpus dir basename, e.g. "uniform" or
    # "morning_rush") and scheduler name in the revision dir so we
    # can A/B per (corpus, scheduler) pair without collision.
    corpus_name = args.corpus.resolve().name
    revision_dir = f"{corpus_name}__{args.scheduler}__{revision}"
    out_root = args.out_base / revision_dir
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"corpus:    {corpus_name}")
    print(f"scheduler: {args.scheduler}")
    print(f"writing artifacts to: {out_root.relative_to(PROJECT_ROOT)}")

    fail_count = 0
    main_script = SRC / "main.py"
    config = load_config(args.config)

    # (name, out_dir, ok, RunStats|None, passengers|None) — collected so
    # we can write per-seed summary AND aggregate stats at the end.
    seed_results: "list[tuple[str, Path, bool, Optional[RunStats], Optional[list]]]" = []

    for csv_path in csvs:
        name = csv_path.stem
        out_dir = out_root / name
        proc = subprocess.run(
            [
                sys.executable, str(main_script),
                "--config", str(args.config),
                "--requests", str(csv_path),
                "--out-dir", str(out_dir),
                "--scheduler", args.scheduler,
            ],
            capture_output=True, text=True, check=False,
        )
        ok = proc.returncode == 0 and "Simulation complete" in proc.stdout
        if not ok:
            fail_count += 1
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            last = tail[-1] if tail else "(no output)"
            print(f"FAIL  {name}: {last}")
            seed_results.append((name, out_dir, False, None, None))
            continue

        # Reconstruct stats from the artifacts on disk — single source
        # of truth, no need to parse main.py's stdout.
        pax = load_passenger_stats(out_dir / "passengers.csv")
        makespan = _last_tick(out_dir / "positions.csv")
        try:
            stats = compute_run_stats(pax, config.load_time, makespan)
        except ValueError as e:
            print(f"FAIL  {name}: {e}")
            fail_count += 1
            seed_results.append((name, out_dir, False, None, None))
            continue

        seed_results.append((name, out_dir, True, stats, pax))
        print(
            f"PASS  {name}  pax={stats.num_delivered}  "
            f"tick={stats.makespan}  "
            f"wait_mean={stats.wait_mean:.1f}  "
            f"overhead_mean={stats.overhead_mean:.1f}"
        )

    _write_summary(out_root / "summary.csv", seed_results)
    _write_aggregate(out_root / "aggregate.csv", seed_results, config.load_time)
    _write_meta(out_root / "meta.json", revision, argv or sys.argv[1:], len(csvs), fail_count)

    print("---")
    print(f"{len(csvs) - fail_count}/{len(csvs)} passed; {fail_count} failed")
    print(f"per-seed:   {(out_root / 'summary.csv').relative_to(PROJECT_ROOT)}")
    print(f"aggregate:  {(out_root / 'aggregate.csv').relative_to(PROJECT_ROOT)}")
    return fail_count


def _derive_revision() -> str:
    """Short git SHA, with `-dirty` if the worktree has uncommitted changes.

    Falls back to a UTC timestamp if we're not in a git repo (or git is
    unavailable). Keeps the revision label out of the user's way: the
    common case is "I'm iterating, just tag this run somehow."
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _write_summary(
    path: Path,
    seed_results: "list[tuple[str, Path, bool, Optional[RunStats], Optional[list]]]",
) -> None:
    """One row per seed with all per-sim stats — for paired comparison
    against another revision's `summary.csv`."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "status", "num_passengers", "num_delivered", "makespan",
            "wait_mean", "wait_p50", "wait_p95", "wait_max",
            "total_mean", "total_p50", "total_p95", "total_max",
            "overhead_mean", "overhead_p50", "overhead_p95", "overhead_max",
            "ideal_mean",
        ])
        for name, _, ok, stats, _ in seed_results:
            if not ok or stats is None:
                w.writerow([name, "FAIL"] + [""] * 16)
                continue
            w.writerow([
                name, "PASS",
                stats.num_passengers, stats.num_delivered, stats.makespan,
                f"{stats.wait_mean:.1f}",
                f"{stats.wait_p50:.1f}",
                f"{stats.wait_p95:.1f}",
                stats.wait_max,
                f"{stats.total_mean:.1f}",
                f"{stats.total_p50:.1f}",
                f"{stats.total_p95:.1f}",
                stats.total_max,
                f"{stats.overhead_mean:.1f}",
                f"{stats.overhead_p50:.1f}",
                f"{stats.overhead_p95:.1f}",
                stats.overhead_max,
                f"{stats.ideal_mean:.1f}",
            ])


def _write_aggregate(
    path: Path,
    seed_results: "list[tuple[str, Path, bool, Optional[RunStats], Optional[list]]]",
    load_time: int,
) -> None:
    """Two rows: macro (mean-of-per-seed) and micro (pool-all-passengers).

    These are the headline numbers — what a future
    `compare_revisions.py` diffs to answer "did scheduler B beat A?"
    """
    runs = [
        (stats, pax) for _, _, ok, stats, pax in seed_results
        if ok and stats is not None and pax is not None
    ]
    if not runs:
        # Still emit a header so downstream parsers don't choke.
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["aggregation"])
        return

    agg = aggregate_runs(runs, load_time=load_time)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "aggregation", "num_seeds", "num_passengers_total",
            "wait_mean", "wait_p95", "wait_max",
            "total_mean", "total_p95",
            "overhead_mean", "overhead_p95", "overhead_max",
            "makespan",
        ])
        w.writerow([
            "macro", agg.num_seeds, agg.num_passengers_total,
            f"{agg.macro_wait_mean:.1f}",
            f"{agg.macro_wait_p95:.1f}",
            f"{agg.macro_wait_max:.1f}",
            f"{agg.macro_total_mean:.1f}",
            f"{agg.macro_total_p95:.1f}",
            f"{agg.macro_overhead_mean:.1f}",
            f"{agg.macro_overhead_p95:.1f}",
            f"{agg.macro_overhead_max:.1f}",
            f"{agg.macro_makespan:.1f}",
        ])
        w.writerow([
            "micro", agg.num_seeds, agg.num_passengers_total,
            f"{agg.micro_wait_mean:.1f}",
            f"{agg.micro_wait_p95:.1f}",
            agg.micro_wait_max,
            f"{agg.micro_total_mean:.1f}",
            f"{agg.micro_total_p95:.1f}",
            f"{agg.micro_overhead_mean:.1f}",
            f"{agg.micro_overhead_p95:.1f}",
            agg.micro_overhead_max,
            "",   # makespan is per-sim only; micro doesn't apply
        ])


def _last_tick(positions_csv: Path) -> int:
    """Final tick = last row's `time` column."""
    last = 0
    with open(positions_csv) as f:
        for row in csv.DictReader(f):
            last = int(row["time"])
    return last


def _write_meta(
    path: Path, revision: str, argv: list[str], num_seeds: int, fail_count: int
) -> None:
    meta = {
        "revision": revision,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "argv": argv,
        "num_seeds": num_seeds,
        "fail_count": fail_count,
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_dirty"] = bool(status)
        if status:
            meta["git_dirty_files"] = status.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
