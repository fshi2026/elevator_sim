"""Compare two stress-test revisions side-by-side.

Reads `summary.csv` (per-seed) and `aggregate.csv` (per-scheduler)
from two revision directories produced by `stress_test.py`, and prints:

  1. **Aggregate diff** — for each headline metric, show A's value, B's
     value, and B - A, in both macro and micro aggregations. This is the
     "did scheduler B beat A overall?" answer.

  2. **Per-seed paired delta** — for each seed shared between the two
     runs, compute B's metric minus A's metric. Report the distribution
     of deltas (min / p25 / p50 / p75 / max) plus the count of seeds
     where B wins (delta < 0 for "lower is better" metrics like wait /
     overhead / makespan). Pairing controls for seed-to-seed workload
     variance, which can otherwise swing aggregate comparisons.

Usage:
    python scripts/compare_revisions.py out/corpus/REV_A out/corpus/REV_B

A sanity check: comparing a revision to itself prints all-zero deltas
and 0/N wins on every metric.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# "lower is better" — these metrics improve when going DOWN. B wins
# when delta < 0. Anything not in this list is treated the same (we
# only have lower-is-better metrics today; if a future "throughput"
# metric joins, add an opposite list).
LOWER_IS_BETTER = {
    "wait_mean", "wait_p50", "wait_p95", "wait_max",
    "total_mean", "total_p50", "total_p95", "total_max",
    "overhead_mean", "overhead_p50", "overhead_p95", "overhead_max",
    "makespan",
}

# Metrics to print in the aggregate diff table. Subset of aggregate.csv
# columns we care most about.
AGGREGATE_METRICS = [
    "wait_mean", "wait_p95", "wait_max",
    "total_mean", "total_p95",
    "overhead_mean", "overhead_p95", "overhead_max",
    "makespan",
]

# Metrics to print per-seed deltas for. Subset of summary.csv columns.
PER_SEED_METRICS = [
    "wait_mean", "wait_p95", "total_mean",
    "overhead_mean", "overhead_max", "makespan",
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision_a", type=Path, help="baseline revision dir")
    parser.add_argument("revision_b", type=Path, help="candidate revision dir")
    args = parser.parse_args(argv)

    for d in (args.revision_a, args.revision_b):
        if not (d / "summary.csv").exists() or not (d / "aggregate.csv").exists():
            print(f"error: {d} missing summary.csv and/or aggregate.csv", file=sys.stderr)
            return 2

    name_a = args.revision_a.name
    name_b = args.revision_b.name

    agg_a = _read_aggregate(args.revision_a / "aggregate.csv")
    agg_b = _read_aggregate(args.revision_b / "aggregate.csv")
    _print_aggregate_diff(name_a, name_b, agg_a, agg_b)

    sum_a = _read_summary(args.revision_a / "summary.csv")
    sum_b = _read_summary(args.revision_b / "summary.csv")
    _print_per_seed_diff(name_a, name_b, sum_a, sum_b)

    return 0


def _read_aggregate(path: Path) -> dict[str, dict[str, float]]:
    """Returns `{aggregation: {metric: value}}`. Aggregation is
    `macro` or `micro`."""
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            agg = row["aggregation"]
            out[agg] = {
                k: float(v) for k, v in row.items()
                if k not in ("aggregation", "num_seeds", "num_passengers_total")
                and v != ""
            }
    return out


def _read_summary(path: Path) -> dict[str, dict[str, float]]:
    """Returns `{seed: {metric: value}}` for PASS rows only."""
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "PASS":
                continue
            seed = row["seed"]
            out[seed] = {
                k: float(v) for k, v in row.items()
                if k not in ("seed", "status",
                             "num_passengers", "num_delivered")
                and v != ""
            }
    return out


def _print_aggregate_diff(
    name_a: str, name_b: str,
    agg_a: dict[str, dict[str, float]],
    agg_b: dict[str, dict[str, float]],
) -> None:
    print(f"=== Aggregate diff: {name_b} vs {name_a} ===")
    print(f"{'metric':<18} {'aggregation':<6} "
          f"{name_a:>12} {name_b:>12} {'delta':>10} {'%':>8}")
    print("-" * 80)
    for metric in AGGREGATE_METRICS:
        for agg_name in ("macro", "micro"):
            va = agg_a.get(agg_name, {}).get(metric)
            vb = agg_b.get(agg_name, {}).get(metric)
            if va is None or vb is None:
                continue
            delta = vb - va
            pct = (delta / va * 100) if va != 0 else 0.0
            arrow = "↓" if delta < 0 else ("↑" if delta > 0 else " ")
            print(f"{metric:<18} {agg_name:<6} "
                  f"{va:>12.1f} {vb:>12.1f} {delta:>+9.1f}{arrow} {pct:>+7.1f}%")
    print()


def _print_per_seed_diff(
    name_a: str, name_b: str,
    sum_a: dict[str, dict[str, float]],
    sum_b: dict[str, dict[str, float]],
) -> None:
    common = sorted(set(sum_a) & set(sum_b))
    only_a = set(sum_a) - set(sum_b)
    only_b = set(sum_b) - set(sum_a)
    if only_a or only_b:
        print(f"warning: seeds in only one revision (omitted from paired diff):")
        if only_a:
            print(f"  only in {name_a}: {sorted(only_a)}")
        if only_b:
            print(f"  only in {name_b}: {sorted(only_b)}")
        print()
    if not common:
        print("no shared seeds — nothing to compare per-seed.")
        return

    print(f"=== Per-seed paired delta (B - A) over {len(common)} shared seeds ===")
    print(f"{'metric':<16} {'b_wins':>10} "
          f"{'delta_min':>10} {'delta_p25':>10} {'delta_p50':>10} "
          f"{'delta_p75':>10} {'delta_max':>10}")
    print("-" * 88)
    for metric in PER_SEED_METRICS:
        deltas = []
        for seed in common:
            va = sum_a[seed].get(metric)
            vb = sum_b[seed].get(metric)
            if va is None or vb is None:
                continue
            deltas.append(vb - va)
        if not deltas:
            continue
        wins = sum(1 for d in deltas if d < 0) if metric in LOWER_IS_BETTER else \
               sum(1 for d in deltas if d > 0)
        ds = sorted(deltas)
        print(f"{metric:<16} {wins:>6}/{len(deltas):<3} "
              f"{ds[0]:>+10.1f} {_q(ds, 0.25):>+10.1f} {_q(ds, 0.5):>+10.1f} "
              f"{_q(ds, 0.75):>+10.1f} {ds[-1]:>+10.1f}")


def _q(sorted_xs: list[float], q: float) -> float:
    """Quantile with linear interpolation on a pre-sorted list."""
    if not sorted_xs:
        return 0.0
    rank = q * (len(sorted_xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = rank - lo
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * frac


if __name__ == "__main__":
    sys.exit(main())
