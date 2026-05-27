# Elevator Simulator

A discrete-time simulation of a Destination Dispatch elevator system —
passengers request a destination at the time of pickup, the controller
binds them to one specific elevator immediately, and that binding can't
change later. Pluggable scheduler interface for comparing assignment
policies on the same workload.

## Requirements

- Python 3.9+ (no third-party runtime deps; uses only stdlib).
- `pytest` for tests.

```sh
pip install -r requirements-dev.txt   # pulls pytest
```

`requirements.txt` is intentionally empty — the runtime depends only on
the standard library.

## Running tests

From the project root:

```sh
pytest
```

## A smoke run

The simulator is driven by two files: a JSON building config and a
request CSV. A 3-passenger example is committed:

```sh
python src/main.py \
    --config examples/config.json \
    --requests examples/sample_requests.csv \
    --out-dir out/smoke \
    --scheduler nearest
```

Output: a one-screen summary to stdout, plus three files under
`out/smoke/`:

- `positions.csv`     — per-tick `(time, ele0_pos, ele1_pos, ...)`.
- `passengers.csv`    — per-passenger request/pickup/dropoff times,
                        wait/total time, assigned elevator.
- `simulation.log`    — debug info.

Try `--scheduler round_robin` on the same input to see how the
assignment policy changes the outcome.

## Generating synthetic requests

Three workload patterns are built in: `uniform`, `morning_rush`,
`evening_rush`. A single CSV at a time:

```sh
python src/generate_requests.py \
    --config examples/config.json \
    --num-passengers 100 \
    --duration 200 \
    --seed 42 \
    --pattern morning_rush \
    --out out/inputs/morning100.csv
```

`--seed` makes the output deterministic (skip for non-deterministic).

## Generating a corpus for stress testing

A corpus is a bank of fixed CSVs with varying seeds and workload sizes,
so every scheduler is graded on the same inputs:

```sh
python scripts/generate_corpus.py --pattern uniform       # 50 CSVs
python scripts/generate_corpus.py --pattern morning_rush
python scripts/generate_corpus.py --pattern evening_rush
```

Output: `examples/generated/<pattern>/run_seedNNN.csv` (50 files, seeds
0–49 by default). These are gitignored — reproducible from the seed.

## Stress test: one scheduler across a whole corpus

```sh
python scripts/stress_test.py \
    --corpus examples/generated/uniform \
    --scheduler nearest
```

Writes per-revision artifacts to
`out/corpus/<corpus>__<scheduler>__<git-short-sha>[-dirty]/`:

```
out/corpus/uniform__nearest__abc1234-dirty/
├── meta.json         git SHA + dirty flag + argv + timestamp
├── summary.csv       per-seed stats (wait_mean/p50/p95/max + makespan + ...)
├── aggregate.csv     macro + micro roll-up across all seeds
├── run_seed000/
│   ├── passengers.csv
│   ├── positions.csv
│   └── simulation.log
├── run_seed001/
│   └── ...
└── ...
```

The dir is tagged by git SHA + scheduler + corpus name, so back-to-back
runs of different scheduler ideas don't overwrite each other. To label
a non-committed iteration, pass `--revision name`.

## A/B comparing two revisions

`scripts/compare_revisions.py` reads two revision dirs and prints:

1. **Aggregate diff** — macro / micro headline numbers, A → B with
   absolute and percentage delta.
2. **Per-seed paired delta** — for each shared seed, B − A on each
   metric; reports the win-count for "lower is better" metrics, plus
   the distribution (min / p25 / p50 / p75 / max) of the deltas.

```sh
python scripts/compare_revisions.py \
    out/corpus/uniform__round_robin__abc1234 \
    out/corpus/uniform__nearest__abc1234
```

A sanity check: comparing a revision to itself prints all zeros and 0/N
wins.

## Available schedulers

| Name                  | Strategy                                                   |
| --------------------- | ---------------------------------------------------------- |
| `round_robin`         | Rotating cursor + binary direction/reachability filter.    |
| `nearest`             | Pick elev minimizing projected pickup ETA.                 |

## Assumptions

### Configurable

| Parameter            | Where                                  | Default     | Notes                                                       |
| -------------------- | -------------------------------------- | ----------- | ----------------------------------------------------------- |
| `num_floors`         | config JSON                            | —           | Required. 1-indexed.                                        |
| `num_elevators`      | config JSON                            | —           | Required. All elevs share the same config.                  |
| `capacity`           | config JSON                            | —           | Required. Per-elev seat count.                              |
| `load_time`          | config JSON                            | 3           | Dwell ticks per stop (not per passenger).                   |
| `scheduler`          | `main.py --scheduler NAME`             | round_robin | One of the schedulers in the table above.                   |
| Request pattern      | `generate_requests.py --pattern NAME`  | uniform     | `uniform`, `morning_rush`, `evening_rush`.                  |

### Hard-coded

- **Causal request reveal.** At tick `t` the scheduler sees only
  requests with `request_time ≤ t`, even though the simulator has
  the full list in memory.
- **Immediate-commit binding.** Once `scheduler.assign()` returns,
  the passenger is locked to that elevator. No rebind path.
- **Capacity checked at stop boundaries, not within a stop.**
  Intra-stop transient cabin counts may exceed capacity briefly
  (door open, swap in progress). Only the post-stop count must lie
  in `[0, capacity]`.
- **One floor per tick.** Fixed travel speed, no acceleration model.

Less load-bearing but worth knowing: floors are 1-indexed (lobby =
floor 1); all elevators are identical and start at floor 1; idle
policy is "stay in place"; same-tick request order = input-file
order; `load_time` is per stop regardless of how many passengers
board/alight.

## Future work

### Reality check: real elevators don't use a single policy

Real-world Destination Dispatch dispatchers switch between **modes**
based on time of day and observed traffic, then dispatch within each
mode. None of the schedulers here do that.

- **Up-peak (morning rush)** — most cars park at the lobby, take one
  express trip up, return immediately. Only 1–2 cars handle the
  inter-floor trickle.
- **Down-peak (evening rush)** — mirror: cars spread across upper
  floors, haul passengers to lobby, return up.
- **Inter-floor / balanced (mid-day)** — nearest-car with capacity
  awareness; closest in shape to our `NearestScheduler`.
- **Express + local pairs (tall buildings)** — physical
  specialization: some elevators serve only the lobby + a "sky lobby."

### Next ideas

- **`ExpressScheduler`** — per-elev `served_floors` field +
  `BuildingConfig` schema change.
- **`RushModeScheduler`** — detect a lobby surge (e.g., over the
  last N ticks, X% of new requests originate at floor 1) and switch
  from delegating-to-Nearest to up-peak parking. Detection threshold
  and recovery hysteresis are the design questions.
- Better idle policy.

### Sweep-based routing redesign

The router currently places a new passenger's pickup and dropoff
one list position at a time. Each placement checks only its local
neighborhood — is this floor on the elevator's path between the
events immediately before and after? That's a local approximation
of the actual property we want, which is global: every passenger's
ride from pickup to dropoff should move monotonically in their
direction, with no backward detour. The local rule can accept
pickup-dropoff pairs that span an existing direction reversal;
a separate per-passenger verification catches those after the fact,
but the structure feels patched.

A cleaner formulation is to think in **sweeps** — the elevator's
planned path naturally partitions into monotonic runs of UP and
DOWN segments separated by the floors where it reverses direction.
The invariant we want is then easy to state: a new pickup–dropoff
pair must land entirely within a single matching-direction sweep,
or start a fresh sweep at the end of the queue. Under that framing
the desired property holds by construction; nothing needs to be
filtered out afterward.

The remaining open question is whether the bounds of an existing
sweep should be **extendable** to accept pickups or dropoffs just
outside the current range. There's a pinned regression test
showing a 3-passenger scenario where one such "V-shape" extension
would save roughly twenty ticks of total trip time. Tempting — but
corpus stress tests showed that under the current nearest-pickup-ETA
scheduler, allowing extensions hurts average wait times sharply.
The added flexibility lets the scheduler concentrate new requests
on whichever elevator already has a matching-direction sweep, while
quietly invalidating the pickup ETAs it had already promised to
passengers earlier in that elevator's queue. Wait times explode
and the scheduler's choices stop being trustworthy.

So extensions stay off the table until they're paired with a
scheduler that explicitly models that perturbation cost — one that
scores not just the new request's pickup time but also the harm
inflicted on already-bound passengers. Without that, the safest
policy is "no extensions": new requests either fit a sweep that
already exists, or wait their turn in a fresh sweep at the end.
The sweep redesign under that policy is a clarity refactor more
than a behavior change; its real payoff arrives the day a
load-aware scheduler does.

## Project layout

```
elevator_sim/
├── requirements.txt           runtime deps (currently none)
├── requirements-dev.txt       test deps (pytest)
├── src/                       library + CLI entry points
├── scripts/                   stress test + comparison drivers
├── tests/                     pytest suite
└── examples/                  config + the 3-passenger sample
    └── generated/             gitignored synthetic corpora
```
