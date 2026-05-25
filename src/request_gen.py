"""Random request generators for the elevator simulator.

Produces synthetic Passenger lists for stress-testing schedulers and
for cross-scheduler comparisons.
"""
from __future__ import annotations

import random
from typing import Optional

from passenger import Passenger


def generate_uniform(
    num_passengers: int,
    num_floors: int,
    duration: int,
    *,
    seed: Optional[int] = None,
) -> list[Passenger]:
    """Generate `num_passengers` requests with uniform random pattern.

    Args:
        num_passengers: Number of Passenger records to produce.
        num_floors: Floor range `[1, num_floors]`. Must be >= 2 so
                    origin != destination is always possible.
        duration: Request times sampled uniformly from `[0, duration]`
                  inclusive.
        seed: If given, makes output deterministic (used by tests and
              by anyone who wants to reproduce a specific input).

    Returns:
        Passengers sorted by `(request_time, origin, destination)`,
        with IDs assigned sequentially starting at 0 *after* sorting
        (so a given seed produces a byte-identical CSV regardless of
        any downstream re-sorting).
    """
    return _generate(
        num_passengers, num_floors, duration, _draw_uniform, seed,
    )


def generate_morning_rush(
    num_passengers: int,
    num_floors: int,
    duration: int,
    *,
    lobby_fraction: float = 0.8,
    seed: Optional[int] = None,
) -> list[Passenger]:
    """Generate requests for a "morning rush" pattern.

    `lobby_fraction` of pickups are at floor 1, with destinations
    uniform in `[2, num_floors]`. The remaining `1 - lobby_fraction`
    are uniform inter-floor (same distribution as
    `generate_uniform`).

    Args:
        num_passengers, num_floors, duration, seed: as in
            `generate_uniform`.
        lobby_fraction: fraction in `[0, 1]` of pickups at floor 1.
            Defaults to 0.8 (heavy lobby bias). Set to 0 to recover
            uniform random; set to 1 for an all-lobby workload.
    """
    _validate_lobby_fraction(lobby_fraction)
    draw = lambda rng, nf: _draw_lobby_origin(rng, nf, lobby_fraction)
    return _generate(num_passengers, num_floors, duration, draw, seed)


def generate_evening_rush(
    num_passengers: int,
    num_floors: int,
    duration: int,
    *,
    lobby_fraction: float = 0.8,
    seed: Optional[int] = None,
) -> list[Passenger]:
    """Generate requests for an "evening rush" pattern.

    Symmetric to `generate_morning_rush`: `lobby_fraction` of
    *destinations* are floor 1, with origins uniform in
    `[2, num_floors]`. The rest are uniform inter-floor.
    """
    _validate_lobby_fraction(lobby_fraction)
    draw = lambda rng, nf: _draw_lobby_dest(rng, nf, lobby_fraction)
    return _generate(num_passengers, num_floors, duration, draw, seed)


def _generate(
    num_passengers: int,
    num_floors: int,
    duration: int,
    draw_one,
    seed: Optional[int],
) -> list[Passenger]:
    """Common skeleton: validate inputs, run `draw_one(rng, num_floors)`
    `num_passengers` times to get `(origin, destination)` pairs, sort
    by `(request_time, origin, destination)`, assign sequential IDs."""
    if num_passengers < 0:
        raise ValueError(f"num_passengers must be >= 0, got {num_passengers}")
    if num_floors < 2:
        raise ValueError(
            f"num_floors must be >= 2 to ensure origin != destination, "
            f"got {num_floors}"
        )
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration}")

    rng = random.Random(seed)
    raw: list[tuple[int, int, int]] = []
    for _ in range(num_passengers):
        t = rng.randint(0, duration)
        origin, dest = draw_one(rng, num_floors)
        raw.append((t, origin, dest))

    raw.sort()
    return [
        Passenger(id=pid, request_time=t, origin=origin, destination=dest)
        for pid, (t, origin, dest) in enumerate(raw)
    ]


def _draw_uniform(rng: random.Random, num_floors: int) -> tuple[int, int]:
    """Origin uniform in `[1, num_floors]`; destination uniform among
    the other floors."""
    origin = rng.randint(1, num_floors)
    dest = rng.randint(1, num_floors - 1)
    if dest >= origin:
        dest += 1
    return origin, dest


def _draw_lobby_origin(
    rng: random.Random, num_floors: int, lobby_fraction: float,
) -> tuple[int, int]:
    """`lobby_fraction` chance origin=1 (dest uniform over upper floors);
    otherwise uniform inter-floor."""
    if rng.random() < lobby_fraction:
        return 1, rng.randint(2, num_floors)
    return _draw_uniform(rng, num_floors)


def _draw_lobby_dest(
    rng: random.Random, num_floors: int, lobby_fraction: float,
) -> tuple[int, int]:
    """`lobby_fraction` chance dest=1 (origin uniform over upper floors);
    otherwise uniform inter-floor."""
    if rng.random() < lobby_fraction:
        return rng.randint(2, num_floors), 1
    return _draw_uniform(rng, num_floors)


def _validate_lobby_fraction(lobby_fraction: float) -> None:
    if not 0.0 <= lobby_fraction <= 1.0:
        raise ValueError(
            f"lobby_fraction must be in [0, 1], got {lobby_fraction}"
        )
