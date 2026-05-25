"""Building / system configuration for the elevator simulator.

Holds the static parameters of a single simulation run. Floors are 1-indexed
(matching the input request format in the spec). All elevators share the
same capacity and load_time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildingConfig:
    """Immutable configuration for one simulation run."""

    num_floors: int
    num_elevators: int
    capacity: int
    # stop cost: door-open, load/unload and door-close.
    load_time: int = 3
