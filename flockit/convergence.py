"""Convergence check from `fix ave/time` output files (*.obs).

Drift test: the final window of the series is split in half and the means of
the two halves are compared. An observable is considered stable if the
difference is below 2 standard errors (block averages) or below the relative
tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = {"temp": "c_tmob", "press": "c_pmob", "density": "v_dens", "pe": "c_thermo_pe",
           "vol": "v_vol", "rg": "v_rg"}
MIN_POINTS = 10


@dataclass
class ObsCheck:
    name: str
    first: float
    second: float
    stderr: float
    rel_change: float
    tol: float
    converged: bool
    required: bool


@dataclass
class Report:
    n_points: int
    checks: list[ObsCheck]

    @property
    def converged(self) -> bool:
        return self.n_points >= MIN_POINTS and all(c.converged for c in self.checks if c.required)

    def table(self) -> str:
        lines = [f"{'quantity':<9} {'1st half':>12} {'2nd half':>12} {'std err':>10} "
                 f"{'change':>9} {'tol':>6}  status"]
        for c in self.checks:
            status = ("ok" if c.converged else "DRIFT") + ("" if c.required else " (info)")
            lines.append(f"{c.name:<9} {c.first:12.5g} {c.second:12.5g} {c.stderr:10.3g} "
                         f"{100 * c.rel_change:8.2f}% {100 * c.tol:5.1f}%  {status}")
        verdict = "CONVERGED" if self.converged else "NOT CONVERGED"
        if self.n_points < MIN_POINTS:
            verdict += f" (only {self.n_points} points in the window; minimum {MIN_POINTS})"
        return "\n".join(lines + [verdict])


def read_obs(path: str | Path) -> dict[str, np.ndarray]:
    header = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                header = line[1:].split()
            else:
                break
    if header is None:
        raise ValueError(f"{path}: fix ave/time header not found")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        data = np.empty((0, len(header)))
    return {name: data[:, i] for i, name in enumerate(header)}


def block_stderr(x: np.ndarray, nblocks: int = 4) -> float:
    if len(x) < nblocks:
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("inf")
    means = [b.mean() for b in np.array_split(x, nblocks)]
    return float(np.std(means, ddof=1) / np.sqrt(nblocks))


def check(path: str | Path, observables: list[str], rel_tol: dict[str, float],
          window: float = 0.5, info: tuple[str, ...] = ("rg",)) -> Report:
    data = read_obs(path)
    n_total = len(next(iter(data.values())))
    start = n_total - max(int(round(n_total * window)), 0)
    checks = []
    for name in list(observables) + [i for i in info if i not in observables]:
        col = COLUMNS.get(name, name)
        if col not in data:
            continue
        x = data[col][start:]
        if len(x) < 2:
            checks.append(ObsCheck(name, np.nan, np.nan, np.nan, np.nan, rel_tol.get(name, 0.01),
                                   False, name in observables))
            continue
        a, b = np.array_split(x, 2)
        diff = b.mean() - a.mean()
        se = float(np.hypot(block_stderr(a), block_stderr(b)))
        rel = abs(diff) / max(abs(b.mean()), 1e-12)
        tol = rel_tol.get(name, 0.01)
        checks.append(ObsCheck(name, float(a.mean()), float(b.mean()), se, rel, tol,
                               bool(abs(diff) <= 2 * se or rel <= tol), name in observables))
    return Report(n_total - start, checks)
