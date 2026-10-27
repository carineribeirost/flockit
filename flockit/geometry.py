"""Geometry: placement from internal coordinates (NeRF), tetrahedral directions and PBC."""
from __future__ import annotations

import numpy as np

TETRAHEDRAL = 109.47


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def place(a, b, c, bond: float, angle: float, dihedral: float) -> np.ndarray:
    """Position of d with |cd| = bond, angle b-c-d = angle and dihedral a-b-c-d = dihedral (degrees)."""
    bc = unit(c - b)
    n = unit(np.cross(b - a, bc))
    m = np.cross(n, bc)
    th, ph = np.radians(angle), np.radians(dihedral)
    return c + bond * (-np.cos(th) * bc + np.sin(th) * np.cos(ph) * m + np.sin(th) * np.sin(ph) * n)


def dihedral(a, b, c, d) -> float:
    b0, b1, b2 = a - b, c - b, d - c
    b1n = unit(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def angle(a, b, c) -> float:
    u, v = unit(a - b), unit(c - b)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def two_tetrahedral(center, n1, n2, theta: float = TETRAHEDRAL) -> tuple[np.ndarray, np.ndarray]:
    """The two free (unit) directions of an sp3 carbon that already has two neighbors."""
    u, v = unit(n1 - center), unit(n2 - center)
    bis = -unit(u + v)
    nrm = unit(np.cross(u, v))
    half = np.radians(theta / 2.0)
    return bis * np.cos(half) + nrm * np.sin(half), bis * np.cos(half) - nrm * np.sin(half)


def three_tetrahedral(center, neighbor, ref) -> list[np.ndarray]:
    """The three free directions of an sp3 carbon with one neighbor (`ref` is bonded to the neighbor)."""
    return [unit(place(ref, neighbor, center, 1.0, TETRAHEDRAL, phi) - center)
            for phi in (180.0, 60.0, -60.0)]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def min_image(d: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return d - lengths * np.round(d / lengths)


def min_distances(points: np.ndarray, others: np.ndarray, lengths: np.ndarray,
                  chunk: int = 2048) -> np.ndarray:
    """For each point, the smallest distance (minimum image) to the set `others`."""
    out = np.full(len(points), np.inf)
    if len(others) == 0:
        return out
    for s in range(0, len(points), chunk):
        d = min_image(points[s:s + chunk, None, :] - others[None, :, :], lengths)
        out[s:s + chunk] = np.sqrt((d * d).sum(-1)).min(axis=1)
    return out
