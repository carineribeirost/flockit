"""Configuration: defaults and reading the user's YAML."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "seed": 2024,
    "polymer": {
        "dp": 20,
        "composition": {"AM": 0.7, "AA": 0.3},  # fractions; the sequence is shuffled with exact counts
        "sequence": None,                        # explicit list (e.g. [AM, AA, AM]); takes precedence
        "tacticity": "atactic",                  # atactic | isotactic | syndiotactic
        "trans_fraction": 0.6,                   # fraction of trans dihedrals in the initial conformation
    },
    "solvent": {
        "water": "spce",
        "padding": 12.0,             # A between the polymer and the box edge (no surface)
        "salt": {"NaCl": 0.1},       # mol/L
    },
    "surface": None,
    "protocol": {
        "temperature": 300.0,        # K
        "pressure": 1.0,             # atm
        "cutoff": 10.0,              # A (real-space LJ and Coulomb)
        "kspace_accuracy": 1.0e-4,
        "tdamp": 100.0,              # fs
        "pdamp": 1000.0,             # fs
        "observe_every_ps": 1.0,
        "minimize": {"etol": 1.0e-4, "ftol": 1.0e-6, "maxiter": 5000, "maxeval": 50000},
        "heat": {"ps": 20.0, "dt": 1.0},
        "npt": {"ps": 500.0, "dt": 2.0, "max_extensions": 4},
        "production": {"ps": 2000.0, "dt": 2.0, "ensemble": "nvt",
                       "dump_every_ps": 1.0, "dump_water": False},
        "convergence": {
            "observables": ["density", "pe"],
            "rel_tol": {"density": 0.005, "pe": 0.005, "rg": 0.05},
            "window": 0.5,           # final fraction of the series used by the test
        },
    },
}

SURFACE_DEFAULTS: dict = {
    "type": "model",
    "charge_density": 0.0,      # e/nm^2 on each face (signed)
    "lattice": 2.8,             # A, square lattice of surface sites
    "layer_gap": 2.5,           # A between the two site layers
    "size": [40.0, 40.0],       # A (rounded to lattice multiples)
    "water_thickness": 50.0,    # A of water between the top and bottom faces (through periodicity)
    "polymer_gap": 6.0,         # A between the lowest polymer atom and the top face
}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve(user: dict | None) -> dict:
    """Merge the user's configuration with the defaults."""
    cfg = deep_merge(DEFAULTS, user)
    if cfg["surface"] is not None:
        cfg["surface"] = deep_merge(SURFACE_DEFAULTS, cfg["surface"])
    if (user or {}).get("polymer", {}).get("composition") is not None:
        cfg["polymer"]["composition"] = user["polymer"]["composition"]  # do not inherit default monomers
    return cfg


def read(path: str | Path) -> dict:
    """The YAML as written by the user, without defaults."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load(path: str | Path) -> dict:
    return resolve(read(path))


def save(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
