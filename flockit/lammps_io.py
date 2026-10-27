"""Writing LAMMPS files (data file and coefficients) and system metadata."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from flockit import __version__
from flockit.builder import System, Topology
from flockit.forcefield import ForceField, Term

DATA_FILE, SETTINGS_FILE, META_FILE = "system.data", "system.settings", "system.json"


def _ascii(text: str) -> str:
    # the OPLS wildcard character is written as X in LAMMPS comments
    return text.replace(chr(0x20AC), "X").encode("ascii", "replace").decode()


def _term_line(i: int, term: Term, kind: str) -> str:
    c = term.coeffs
    values = (f"{c[0]:.4f} {int(c[1])} {int(c[2])}" if kind == "improper"
              else " ".join(f"{v:.4f}" for v in c))
    note = f" {term.comment}" if term.comment else ""
    return _ascii(f"{i + 1} {values}  # {term.label}{note}")


def write_data(path: Path, sys: System, topo: Topology, ff: ForceField) -> None:
    sections = [("bond", topo.bonds, topo.bond_types), ("angle", topo.angles, topo.angle_types),
                ("dihedral", topo.dihedrals, topo.dihedral_types),
                ("improper", topo.impropers, topo.improper_types)]
    out = [f"LAMMPS data file - flockit {__version__} - {datetime.now():%Y-%m-%d %H:%M}", ""]
    out.append(f"{len(sys.names)} atoms")
    out += [f"{len(rows)} {kind}s" for kind, rows, _ in sections]
    out.append(f"{len(topo.atom_types)} atom types")
    out += [f"{len(types)} {kind} types" for kind, _, types in sections]
    out.append("")
    for (lo, hi), ax in zip(sys.box, "xyz"):
        out.append(f"{lo:.6f} {hi:.6f} {ax}lo {ax}hi")

    out += ["", "Masses", ""]
    for i, name in enumerate(topo.atom_types):
        t = ff.atom_type(name)
        out.append(_ascii(f"{i + 1} {t.mass:.5f}  # {name} {t.description}"))
    out += ["", "Pair Coeffs # lj/cut/coul/long", ""]
    for i, name in enumerate(topo.atom_types):
        t = ff.atom_type(name)
        out.append(f"{i + 1} {t.epsilon:.7f} {t.sigma:.6f}  # {name}")
    styles = {"bond": "harmonic", "angle": "harmonic", "dihedral": "opls", "improper": "cvff"}
    for kind, _, types in sections:
        if types:
            out += ["", f"{kind.capitalize()} Coeffs # {styles[kind]}", ""]
            out += [_term_line(i, term, kind) for i, term in enumerate(types)]

    out += ["", "Atoms # full", ""]
    for i, (m, q, p) in enumerate(zip(sys.mol, sys.charges, sys.pos)):
        # 10 decimals: the fractional surface-site charges summed over hundreds of atoms
        # must stay neutral for LAMMPS (tolerance 1e-5)
        out.append(f"{i + 1} {m} {topo.atom_type_of[i] + 1} {q:.10f} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    for kind, rows, _ in sections:
        if rows:
            out += ["", f"{kind.capitalize()}s", ""]
            out += [f"{n + 1} {row[0] + 1} " + " ".join(str(a + 1) for a in row[1:])
                    for n, row in enumerate(rows)]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_settings(path: Path, sys: System, topo: Topology, ff: ForceField) -> dict:
    """Cross pair coefficients (Lorentz-Berthelot for ions) and groups. Returns the type ids used."""
    tid = {name: i + 1 for i, name in enumerate(topo.atom_types)}
    out = ["# flockit: mixing and groups (include after read_data)",
           "pair_modify mix geometric tail yes", ""]
    types = [ff.atom_type(n) for n in topo.atom_types]
    out.append("# pairs involving Joung-Cheatham ions: Lorentz-Berthelot rule")
    for i, ti in enumerate(types):
        for j in range(i, len(types)):
            tj = types[j]
            if "arithmetic" in (ti.mixing, tj.mixing):
                eps = (ti.epsilon * tj.epsilon) ** 0.5
                sig = (ti.sigma + tj.sigma) / 2
                out.append(f"pair_coeff {i + 1} {j + 1} {eps:.7f} {sig:.6f}  # {ti.name}-{tj.name}")
    roles = {r: sorted({tid[sys.types[k]] for k in range(len(sys.types)) if sys.role[k] == r})
             for r in ("water", "ion", "surface")}
    out += ["", "group polymer molecule 1"]
    if roles["water"]:
        out.append("group water type " + " ".join(map(str, roles["water"])))
    if roles["ion"]:
        out.append("group ions type " + " ".join(map(str, roles["ion"])))
    if roles["surface"]:
        out += ["group surface type " + " ".join(map(str, roles["surface"])),
                "group mobile subtract all surface"]
    else:
        out.append("group mobile union all")
    path.write_text(_ascii("\n".join(out)) + "\n", encoding="ascii")
    water_bond = next((i + 1 for i, t in enumerate(topo.bond_types) if t.label == "OW_HW"), None)
    water_angle = next((i + 1 for i, t in enumerate(topo.angle_types) if t.label == "HW_OW_HW"), None)
    return {"atom_types": tid, "water_bond_type": water_bond, "water_angle_type": water_angle}


def system_metadata(sys: System, chain, ff: ForceField, lammps_types: dict, cfg: dict) -> dict:
    poly = [i for i, r in enumerate(sys.role) if r == "polymer"]   # they come first, in chain order
    ions: dict[str, list[int]] = {}
    for i, r in enumerate(sys.role):
        if r == "ion":
            ions.setdefault(ff.atom_type(sys.types[i]).name, []).append(i + 1)
    oxygens = [i + 1 for i, nm in enumerate(sys.names) if nm == "OW"]
    surface_ids = [i + 1 for i, r in enumerate(sys.role) if r == "surface"]
    surface = dict(sys.meta["surface"]) if sys.meta["surface"] else None
    if surface:
        surface["atom_id_range"] = [surface_ids[0], surface_ids[-1]]
    return {
        "flockit_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_atoms": len(sys.names),
        "box": sys.box.tolist(),
        "lammps_types": lammps_types,
        "polymer": {
            "sequence": chain.sequence,
            "charge": sys.meta["polymer_charge"],
            "atom_ids": [i + 1 for i in poly],
            "names": [sys.names[i] for i in poly],
            "monomer": [int(m) for m in chain.monomer],
            "masses": [ff.atom_type(sys.types[i]).mass for i in poly],
            "heavy": [not sys.names[i].startswith("H") for i in poly],
            "backbone_ids": [b + 1 for b in chain.backbone],
            "charged_groups": [[a + 1 for a in g] for g in chain.charged_groups],
        },
        "ions": ions,
        "water": {"n": len(oxygens), "first_oxygen_id": oxygens[0] if oxygens else None},
        "surface": surface,
        "salt": sys.meta["salt"],
        "counterions": sys.meta["counterions"],
        "config": cfg,
    }


def write_system(outdir: Path, sys: System, chain, topo: Topology, ff: ForceField, cfg: dict) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    write_data(outdir / DATA_FILE, sys, topo, ff)
    lammps_types = write_settings(outdir / SETTINGS_FILE, sys, topo, ff)
    meta = system_metadata(sys, chain, ff, lammps_types, cfg)
    (outdir / META_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def load_metadata(outdir: Path) -> dict:
    return json.loads((Path(outdir) / META_FILE).read_text(encoding="utf-8"))


def box_lengths(meta: dict) -> np.ndarray:
    box = np.array(meta["box"])
    return box[:, 1] - box[:, 0]
