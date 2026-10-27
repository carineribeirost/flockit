"""System assembly: polymer + model surface (optional) + SPC/E water + ions.

No surface: cubic periodic box with the polymer at the center.
With a surface: a frozen two-layer slab in the middle of the box (z = 0),
periodic in x and y; water sits above the top face and, through periodicity
in z, below the bottom face. The polymer starts above the top face.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flockit.forcefield import ForceField, Term
from flockit.geometry import min_distances, min_image, random_rotation
from flockit.polymer import Chain, build_chain, make_sequence

WATER_NUMBER_DENSITY = 0.03344          # molecules/A^3 (~0.997 g/cm^3)
WATER_MOLARITY = 55.5                   # mol/L of pure water: N_salt = c * N_water / 55.5
EXCL_HEAVY, EXCL_H, EXCL_SURFACE = 2.6, 2.0, 2.8           # A, exclusion radii when adding water
ION_MIN_POLYMER, ION_MIN_ION, ION_MIN_SURFACE = 5.0, 5.0, 4.0  # A, initial ion separations
SPCE_H = np.array([[0.8164904, 0.0, 0.5773590], [-0.8164904, 0.0, 0.5773590]])  # relative to O
SALTS = {"NaCl": ("Na", "Cl")}


@dataclass
class System:
    box: np.ndarray                                  # [[lo, hi]] x 3
    names: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    charges: list[float] = field(default_factory=list)
    pos: list[np.ndarray] = field(default_factory=list)
    mol: list[int] = field(default_factory=list)
    role: list[str] = field(default_factory=list)    # polymer | surface | water | ion
    bonds: list[tuple[int, int]] = field(default_factory=list)
    water_angles: list[tuple[int, int, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, name: str, typ: str, q: float, xyz, mol: int, role: str) -> int:
        self.names.append(name)
        self.types.append(typ)
        self.charges.append(float(q))
        self.pos.append(np.asarray(xyz, float))
        self.mol.append(mol)
        self.role.append(role)
        return len(self.names) - 1

    @property
    def lengths(self) -> np.ndarray:
        return self.box[:, 1] - self.box[:, 0]

    def positions(self, role: str | None = None) -> np.ndarray:
        idx = [i for i, r in enumerate(self.role) if role is None or r == role]
        return np.array([self.pos[i] for i in idx]).reshape(-1, 3)


@dataclass
class Topology:
    """Types and lists ready for the LAMMPS data file (0-based indices)."""
    atom_types: list[str]
    bond_types: list[Term]
    angle_types: list[Term]
    dihedral_types: list[Term]
    improper_types: list[Term]
    atom_type_of: list[int]
    bonds: list[tuple[int, int, int]]                 # (type, i, j)
    angles: list[tuple[int, int, int, int]]
    dihedrals: list[tuple[int, int, int, int, int]]
    impropers: list[tuple[int, int, int, int, int]]


# ---------------------------------------------------------------------------
def _principal_frame(xyz: np.ndarray) -> np.ndarray:
    """Rotate the polymer so that its direction of smallest extent lies along z."""
    c = xyz - xyz.mean(axis=0)
    _, vecs = np.linalg.eigh(c.T @ c)              # eigenvalues in ascending order
    rot = vecs[:, ::-1]                            # largest -> x, smallest -> z
    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1
    return c @ rot


def _surface(sys: System, s: dict, ff: ForceField, mol: int) -> dict:
    a, gap = float(s["lattice"]), float(s["layer_gap"])
    nx, ny = (max(2, round(float(L) / a)) for L in s["size"])
    area_nm2 = nx * a * ny * a / 100.0
    q_face = round(float(s["charge_density"]) * area_nm2)
    q_site = q_face / (nx * ny)
    sw = ff.atom_type("SW")
    for layer, (z, shift, q) in enumerate(((-gap / 2, 0.0, q_site), (gap / 2, 0.5, q_site))):
        for i in range(nx):
            for j in range(ny):
                sys.add(f"S{layer}", sw.name, q, [(i + shift) * a, (j + shift) * a, z], mol, "surface")
    return {"z_top": gap / 2, "z_bottom": -gap / 2, "area_nm2": area_nm2, "sites_per_face": nx * ny,
            "charge_per_face": q_face, "charge_density_e_nm2": q_face / area_nm2,
            "requested_charge_density_e_nm2": float(s["charge_density"])}


def _fill_water(sys: System, ff: ForceField, rng: np.random.Generator, first_mol: int) -> int:
    lo, L = sys.box[:, 0], sys.lengths
    # a grid slightly denser than the target; points are then drawn down to the water density
    n = np.maximum(1, np.ceil(L * WATER_NUMBER_DENSITY ** (1 / 3)).astype(int))
    grid = lo + (np.indices(n).reshape(3, -1).T + 0.5) * (L / n)
    heavy = [p for p, r, nm in zip(sys.pos, sys.role, sys.names) if r == "polymer" and not nm.startswith("H")]
    hydro = [p for p, r, nm in zip(sys.pos, sys.role, sys.names) if r == "polymer" and nm.startswith("H")]
    keep = min_distances(grid, np.array(heavy).reshape(-1, 3), L) >= EXCL_HEAVY
    keep &= min_distances(grid, np.array(hydro).reshape(-1, 3), L) >= EXCL_H
    keep &= min_distances(grid, sys.positions("surface"), L) >= EXCL_SURFACE
    target = int(round(WATER_NUMBER_DENSITY * np.prod(L) * keep.mean()))  # target density in the free volume
    chosen = np.sort(rng.choice(np.flatnonzero(keep), size=min(target, int(keep.sum())), replace=False))
    ow, hw = ff.atom_type("OW"), ff.atom_type("HW")
    mol = first_mol
    for o in grid[chosen]:
        h = o + SPCE_H @ random_rotation(rng).T
        io = sys.add("OW", ow.name, ow.charge, o, mol, "water")
        i1 = sys.add("HW1", hw.name, hw.charge, h[0], mol, "water")
        i2 = sys.add("HW2", hw.name, hw.charge, h[1], mol, "water")
        sys.bonds += [(io, i1), (io, i2)]
        sys.water_angles.append((i1, io, i2))
        mol += 1
    return len(chosen)


def _swap_waters_for_ions(sys: System, ions: list[str], ff: ForceField, rng: np.random.Generator) -> None:
    """Replace water molecules by ions (at the O position), away from the polymer and each other."""
    L = sys.lengths
    oxygens = np.array([i for i, nm in enumerate(sys.names) if nm == "OW"])
    O = np.array([sys.pos[i] for i in oxygens])
    d_poly = min_distances(O, sys.positions("polymer"), L)
    d_surf = min_distances(O, sys.positions("surface"), L)
    for scale in (1.0, 0.7, 0.5):
        chosen: list[int] = []
        ok = (d_poly >= ION_MIN_POLYMER * scale) & (d_surf >= ION_MIN_SURFACE * scale)
        for k in rng.permutation(np.nonzero(ok)[0]):
            if chosen:
                d = min_image(O[chosen] - O[k], L)
                if np.sqrt((d * d).sum(1)).min() < ION_MIN_ION * scale:
                    continue
            chosen.append(k)
            if len(chosen) == len(ions):
                break
        if len(chosen) == len(ions):
            break
    else:
        raise RuntimeError(f"no room for {len(ions)} ions; enlarge the box or reduce the salt")
    remove = set()
    next_mol = max(sys.mol) + 1
    for name, k in zip(ions, chosen):
        o = oxygens[k]
        remove.update((o, o + 1, o + 2))
        t = ff.atom_type(name)
        sys.add(name.upper(), t.name, t.charge, sys.pos[o], next_mol, "ion")
        next_mol += 1
    _delete_atoms(sys, remove)


def _delete_atoms(sys: System, remove: set[int]) -> None:
    keep = [i for i in range(len(sys.names)) if i not in remove]
    new = {old: i for i, old in enumerate(keep)}
    for attr in ("names", "types", "charges", "pos", "mol", "role"):
        setattr(sys, attr, [getattr(sys, attr)[i] for i in keep])
    sys.bonds = [(new[a], new[b]) for a, b in sys.bonds if a in new and b in new]
    sys.water_angles = [tuple(new[x] for x in t) for t in sys.water_angles if all(x in new for x in t)]
    # renumber molecules in increasing order, without gaps
    remap: dict[int, int] = {}
    sys.mol = [remap.setdefault(m, len(remap) + 1) for m in sys.mol]


# ---------------------------------------------------------------------------
def build_system(cfg: dict, ff: ForceField | None = None) -> tuple[System, Chain]:
    ff = ff or ForceField()
    rng = np.random.default_rng(cfg["seed"])
    pcfg, scfg, surf = cfg["polymer"], cfg["solvent"], cfg["surface"]
    if scfg.get("water", "spce") != "spce":
        raise ValueError("only the SPC/E water model is available in this version")
    seq = make_sequence(pcfg, rng)
    chain = build_chain(seq, rng, pcfg["tacticity"], float(pcfg["trans_fraction"]))
    xyz = chain.positions
    cutoff = float(cfg["protocol"]["cutoff"])

    if surf is None:
        ext = np.ptp(xyz, axis=0)
        L = max(float(ext.max()) + 2 * float(scfg["padding"]), 2 * cutoff + 4.0)
        sys = System(box=np.array([[0.0, L]] * 3))
        xyz = xyz - (xyz.min(axis=0) + xyz.max(axis=0)) / 2 + L / 2
    else:
        xyz = _principal_frame(xyz)
        a = float(surf["lattice"])
        Lx, Ly = (max(2, round(float(v) / a)) * a for v in surf["size"])
        Lz = float(surf["layer_gap"]) + float(surf["water_thickness"])
        sys = System(box=np.array([[0.0, Lx], [0.0, Ly], [-Lz / 2, Lz / 2]]))
        ext = np.ptp(xyz, axis=0)
        gap = float(surf["polymer_gap"])
        if ext[0] > Lx - 4.0 or ext[1] > Ly - 4.0:
            raise ValueError(f"polymer extent {ext[:2].round(1)} A does not fit on the "
                             f"{Lx:.1f} x {Ly:.1f} A surface; increase surface.size")
        if ext[2] + 2 * gap > float(surf["water_thickness"]):
            raise ValueError(f"increase surface.water_thickness to at least {ext[2] + 2 * gap:.0f} A")
        xyz = xyz - [(xyz[:, 0].min() + xyz[:, 0].max()) / 2 - Lx / 2,
                     (xyz[:, 1].min() + xyz[:, 1].max()) / 2 - Ly / 2,
                     xyz[:, 2].min() - (float(surf["layer_gap"]) / 2 + gap)]

    for name, typ, p in zip(chain.names, chain.types, xyz):
        sys.add(name, typ, ff.atom_type(typ).charge, p, 1, "polymer")
    sys.bonds += list(chain.bonds)
    q_polymer = round(sum(sys.charges))

    surface_meta = None
    if surf is not None:
        surface_meta = _surface(sys, surf, ff, mol=2)

    n_water = _fill_water(sys, ff, rng, first_mol=max(sys.mol) + 1)

    salt = scfg.get("salt") or {}
    unknown = set(salt) - set(SALTS)
    if unknown:
        raise ValueError(f"unsupported salts: {sorted(unknown)} (available: {sorted(SALTS)})")
    conc = float(salt.get("NaCl", 0.0))
    n_pairs = round(conc * n_water / WATER_MOLARITY)
    q_fixed = q_polymer + (2 * surface_meta["charge_per_face"] if surface_meta else 0)
    cation, anion = SALTS["NaCl"]
    ions = [cation] * (n_pairs + max(0, -q_fixed)) + [anion] * (n_pairs + max(0, q_fixed))
    if ions:
        _swap_waters_for_ions(sys, ions, ff, rng)

    total_q = sum(sys.charges)
    if abs(total_q) > 1e-6:
        raise RuntimeError(f"system has a net charge of {total_q:.6f}")
    n_water_final = sys.names.count("OW")
    sys.meta = {
        "sequence": seq,
        "polymer_charge": q_polymer,
        "n_water": n_water_final,
        "salt": {"NaCl_requested_M": conc, "pairs": n_pairs,
                 "NaCl_effective_M": round(n_pairs * WATER_MOLARITY / max(n_water_final, 1), 4)},
        "counterions": {cation: max(0, -q_fixed), anion: max(0, q_fixed)},
        "surface": surface_meta,
    }
    return sys, chain


def parametrize(sys: System, ff: ForceField) -> Topology:
    atom_types = list(dict.fromkeys(sys.types))
    tindex = {t: i for i, t in enumerate(atom_types)}
    registries: dict[str, dict] = {"bond": {}, "angle": {}, "dihedral": {}, "improper": {}}

    def type_id(kind: str, term: Term) -> int:
        reg = registries[kind]
        return reg.setdefault((term.label, term.coeffs, term.comment), len(reg))

    w = ff.water
    water_bond = Term((w["bond"]["k"], w["bond"]["r0"]), "OW_HW", "SPC/E (rigid via SHAKE)")
    water_angle = Term((w["angle"]["k"], w["angle"]["theta0"]), "HW_OW_HW", "SPC/E (rigid via SHAKE)")
    T = sys.types
    n = len(T)
    adj: list[list[int]] = [[] for _ in range(n)]
    bonds = []
    for a, b in sys.bonds:
        if sys.role[a] == "water":
            bonds.append((type_id("bond", water_bond), a, b))
            continue
        adj[a].append(b)
        adj[b].append(a)
        bonds.append((type_id("bond", ff.bond(T[a], T[b])), a, b))
    angles = [(type_id("angle", water_angle), *t) for t in sys.water_angles]
    dihedrals, impropers = [], []
    for j in range(n):
        nb = adj[j]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                angles.append((type_id("angle", ff.angle(T[nb[x]], T[j], T[nb[y]])), nb[x], j, nb[y]))
        if len(nb) == 3:
            hit = ff.improper(T[j], *(T[k] for k in nb))
            if hit:
                term, order = hit
                atoms = [j, *nb]
                impropers.append((type_id("improper", term), *(atoms[o] for o in order)))
    for j in range(n):
        for k in adj[j]:
            if k < j:
                continue
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l in (i, j):
                        continue
                    dihedrals.append((type_id("dihedral", ff.dihedral(T[i], T[j], T[k], T[l])), i, j, k, l))

    def as_terms(kind: str) -> list[Term]:
        return [Term(coeffs, label, comment) for (label, coeffs, comment) in registries[kind]]

    return Topology(atom_types, as_terms("bond"), as_terms("angle"), as_terms("dihedral"),
                    as_terms("improper"), [tindex[t] for t in T], bonds, angles, dihedrals, impropers)
