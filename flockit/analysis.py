"""Trajectory analysis: conformation, adsorption and counterion condensation.

Conformation: radius of gyration (total, and components parallel and
perpendicular to the surface), relative shape anisotropy kappa^2 (0 = sphere,
1 = rod) and end-to-end distance.

Adsorption (surface systems only): a monomer is adsorbed if any of its heavy
atoms is within `d_ads` A of a face. The monomer sequence is split into trains
(adsorbed), loops (free, between trains) and tails (free, at the chain ends),
the classical Scheutjens-Fleer description.

Condensation: a charged group has a condensed counterion if a cation is within
`r_cond` A of one of its oxygens (minimum image).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from flockit.convergence import block_stderr
from flockit.geometry import min_image

ANALYSIS_DIR = "analysis"


def iter_dump(path: str | Path) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Frames of a custom dump with id and xu yu zu: (step, 3x2 box, ids, coordinates)."""
    with open(path, encoding="ascii") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            n = int(fh.readline())
            fh.readline()
            box = np.array([fh.readline().split()[:2] for _ in range(3)], float)
            cols = fh.readline().split()[2:]
            data = np.array(" ".join(fh.readline() for _ in range(n)).split(), float).reshape(n, len(cols))
            ids = data[:, cols.index("id")].astype(int)
            xyz = data[:, [cols.index(c) for c in ("xu", "yu", "zu")]]
            order = np.argsort(ids)
            yield step, box, ids[order], xyz[order]


def segments(adsorbed: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Lengths of trains, loops and tails along the chain."""
    if not adsorbed.any():
        return [], [], []
    change = np.flatnonzero(np.diff(adsorbed.astype(int))) + 1
    bounds = np.concatenate(([0], change, [len(adsorbed)]))
    runs = [(bool(adsorbed[a]), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]
    trains = [n for ads, n in runs if ads]
    tails = [n for k, (ads, n) in enumerate(runs) if not ads and k in (0, len(runs) - 1)]
    loops = [n for k, (ads, n) in enumerate(runs) if not ads and 0 < k < len(runs) - 1]
    return trains, loops, tails


def surface_distance(z: np.ndarray, zlo: float, lz: float, z_top: float, z_bottom: float) -> np.ndarray:
    """Distance to the nearest slab face (above the top face or below the bottom face)."""
    zw = zlo + np.mod(z - zlo, lz)
    return np.minimum(np.mod(zw - z_top, lz), np.mod(z_bottom - zw, lz))


class Analyzer:
    def __init__(self, meta: dict, d_ads: float = 5.0, r_cond: float = 3.2,
                 profile_bin: float = 0.25) -> None:
        poly = meta["polymer"]
        self.poly_ids = np.array(poly["atom_ids"])
        self.mass = np.array(poly["masses"])
        self.monomer = np.array(poly["monomer"])
        self.heavy = np.array(poly["heavy"])
        self.sequence = poly["sequence"]
        names = dict(zip(poly["atom_ids"], poly["names"]))
        self.ends = (poly["backbone_ids"][0], poly["backbone_ids"][-1])
        self.group_oxygens = [[a for a in g if names[a].startswith("O")] for g in poly["charged_groups"]]
        self.cations = np.array(meta["ions"].get("Na", []), int)
        self.anions = np.array(meta["ions"].get("Cl", []), int)
        self.surface = meta["surface"]
        self.d_ads, self.r_cond, self.bin = d_ads, r_cond, profile_bin
        self.rows: list[dict] = []
        self.ads_count = np.zeros(len(self.sequence))
        self.mon_dist_sum = np.zeros(len(self.sequence))
        self.profiles: dict[str, np.ndarray] = {}
        self.n_frames = 0

    def _rows_of(self, ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(ids, wanted)
        if len(wanted) and (rows.max() >= len(ids) or not np.array_equal(ids[rows], wanted)):
            raise ValueError("the trajectory does not contain all required atoms (polymer and ions)")
        return rows

    def frame(self, step: int, box: np.ndarray, ids: np.ndarray, xyz: np.ndarray) -> dict:
        L = box[:, 1] - box[:, 0]
        p = xyz[self._rows_of(ids, self.poly_ids)]
        com = (self.mass[:, None] * p).sum(0) / self.mass.sum()
        c = p - com
        S = (self.mass[:, None, None] * c[:, :, None] * c[:, None, :]).sum(0) / self.mass.sum()
        lam = np.linalg.eigvalsh(S)
        tr = lam.sum()
        e0, e1 = (xyz[self._rows_of(ids, np.array([e]))][0] for e in self.ends)
        row = {"step": step, "rg": float(np.sqrt(tr)),
               "kappa2": float(1 - 3 * (lam[0] * lam[1] + lam[1] * lam[2] + lam[2] * lam[0]) / tr ** 2),
               "ree": float(np.linalg.norm(e1 - e0))}

        if self.group_oxygens and len(self.cations):
            cat = xyz[self._rows_of(ids, self.cations)]
            n_near = []
            for oxy in self.group_oxygens:
                o = xyz[self._rows_of(ids, np.array(oxy))]
                d = np.linalg.norm(min_image(cat[:, None, :] - o[None, :, :], L), axis=-1).min(axis=1)
                n_near.append(int((d <= self.r_cond).sum()))
            n_near = np.array(n_near)
            row["f_condensed"] = float((n_near > 0).mean())
            row["cations_per_group"] = float(n_near.mean())

        if self.surface:
            s = self.surface
            dist = lambda z: surface_distance(z, box[2, 0], L[2], s["z_top"], s["z_bottom"])  # noqa: E731
            d_atoms = dist(p[:, 2])
            d_mon = np.full(len(self.sequence), np.inf)
            np.minimum.at(d_mon, self.monomer[self.heavy], d_atoms[self.heavy])
            ads = d_mon <= self.d_ads
            trains, loops, tails = segments(ads)
            row.update({"rg_par": float(np.sqrt(S[0, 0] + S[1, 1])), "rg_perp": float(np.sqrt(S[2, 2])),
                        "com_dist": float(dist(np.array([com[2]]))[0]), "f_ads": float(ads.mean()),
                        "n_trains": len(trains),
                        "train_len": float(np.mean(trains)) if trains else 0.0,
                        "loop_len": float(np.mean(loops)) if loops else 0.0,
                        "tail_len": float(np.mean(tails)) if tails else 0.0})
            self.ads_count += ads
            self.mon_dist_sum += d_mon
            if "_edges" not in self.profiles:   # fixed bins (Lz may change under NPT)
                half = (L[2] - (s["z_top"] - s["z_bottom"])) / 2
                self.profiles["_edges"] = np.arange(0.0, half + self.bin, self.bin)
            edges = self.profiles["_edges"]
            area = L[0] * L[1]
            for name, z in (("polymer_heavy", p[self.heavy, 2]),
                            ("Na", xyz[self._rows_of(ids, self.cations), 2] if len(self.cations) else []),
                            ("Cl", xyz[self._rows_of(ids, self.anions), 2] if len(self.anions) else [])):
                h, _ = np.histogram(dist(np.asarray(z)), bins=edges)
                # each dz bin corresponds to two slices (one on each side of the slab)
                dens = h / (2 * area * self.bin) * 1000.0      # 1/nm^3
                self.profiles[name] = self.profiles.get(name, 0) + dens
        self.rows.append(row)
        self.n_frames += 1
        return row


def analyze(outdir: Path, meta: dict, traj: str | None = None, d_ads: float = 5.0,
            r_cond: float = 3.2, skip: int = 0, plot: bool = True) -> dict:
    outdir = Path(outdir)
    traj_path = outdir / (traj or "04_prod.lammpstrj")
    dt = 2.0
    proto = outdir / "protocol.json"
    if proto.exists():
        dt = json.loads(proto.read_text(encoding="utf-8"))["production_dt_fs"]
    an = Analyzer(meta, d_ads, r_cond)
    for k, frame in enumerate(iter_dump(traj_path)):
        if k >= skip:
            an.frame(*frame)
    if not an.rows:
        raise RuntimeError(f"no frames analyzed in {traj_path}")

    res = outdir / ANALYSIS_DIR
    res.mkdir(exist_ok=True)
    keys = list(an.rows[0])
    for r in an.rows:
        r["time_ps"] = r["step"] * dt / 1000.0
    with open(res / "timeseries.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "time_ps"] + keys[1:])
        w.writeheader()
        w.writerows(an.rows)

    summary = {"frames": an.n_frames, "d_ads_A": d_ads, "r_cond_A": r_cond, "observables": {}}
    for key in keys[1:]:
        x = np.array([r[key] for r in an.rows], float)
        summary["observables"][key] = {"mean": float(x.mean()), "stderr": block_stderr(x, 5)
                                       if len(x) >= 5 else float("nan"), "first": float(x[0]), "last": float(x[-1])}
    if meta["surface"]:
        p_ads = an.ads_count / an.n_frames
        summary["adsorbed_fraction_by_monomer"] = {
            name: float(p_ads[[i for i, s in enumerate(an.sequence) if s == name]].mean())
            for name in sorted(set(an.sequence))}
        with open(res / "monomers.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "monomer", "p_adsorbed", "mean_distance_A"])
            for i, name in enumerate(an.sequence):
                w.writerow([i, name, f"{p_ads[i]:.4f}", f"{an.mon_dist_sum[i] / an.n_frames:.3f}"])
        edges = an.profiles.pop("_edges")
        centers = (edges[:-1] + edges[1:]) / 2
        with open(res / "profiles.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["distance_A"] + [f"{k}_per_nm3" for k in an.profiles])
            for i, d in enumerate(centers):
                w.writerow([f"{d:.3f}"] + [f"{an.profiles[k][i] / an.n_frames:.5f}" for k in an.profiles])
    (res / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"{an.n_frames} frames analyzed; results in {res}/")
    for key, v in summary["observables"].items():
        print(f"  {key:<18} {v['mean']:10.4g} +- {v['stderr']:.2g}")
    for name, f in summary.get("adsorbed_fraction_by_monomer", {}).items():
        print(f"  P(adsorbed | {name}) = {f:.3f}")
    if plot:
        _plot(res, an, meta, dt)
    return summary


def _plot(res: Path, an: Analyzer, meta: dict, dt: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: plots skipped)")
        return
    t = np.array([r["time_ps"] for r in an.rows])
    col = lambda k: np.array([r[k] for r in an.rows])  # noqa: E731
    n = 3 if meta["surface"] else 2
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(t, col("rg"), label="$R_g$")
    if meta["surface"]:
        ax.plot(t, col("rg_par"), label=r"$R_{g,\parallel}$")
        ax.plot(t, col("rg_perp"), label=r"$R_{g,\perp}$")
    ax.set(xlabel="t (ps)", ylabel=r"$\AA$", title="Conformation")
    ax.legend(frameon=False)
    ax = axes[1]
    if "f_condensed" in an.rows[0]:
        ax.plot(t, col("f_condensed"), label="condensed counterion")
    if meta["surface"]:
        ax.plot(t, col("f_ads"), label="adsorbed monomers")
    ax.set(xlabel="t (ps)", ylabel="fraction", ylim=(-0.02, 1.02), title="Fractions")
    ax.legend(frameon=False)
    if meta["surface"]:
        ax = axes[2]
        with open(res / "profiles.csv", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        head, data = rows[0], np.array(rows[1:], float)
        for k in range(1, len(head)):
            ax.plot(data[:, 0], data[:, k], label=head[k].replace("_per_nm3", ""))
        ax.set(xlabel=r"distance to face ($\AA$)", ylabel="nm$^{-3}$", title="Profiles")
        ax.legend(frameon=False)
    fig.savefig(res / "summary.png", dpi=130)
    plt.close(fig)
