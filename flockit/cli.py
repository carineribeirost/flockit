"""Command line: flockit build | protocol | run | check | analyze."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flockit import __version__, config


def _cmd_build(args) -> None:
    from flockit.builder import build_system, parametrize
    from flockit.forcefield import ForceField
    from flockit.lammps_io import write_system
    from flockit.protocol import write_input

    cfg = config.load(args.config)
    outdir = Path(args.output or cfg.get("output") or Path(args.config).stem)
    ff = ForceField()
    system, chain = build_system(cfg, ff)
    topo = parametrize(system, ff)
    meta = write_system(outdir, system, chain, topo, ff, cfg)
    config.save(cfg, outdir / "flockit.yaml")
    write_input(outdir, meta, cfg["protocol"])
    seq = "".join("+" if s == "AA" else "." for s in chain.sequence) if set(chain.sequence) <= {"AM", "AA"} else ""
    print(f"system in {outdir}/")
    print(f"  polymer: DP {len(chain.sequence)}, {len(chain.names)} atoms, charge {meta['polymer']['charge']:+d}"
          + (f"  [{seq}]  (+ = AA)" if seq else ""))
    print(f"  water: {meta['water']['n']} molecules; NaCl: {meta['salt']['pairs']} pairs "
          f"(~{meta['salt']['NaCl_effective_M']} M); counterions: {meta['counterions']}")
    if meta["surface"]:
        s = meta["surface"]
        print(f"  surface: {s['area_nm2']:.1f} nm2/face, charge {s['charge_per_face']:+d} e/face "
              f"({s['charge_density_e_nm2']:+.3f} e/nm2)")
    box = meta["box"]
    print(f"  box: {' x '.join(f'{hi - lo:.1f}' for lo, hi in box)} A; {meta['n_atoms']} atoms")
    print(f"  types: {len(topo.atom_types)} atom, {len(topo.bond_types)} bond, {len(topo.angle_types)} angle, "
          f"{len(topo.dihedral_types)} dihedral, {len(topo.improper_types)} improper")
    print(f"  LAMMPS input: {outdir}/in.flockit  (or: flockit run {outdir})")


def _load_dir(d: str):
    from flockit.lammps_io import load_metadata
    outdir = Path(d)
    return outdir, load_metadata(outdir), config.load(outdir / "flockit.yaml")


def _cmd_protocol(args) -> None:
    from flockit.protocol import write_input
    outdir, meta, cfg = _load_dir(args.dir)
    if args.config:
        cfg = config.deep_merge(cfg, {"protocol": config.load(args.config)["protocol"]})
        config.save(cfg, outdir / "flockit.yaml")
    print(write_input(outdir, meta, cfg["protocol"]))


def _cmd_run(args) -> None:
    from flockit.protocol import run
    outdir, meta, cfg = _load_dir(args.dir)
    ok = run(outdir, meta, cfg["protocol"], omp=args.omp)
    sys.exit(0 if ok else 2)


def _cmd_check(args) -> None:
    from flockit.convergence import check
    conv = config.DEFAULTS["protocol"]["convergence"]
    obs = args.observables.split(",") if args.observables else conv["observables"]
    report = check(args.obs, obs, conv["rel_tol"], args.window)
    print(report.table())
    sys.exit(0 if report.converged else 2)


def _cmd_analyze(args) -> None:
    from flockit.analysis import analyze
    outdir, meta, _ = _load_dir(args.dir)
    analyze(outdir, meta, traj=args.traj, d_ads=args.d_ads, r_cond=args.r_cond,
            skip=args.skip, plot=not args.no_plot)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="flockit", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build polymer + water + salt (+ surface) and write the LAMMPS files")
    p.add_argument("config", help="YAML file")
    p.add_argument("-o", "--output", help="output directory (default: the YAML name)")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("protocol", help="(re)write in.flockit for a built system")
    p.add_argument("dir")
    p.add_argument("--config", help="YAML with a protocol: section to override")
    p.set_defaults(func=_cmd_protocol)

    p = sub.add_parser("run", help="run the protocol through the LAMMPS Python module, extending NPT until converged")
    p.add_argument("dir")
    p.add_argument("--omp", type=int, default=1, help="OpenMP threads (LAMMPS OPENMP package)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="test the convergence of an .obs file (fix ave/time)")
    p.add_argument("obs")
    p.add_argument("--observables", help="comma-separated list (default: density,pe)")
    p.add_argument("--window", type=float, default=0.5, help="final fraction of the series to use")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("analyze", help="conformation, adsorption and counterion condensation from the trajectory")
    p.add_argument("dir")
    p.add_argument("--traj", help="trajectory (default: 04_prod.lammpstrj)")
    p.add_argument("--d-ads", type=float, default=5.0, help="distance (A) to a face that counts as adsorbed")
    p.add_argument("--r-cond", type=float, default=3.2, help="cation-O radius (A) for a condensed counterion")
    p.add_argument("--skip", type=int, default=0, help="number of initial frames to discard")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=_cmd_analyze)

    args = ap.parse_args(argv)
    args.func(args)
