"""Equilibration and production protocol.

Stages: (1) minimization without SHAKE; (2) NVT heating with SHAKE on X-H
bonds and water; (3) NPT, isotropic without a surface or along z only with
the frozen slab; (4) NVT or NPT production, with a trajectory of the polymer
and the ions.

`write_input` writes a single `in.flockit` that runs on any LAMMPS (e.g. on a
cluster). `run` executes the same stages through the LAMMPS Python module and
extends the NPT stage in blocks until the convergence check passes.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from flockit import __version__
from flockit.convergence import check
from flockit.lammps_io import DATA_FILE, SETTINGS_FILE

INPUT_FILE, PROTOCOL_FILE = "in.flockit", "protocol.json"
OBS_COLUMNS = "c_tmob c_pmob v_dens c_thermo_pe v_vol v_rg"


@dataclass
class Stage:
    name: str
    setup: str
    steps: int
    teardown: str
    dt: float = 0.0
    obs_file: str | None = None


def _steps(ps: float, dt: float) -> int:
    return max(1, round(ps * 1000.0 / dt))


def _ave_time(p: dict, dt: float, fname: str) -> str:
    nfreq = _steps(p["observe_every_ps"], dt)
    nevery = max(1, nfreq // 50)
    nfreq -= nfreq % nevery
    return f"fix obs all ave/time {nevery} {nfreq // nevery} {nfreq} {OBS_COLUMNS} file {fname}"


def build_stages(meta: dict, p: dict) -> tuple[str, list[Stage]]:
    surface = meta["surface"] is not None
    has_ions = bool(meta["ions"])
    T, P = p["temperature"], p["pressure"]
    lt = meta["lammps_types"]
    seed = int(meta["config"]["seed"]) % 900000 + 1

    header = f"""# flockit {__version__}: protocol written {datetime.now():%Y-%m-%d %H:%M}
units           real
atom_style      full
boundary        p p p
pair_style      lj/cut/coul/long {p['cutoff']}
bond_style      harmonic
angle_style     harmonic
dihedral_style  opls
improper_style  cvff
special_bonds   lj/coul 0.0 0.0 0.5
read_data       {DATA_FILE}
include         {SETTINGS_FILE}
kspace_style    pppm {p['kspace_accuracy']}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
"""
    if surface:
        header += ("neigh_modify    exclude group surface surface\n"
                   "fix             freeze surface setforce 0.0 0.0 0.0\n")
    header += """compute         tmob mobile temp
compute         pmob all pressure tmob
compute         rg polymer gyration
variable        dens equal density
variable        vol equal vol
variable        rg equal c_rg
thermo_style    custom step time c_tmob c_pmob pe ke etotal density vol c_rg
thermo_modify   flush yes
"""
    m = p["minimize"]
    stages = [Stage("minimize", f"""
# ---- 1) minimization (no SHAKE; water uses the harmonic constants)
thermo          100
min_style       cg
""", 0, f"""minimize        {m['etol']} {m['ftol']} {m['maxiter']} {m['maxeval']}
reset_timestep  0
write_data      01_min.data pair ij
""")]

    h = p["heat"]
    shake = f"fix             shk mobile shake 1.0e-4 20 0 b {lt['water_bond_type']} a {lt['water_angle_type']} m 1.008"
    stages.append(Stage("heat", f"""
# ---- 2) NVT heating
{shake}
velocity        mobile create {T} {seed} dist gaussian mom yes rot no
timestep        {h['dt']}
thermo          {_steps(p['observe_every_ps'], h['dt'])}
fix             integ mobile nvt temp {T} {T} {p['tdamp']}
{_ave_time(p, h['dt'], '02_heat.obs')}
""", _steps(h["ps"], h["dt"]), """unfix           obs
unfix           integ
""", h["dt"], "02_heat.obs"))

    n = p["npt"]
    # with the frozen slab, only mobile atoms are rescaled when Lz changes
    baro = f"z {P} {P} {p['pdamp']} dilate mobile" if surface else f"iso {P} {P} {p['pdamp']}"
    stages.append(Stage("npt", f"""
# ---- 3) NPT ({'z only, frozen slab' if surface else 'isotropic'})
timestep        {n['dt']}
thermo          {_steps(p['observe_every_ps'], n['dt'])}
fix             integ mobile npt temp {T} {T} {p['tdamp']} {baro}
{_ave_time(p, n['dt'], '03_npt.obs')}
""", _steps(n["ps"], n["dt"]), """unfix           obs
unfix           integ
write_data      03_npt.data pair ij
""", n["dt"], "03_npt.obs"))

    pr = p["production"]
    if pr["ensemble"] == "npt":
        integ = f"fix             integ mobile npt temp {T} {T} {p['tdamp']} {baro}"
    elif pr["ensemble"] == "nvt":
        integ = f"fix             integ mobile nvt temp {T} {T} {p['tdamp']}"
    else:
        raise ValueError("production.ensemble must be nvt or npt")
    if pr.get("dump_water"):
        dump_group = "all"
        group_cmd = ""
    else:
        dump_group = "dumpgrp"
        group_cmd = "group           dumpgrp union polymer ions\n" if has_ions else "group           dumpgrp union polymer\n"
    stages.append(Stage("production", f"""
# ---- 4) production
reset_timestep  0
timestep        {pr['dt']}
thermo          {_steps(p['observe_every_ps'], pr['dt'])}
{integ}
{group_cmd}dump            traj {dump_group} custom {_steps(pr['dump_every_ps'], pr['dt'])} 04_prod.lammpstrj id mol type xu yu zu
dump_modify     traj sort id
{_ave_time(p, pr['dt'], '04_prod.obs')}
""", _steps(pr["ps"], pr["dt"]), """write_data      04_final.data pair ij
write_restart   04_final.restart
""", pr["dt"], "04_prod.obs"))
    return header, stages


def _protocol_record(meta: dict, p: dict, stages: list[Stage]) -> dict:
    return {"flockit_version": __version__, "protocol": p,
            "stages": [{k: v for k, v in asdict(s).items() if k not in ("setup", "teardown")} for s in stages],
            "production_dt_fs": p["production"]["dt"],
            "dump_every_steps": _steps(p["production"]["dump_every_ps"], p["production"]["dt"])}


def write_input(outdir: Path, meta: dict, p: dict) -> Path:
    header, stages = build_stages(meta, p)
    text = header + "".join(
        s.setup + (f"run             {s.steps}\n" if s.steps else "") + s.teardown for s in stages)
    path = Path(outdir) / INPUT_FILE
    path.write_text(text, encoding="ascii")
    (Path(outdir) / PROTOCOL_FILE).write_text(json.dumps(_protocol_record(meta, p, stages), indent=1),
                                              encoding="utf-8")
    return path


def run(outdir: Path, meta: dict, p: dict, omp: int = 1, echo: Callable[[str], None] = print) -> bool:
    """Run the protocol through the LAMMPS Python module. Returns True if the NPT stage converged."""
    from lammps import lammps  # noqa: PLC0415 (optional dependency)

    header, stages = build_stages(meta, p)
    write_input(outdir, meta, p)          # keep in.flockit equivalent to what was run
    conv = p["convergence"]
    cwd = os.getcwd()
    os.chdir(outdir)
    args = ["-log", "log.lammps", "-screen", "none"]
    if omp > 1:
        args += ["-sf", "omp", "-pk", "omp", str(omp)]
    lmp = lammps(cmdargs=args)
    converged = False
    try:
        lmp.commands_string(header)
        for s in stages:
            echo(f"[flockit] stage {s.name}" + (f": {s.steps} steps of {s.dt} fs" if s.steps else ""))
            lmp.commands_string(s.setup)
            if s.name == "npt":
                for ext in range(int(p["npt"]["max_extensions"]) + 1):
                    if ext:
                        echo(f"[flockit] NPT extended (+{s.steps} steps, extension {ext})")
                    lmp.command(f"run {s.steps}")
                    report = check(s.obs_file, conv["observables"], conv["rel_tol"], conv["window"])
                    echo(report.table())
                    if report.converged:
                        converged = True
                        break
                else:
                    echo("[flockit] WARNING: NPT did not converge within the extension limit")
            elif s.steps:
                lmp.command(f"run {s.steps}")
            lmp.commands_string(s.teardown)
    finally:
        lmp.close()
        os.chdir(cwd)
    return converged
