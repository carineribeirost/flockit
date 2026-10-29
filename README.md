# flockit

A minimal toolkit for simulating **flocculant polyelectrolytes** in LAMMPS, from scratch to numbers:

```
flockit build  ->  flockit run (or in.flockit on a cluster)  ->  flockit check  ->  flockit analyze
 polymer +          minimization, NVT, NPT extended               convergence        conformation, adsorption
 water + salt       until converged, production                   (drift)            (trains/loops/tails),
 (+ surface)                                                                         condensed counterions
```

This first version covers copolymers of **acrylamide (AM)** and **sodium acrylate (AA)**, i.e. PAM and
HPAM, in SPC/E water with NaCl. Optionally, the polymer sits above a **charged model surface**.

## Installation

```bash
pip install -e .            # numpy + pyyaml
pip install -e ".[plot]"    # + matplotlib for the analysis plots
```

`flockit run` needs the LAMMPS Python module with the MOLECULE, KSPACE and RIGID packages (plus
OPENMP for threads). Two tested options:

- **Linux/WSL:** `pip install "lammps==2024.8.29.3.0"`. This is a serial build with OpenMP; the
  2025 PyPI releases require an MPICH library (`libmpi.so.12`).
- **Native Windows:** the PyPI wheel requires the Microsoft MPI runtime (`msmpi.dll`). Install
  MS-MPI or use WSL.

Without the Python module, use the generated `in.flockit` with any LAMMPS executable (on a cluster,
for example). In that case the NPT stage runs for its fixed length and you check convergence with
`flockit check`.

## Usage

```bash
flockit build examples/hpam_bulk.yaml            # creates hpam_bulk/ with system.data, in.flockit, etc.
flockit run hpam_bulk --omp 4                    # runs everything, extending NPT until converged
flockit check hpam_bulk/03_npt.obs               # (re)test the convergence of any .obs file
flockit analyze hpam_bulk                        # analysis/timeseries.csv, summary.json, summary.png
```

To run on a cluster: `cd hpam_bulk && mpirun -np 32 lmp -in in.flockit`, then `flockit check` and
`flockit analyze`.

Example configuration (omitted values use the defaults in `flockit/config.py`):

```yaml
output: hpam_surface
seed: 11
polymer:
  dp: 20
  composition: {AM: 0.7, AA: 0.3}      # exact counts, shuffled sequence
  # sequence: [AM, AA, AM, ...]        # or an explicit sequence
  tacticity: atactic                   # atactic | isotactic | syndiotactic
solvent:
  salt: {NaCl: 0.1}                    # mol/L, on top of the counterions
surface:                               # omit for a system without a surface
  charge_density: 0.3                  # e/nm² on each face
  size: [45.0, 45.0]
  water_thickness: 50.0
protocol:
  npt: {ps: 500.0, max_extensions: 4}
  production: {ps: 5000.0, ensemble: nvt, dump_every_ps: 2.0}
```

### Output files

| file | contents |
|---|---|
| `system.data` | LAMMPS data file (`atom_style full`) with all coefficients and source comments |
| `system.settings` | `pair_modify`, ion cross pairs and groups (`polymer`, `water`, `ions`, `surface`, `mobile`) |
| `system.json` | metadata for the analysis: polymer ids per monomer, charged groups, ions, surface |
| `flockit.yaml` | full configuration used (defaults included), for reproducibility |
| `in.flockit`, `protocol.json` | LAMMPS input and protocol record |
| `0N_*.obs` | `fix ave/time` averages per stage: T, P, density, potential energy, volume, Rg |
| `04_prod.lammpstrj` | trajectory of the polymer and the ions (unwrapped coordinates `xu yu zu`) |
| `analysis/` | `timeseries.csv`, `summary.json`, `monomers.csv`, `profiles.csv`, `summary.png` |

## Model and choices

**Polymer: OPLS-AA.** Parameters come from moltemplate's `oplsaa2024.lt`, generated from the
supporting information of Jorgensen et al., *J. Phys. Chem. B* (2024), DOI 10.1021/acs.jpcb.3c06602,
which uses the `opls_XXX` numbering. The script `tools/extract_oplsaa.py` reproduces
`flockit/data/oplsaa2024.json`.

- In that file some dihedrals have several variants for the same class pattern (OPLS-AA/M,
  OPLS/2020, specific model compounds). flockit **never picks one silently**: the choices live in
  `flockit/data/oplsaa_selections.yaml`, each with the model compound that justifies it (e.g.
  butanamide for C-C-C-C(=O)NH2 and propanamide for C-C-C(=O)-N). If an ambiguity without a rule
  shows up, `build` stops and lists the options.
- Every coefficient in `system.data` carries a comment with the pattern and variant used.
- Beware: moltemplate's `oplsaa2008.lt` uses TINKER numbering (its `136` is not `opls_136`). That
  is why the 2024 file is the source here.

**Water and ions.** Rigid SPC/E (SHAKE). Na⁺ and Cl⁻ from Joung and Cheatham (2008), fitted for
SPC/E. Pairs involving ions use the Lorentz-Berthelot rule, with which they were derived; all other
pairs use geometric mixing (OPLS). Standard OPLS treatment: 1-4 interactions scaled by 0.5, LJ and
Coulomb cut off at 10 Å, PPPM and tail correction.

**Model surface.** Two frozen layers of sites with SPC/E oxygen LJ (a generic hydrophilic surface).
Each face's charge is spread equally over its sites. The slab sits in the middle of the box,
periodic in x and y, with water on both sides. It is a model for studying the effect of surface
charge, **not a mineral**. For kaolinite, silica, etc., replace it with a force field such as ClayFF.

LAMMPS warns "Neighbor exclusions used with KSpace solver" for these systems. This is expected:
interactions between sites of the frozen slab are excluded from the neighbor list, but PPPM still
includes them as a constant, which affects neither forces nor dynamics.

**Initial conformation.** A random coil generated by growth: trans/gauche± dihedrals, with overlap
rejection and backtracking. With a surface, the chain is rotated to lie flat over the face and
starts `polymer_gap` Å above it.

**Protocol.**

1. Conjugate-gradient minimization, without SHAKE.
2. NVT heating to T, with SHAKE on X-H bonds and water.
3. NPT at 2 fs: isotropic without a surface; z only, with `dilate mobile`, with the frozen slab.
   `fix npt` measures temperature over group `all`, which would count the slab's frozen sites and
   heat the water and polymer to ~327 K; `fix_modify integ temp tmob press pmob` makes it use the
   mobile atoms. LAMMPS then warns "Temperature for fix modify is not for group all", as intended.
4. NVT (or NPT) production.

**Convergence** (`flockit check`, applied to the NPT stage by `flockit run`): the final window of the
series (50% by default) is split in half and the means are compared. Density and potential energy
pass if the difference is below 2 standard errors (block averages) **or** below the relative
tolerance (0.5%). Rg is reported for information only, because it relaxes much more slowly than
the density and should not hold back the NPT stage.

**Analysis.**

- **Rg:** mass-weighted.
- **Rg‖ and Rg⊥:** components parallel and perpendicular to the surface.
- **κ²:** shape anisotropy (0 = sphere, 1 = rod).
- **Ree:** end-to-end distance.
- **Adsorption:** a monomer is adsorbed if any heavy atom is within `--d-ads` Å (5 Å by default)
  of a face. The chain is split into trains, loops and tails (Scheutjens-Fleer), with the
  adsorption probability per monomer type.
- **Condensed counterion:** a Na⁺ is within `--r-cond` Å (3.2 Å, first Na-O minimum) of a
  carboxylate oxygen.
- **Profiles:** number density (nm⁻³) of the polymer and ions versus distance to the face.

## Adding a monomer

Create `flockit/data/monomers/XX.yaml` following `AM.yaml`/`AA.yaml`:

- OPLS types of the backbone carbons (`CA` = CH2, `CB` = CH carrying the side group);
- head and tail variants;
- the side group in internal coordinates.

Hydrogen names must start with `H`. Run `flockit build` and, if it reports ambiguous dihedrals,
add the rules to `oplsaa_selections.yaml`.

## Known limitations (next steps)

- AM and AA only. Cationic monomers (AETAC, DADMAC) need validated OPLS charges.
- NaCl only. Ca²⁺/Mg²⁺ (important for cation bridging) need Li-Merz-type parameters.
- One chain per box. Bridging between particles needs two surfaces or several polymers.
- The surface is generic (see above).
- Adsorption free energies (PMF by umbrella sampling with `fix colvars`) are not included yet.
- Before production use, compare the generated terms with another OPLS-AA implementation
  (LigParGen or GROMACS `oplsaa.ff`) for a short oligomer.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```
