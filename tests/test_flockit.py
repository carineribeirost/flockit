import numpy as np
import pytest

from flockit import config, geometry
from flockit.analysis import segments, surface_distance
from flockit.builder import build_system, parametrize
from flockit.convergence import check
from flockit.forcefield import ForceField, ParameterError
from flockit.lammps_io import write_system
from flockit.polymer import build_chain, find_clashes, make_sequence


@pytest.fixture(scope="module")
def ff():
    return ForceField()


def test_nerf_roundtrip():
    a, b, c = np.array([0.0, 1.0, 0.0]), np.zeros(3), np.array([1.5, 0.0, 0.0])
    for phi in (-120.0, -60.0, 0.0, 60.0, 180.0):
        d = geometry.place(a, b, c, 1.2, 110.0, phi)
        assert np.linalg.norm(d - c) == pytest.approx(1.2)
        assert geometry.angle(b, c, d) == pytest.approx(110.0)
        assert geometry.dihedral(a, b, c, d) % 360 == pytest.approx(phi % 360, abs=1e-6)


def test_forcefield_lookups(ff):
    assert ff.bond("opls_136", "opls_137").coeffs == (268.0, 1.529)
    assert ff.angle("opls_272", "opls_271", "opls_272").coeffs == (80.0, 126.0)
    amide = ff.dihedral("opls_137", "opls_136", "opls_137", "opls_235")
    assert amide.comment == "butanamide"
    assert ff.dihedral("opls_235", "opls_137", "opls_136", "opls_137") == amide  # reversed order
    assert ff.dihedral("opls_137", "opls_136", "opls_275", "opls_271").comment == "carboxylate ion"
    assert ff.dihedral("opls_140", "opls_136", "opls_137", "opls_235").comment == "all carbonyls"
    term, order = ff.improper("opls_235", "opls_137", "opls_236", "opls_237")
    assert term.coeffs[0] == 10.5 and order[0] == 0
    assert ff.improper("opls_271", "opls_275", "opls_272", "opls_272") is None
    # Joung-Cheatham ions: sigma = 2 Rmin/2 / 2^(1/6)
    assert ff.atom_type("Na").sigma == pytest.approx(2.1595, abs=1e-3)


def test_ambiguous_variant_is_an_error():
    ff = ForceField()
    ff._selections = []
    with pytest.raises(ParameterError, match="variants"):
        ff.dihedral("opls_137", "opls_136", "opls_137", "opls_235")


def test_sequence_has_exact_composition():
    seq = make_sequence({"dp": 10, "composition": {"AM": 0.7, "AA": 0.3}}, np.random.default_rng(0))
    assert sorted(seq) == ["AA"] * 3 + ["AM"] * 7


@pytest.mark.parametrize("tacticity", ["atactic", "isotactic", "syndiotactic"])
def test_chain_geometry_and_charge(ff, tacticity):
    seq = ["AM", "AA", "AM", "AA", "AM", "AM", "AA", "AM"]
    chain = build_chain(seq, np.random.default_rng(5), tacticity)
    q = sum(ff.atom_type(t).charge for t in chain.types)
    assert q == pytest.approx(-3)
    assert len(chain.charged_groups) == 3 and len(chain.counterions) == 3
    lengths = [np.linalg.norm(chain.positions[a] - chain.positions[b]) for a, b in chain.bonds]
    assert 1.0 < min(lengths) and max(lengths) < 1.53
    assert find_clashes(chain) == []
    # CH3 head (opls_135) and CH2 tail with 2 H
    head_h = [n for n, m in zip(chain.names, chain.monomer) if m == 0 and n.startswith("HA")]
    tail_h = [n for n, m in zip(chain.names, chain.monomer) if m == len(seq) - 1 and n.startswith("HB")]
    assert len(head_h) == 3 and len(tail_h) == 2


def _small_cfg(**surface):
    return config.resolve({
        "seed": 3,
        "polymer": {"dp": 6, "composition": {"AM": 0.5, "AA": 0.5}},
        "solvent": {"padding": 9.0, "salt": {"NaCl": 0.2}},
        "surface": surface or None,
    })


def test_bulk_system_is_neutral_and_written(ff, tmp_path):
    cfg = _small_cfg()
    system, chain = build_system(cfg, ff)
    assert abs(sum(system.charges)) < 1e-6
    assert system.names.count("NA") == system.meta["salt"]["pairs"] + 3
    topo = parametrize(system, ff)
    meta = write_system(tmp_path, system, chain, topo, ff, cfg)
    text = (tmp_path / "system.data").read_text()
    assert f"{len(system.names)} atoms" in text and "Dihedral Coeffs # opls" in text
    assert meta["polymer"]["charge"] == -3
    assert len(meta["polymer"]["atom_ids"]) == len(chain.names)
    # initial water density (over the whole box, polymer included) just below 1 g/cm3
    L = system.lengths
    rho = meta["water"]["n"] * 18.015 / 6.022e23 / (np.prod(L) * 1e-24)
    assert 0.85 < rho < 1.0


def test_surface_system(ff, tmp_path):
    cfg = _small_cfg(charge_density=0.5, size=[30.0, 30.0], water_thickness=40.0)
    system, chain = build_system(cfg, ff)
    s = system.meta["surface"]
    assert abs(sum(system.charges)) < 1e-6
    assert s["charge_per_face"] == round(0.5 * s["area_nm2"])
    zp = system.positions("polymer")[:, 2]
    assert zp.min() >= s["z_top"] + cfg["surface"]["polymer_gap"] - 1e-6
    # the charges as written to the file are also neutral (LAMMPS tolerates 1e-5)
    write_system(tmp_path, system, chain, parametrize(system, ff), ff, cfg)
    lines = (tmp_path / "system.data").read_text().split("Atoms # full")[1].split("Bonds")[0].split("\n")
    q = sum(float(line.split()[3]) for line in lines if line.strip())
    assert abs(q) < 1e-6


def test_segments():
    ads = np.array([0, 1, 1, 0, 0, 1, 0], bool)
    assert segments(ads) == ([2, 1], [2], [1, 1])
    assert segments(np.zeros(4, bool)) == ([], [], [])
    assert segments(np.ones(3, bool)) == ([3], [], [])


def test_surface_distance_wraps_through_bottom_face():
    # box z in [-10, 10], slab between -1 and 1; z = 31 is equivalent to -9, 8 A below the bottom face
    d = surface_distance(np.array([3.0, 9.0, -4.0, 31.0]), -10.0, 20.0, 1.0, -1.0)
    assert d == pytest.approx([2.0, 8.0, 3.0, 8.0])


def test_convergence_detects_drift(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(200)
    header = "# Time-averaged data for fix obs\n# TimeStep c_tmob c_pmob v_dens c_thermo_pe v_vol v_rg\n"
    for name, dens in (("flat", 1.0 + 0.002 * rng.normal(size=200)), ("drift", np.linspace(0.8, 1.0, 200))):
        rows = np.column_stack([t, np.full(200, 300.0), np.zeros(200), dens,
                                -1000 + rng.normal(size=200), np.ones(200), np.full(200, 10.0)])
        path = tmp_path / f"{name}.obs"
        path.write_text(header + "\n".join(" ".join(map(str, r)) for r in rows))
        report = check(path, ["density", "pe"], {"density": 0.005, "pe": 0.005})
        assert report.converged == (name == "flat")
