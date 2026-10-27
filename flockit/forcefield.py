"""Parameters: OPLS-AA 2024 (polymer), SPC/E water, Joung-Cheatham ions, model surface.

Bonded terms are looked up following the OPLS convention: matching by atom
classes (with wildcards), where the most specific pattern wins. When the
OPLS-AA file has more than one variant for the same pattern, the choice comes
from `data/oplsaa_selections.yaml`; without a rule, `ParameterError` is raised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from itertools import permutations

import yaml

_DATA = resources.files("flockit") / "data"


class ParameterError(RuntimeError):
    """Missing or ambiguous force-field parameter."""


@dataclass(frozen=True)
class AtomType:
    name: str                # "opls_136", "OW", "HW", "Na", "Cl", "SW"
    element: str
    mass: float
    charge: float
    epsilon: float
    sigma: float
    mixing: str              # "geometric" (OPLS, water) or "arithmetic" (Joung-Cheatham ions)
    description: str
    classes: dict | None = None


@dataclass(frozen=True)
class Term:
    coeffs: tuple[float, ...]
    label: str               # e.g. "CT_CT" or "CT_CT_CT_C~#3"
    comment: str = ""


_ORIENT = {
    "bond": [(0, 1), (1, 0)],
    "angle": [(0, 1, 2), (2, 1, 0)],
    "dihedral": [(0, 1, 2, 3), (3, 2, 1, 0)],
    "improper": [(0, *p) for p in permutations((1, 2, 3))],  # central atom always first
}


def opls_name(type_id: int | str) -> str:
    return f"opls_{type_id}"


class ForceField:
    def __init__(self) -> None:
        self._opls = json.loads((_DATA / "oplsaa2024.json").read_text(encoding="utf-8"))
        self._wc = self._opls["wildcard_char"]
        self._selections = yaml.safe_load(
            (_DATA / "oplsaa_selections.yaml").read_text(encoding="utf-8")).get("dihedrals", [])
        extra = yaml.safe_load((_DATA / "water_ions.yaml").read_text(encoding="utf-8"))
        self.water = extra["water"]["spce"]
        self._types: dict[str, AtomType] = {}
        for name in ("OW", "HW"):
            p = self.water[name]
            self._types[name] = AtomType(name, name[0], p["mass"], p["charge"], p["epsilon"],
                                         p["sigma"], "geometric", "SPC/E water")
        for name, p in extra["ions"].items():
            sigma = 2.0 * p["rmin_half"] / 2.0 ** (1.0 / 6.0)
            self._types[name] = AtomType(name, name, p["mass"], p["charge"], p["epsilon"],
                                         sigma, "arithmetic", "Joung-Cheatham ion (SPC/E)")
        p = extra["surface"]["SW"]
        self._types["SW"] = AtomType("SW", "O", p["mass"], 0.0, p["epsilon"], p["sigma"],
                                     "geometric", "model surface site (frozen)")
        self._cache: dict = {}

    # ---- atom types --------------------------------------------------------
    def atom_type(self, name: str) -> AtomType:
        if name in self._types:
            return self._types[name]
        if not name.startswith("opls_"):
            raise ParameterError(f"unknown atom type: {name}")
        raw = self._opls["atom_types"].get(name[5:])
        if raw is None:
            raise ParameterError(f"{name} does not exist in OPLS-AA 2024")
        t = AtomType(name, raw["element"], raw["mass"], raw["charge"], raw["epsilon"],
                     raw["sigma"], "geometric", raw["description"], raw["classes"])
        self._types[name] = t
        return t

    # ---- bonded terms ------------------------------------------------------
    def bond(self, a: str, b: str) -> Term:
        return self._lookup("bond", (a, b))[0]

    def angle(self, a: str, b: str, c: str) -> Term:
        return self._lookup("angle", (a, b, c))[0]

    def dihedral(self, a: str, b: str, c: str, d: str) -> Term:
        return self._lookup("dihedral", (a, b, c, d))[0]

    def improper(self, center: str, n1: str, n2: str, n3: str) -> tuple[Term, tuple[int, ...]] | None:
        """Improper term and atom order (indices 0..3, central atom first), or None."""
        try:
            return self._lookup("improper", (center, n1, n2, n3))
        except ParameterError:
            return None

    def _lookup(self, kind: str, names: tuple[str, ...]) -> tuple[Term, tuple[int, ...]]:
        key = (kind, names)
        if key not in self._cache:
            self._cache[key] = self._resolve(kind, names)
        return self._cache[key]

    def _score(self, pattern: list[str], classes: list[str]) -> int | None:
        score = 0
        for p, c in zip(pattern, classes):
            if p == c:
                score += 2
            elif len(p) == len(c) and all(pc in (self._wc, cc) for pc, cc in zip(p, c)):
                score += 0 if set(p) == {self._wc} else 1
            else:
                return None
        return score

    def _resolve(self, kind: str, names: tuple[str, ...]) -> tuple[Term, tuple[int, ...]]:
        types = [self.atom_type(n) for n in names]
        if any(t.classes is None for t in types):
            raise ParameterError(f"{kind} {names}: types outside OPLS-AA have no bonded terms here")
        classes = [t.classes[kind] for t in types]
        hits: list[tuple[int, str, tuple[int, ...]]] = []
        for pattern_key in self._opls[kind + "s"]:
            pattern = pattern_key.split("_")
            for orient in _ORIENT[kind]:
                s = self._score(pattern, [classes[i] for i in orient])
                if s is not None:
                    hits.append((s, pattern_key, orient))
        if not hits:
            raise ParameterError(f"no OPLS-AA parameter for {kind} {names} (classes {classes})")
        best = max(h[0] for h in hits)
        top: dict[str, list[tuple[int, ...]]] = {}
        for s, key, orient in hits:
            if s == best:
                top.setdefault(key, []).append(orient)
        # The file sometimes lists the same pattern in both orientations (A_B_C_D and D_C_B_A):
        # that is a single pattern, with the variants of both keys.
        canonical = {min(k, "_".join(reversed(k.split("_")))) for k in top}
        if len(canonical) > 1:
            raise ParameterError(f"{kind} {names}: tied patterns {sorted(top)}")
        key, entry = self._pick_variant(kind, top, names)
        label = key + (f"#{entry['variant']}" if entry["variant"] else "")
        return Term(tuple(entry["coeffs"]), label, entry["comment"]), top[key][0]

    def _pick_variant(self, kind: str, top: dict[str, list[tuple[int, ...]]],
                      names: tuple[str, ...]) -> tuple[str, dict]:
        candidates = [(key, e) for key in top for e in self._opls[kind + "s"][key]]
        if len(candidates) == 1:
            return candidates[0]
        numbers = [n.removeprefix("opls_") for n in names]
        best = None
        for rule in (self._selections if kind == "dihedral" else []):
            for orient in top.get(rule["classes"], []):
                oriented = [numbers[i] for i in orient]
                if all(str(rt) in ("*", t) for rt, t in zip(rule["types"], oriented)):
                    spec = sum(rt != "*" for rt in rule["types"])
                    if best is None or spec > best[0]:
                        best = (spec, rule)
        if best is None:
            options = "; ".join(f"{k}#{e['variant']}: {e['coeffs']} ({e['comment']})" for k, e in candidates)
            raise ParameterError(
                f"{kind} {names} has {len(candidates)} variants in OPLS-AA 2024 [{options}]. "
                f"Add a rule to flockit/data/oplsaa_selections.yaml.")
        rule = best[1]
        for key, e in candidates:
            if key == rule["classes"] and e["variant"] == rule["variant"]:
                return key, e
        raise ParameterError(f"variant {rule['variant']} does not exist for {rule['classes']}")
