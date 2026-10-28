# SPDX-License-Identifier: GPL-3.0-or-later
"""Units of the variable registry: parsing, dimensions and conversion factors.

Grammar of a registry unit (registry/registry.csv, column unit):

  unit  := "1" | [number " "] term {" " term} {"/" term}
  term  := symbol[exponent]          e.g. m2, m3 (no negative exponents: write W/m2/K)
  mass terms may be followed by one species: "Mt CO2", "kt HFC23", "Mt N", "GtC"

  e.g.  ppm  ppb  ppt  K  W/m2  W/m2/K  GtC/step  Mt CO2/yr  kt CFC11/step  Mt N/step
        1/yr  yr  umol/kg  uatm  kg/m3  1e22 J  ZJ/yr  DU  %  1

parse(unit) returns a Quantity (factor to base units, dimensions, species);
factor(a, b) is the number that converts a value in unit a into unit b. Species
conversions go through a shared element (GtC -> Mt CO2, Mt N -> Mt N2O, Mt S -> Mt SO2).
A model step is 1/6 year, so "GtC/step" -> "GtC/yr" is a factor of 6.
"""

import re
from dataclasses import dataclass

YEAR_S = 365.25 * 86400.0
STEPS_PER_YEAR = 6

# symbol -> (factor to base, {dimension: exponent})
#   base: kg, m, s, K, mol, J; "X" = mole fraction (ppm/ppb/ppt), "DU" = ozone column
_M, _L, _T, _K, _N, _E = "kg", "m", "s", "K", "mol", "J"
SYMBOLS = {
    "1": (1.0, {}), "%": (0.01, {}),
    "ppm": (1e-6, {"X": 1}), "ppb": (1e-9, {"X": 1}), "ppt": (1e-12, {"X": 1}),
    "DU": (1.0, {"DU": 1}),
    "g": (1e-3, {_M: 1}), "kg": (1.0, {_M: 1}), "t": (1e3, {_M: 1}), "kt": (1e6, {_M: 1}),
    "Mt": (1e9, {_M: 1}), "Tg": (1e9, {_M: 1}), "Gt": (1e12, {_M: 1}), "Pg": (1e12, {_M: 1}),
    "mol": (1.0, {_N: 1}), "mmol": (1e-3, {_N: 1}), "umol": (1e-6, {_N: 1}),
    "J": (1.0, {_E: 1}), "ZJ": (1e21, {_E: 1}), "W": (1.0, {_E: 1, _T: -1}),
    "s": (1.0, {_T: 1}), "yr": (YEAR_S, {_T: 1}), "step": (YEAR_S / STEPS_PER_YEAR, {_T: 1}),
    "m": (1.0, {_L: 1}), "cm": (1e-2, {_L: 1}), "km": (1e3, {_L: 1}),
    "K": (1.0, {_K: 1}), "degC": (1.0, {_K: 1}),   # differences; offsets are not converted
    "Pa": (1.0, {"Pa": 1}), "atm": (101325.0, {"Pa": 1}), "uatm": (0.101325, {"Pa": 1}),
}
# carbon shorthands: GtC = Gt C
CARBON = {"GtC": "Gt", "PgC": "Pg", "MtC": "Mt", "TgC": "Tg"}
MASS = {k for k, (_, d) in SYMBOLS.items() if d == {_M: 1}}

# species: element content and molar mass [g/mol]; other species convert only to themselves
_ATOM = {"C": 12.0107, "N": 14.0067, "S": 32.065, "O": 15.9994, "H": 1.00794}
SPECIES = {
    "C": {"C": 1}, "CO2": {"C": 1, "O": 2},
    "N": {"N": 1}, "N2O": {"N": 2, "O": 1}, "NO2": {"N": 1, "O": 2}, "NH3": {"N": 1, "H": 3},
    "S": {"S": 1}, "SO2": {"S": 1, "O": 2},
}
SPECIES_ALIAS = {"NOX": "NO2"}             # NOx emissions are given as NO2 mass


def molar_mass(formula):
    return sum(_ATOM[e] * n for e, n in formula.items())


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    factor: float              # value in this unit * factor = value in base units
    dims: tuple                # sorted ((dimension, exponent), ...)
    species: str | None = None


def _norm_species(s):
    u = s.upper()
    return SPECIES_ALIAS.get(u, u)


def _term(tok):
    """One symbol with an optional positive exponent (m2, m3)."""
    if tok in SYMBOLS:
        return SYMBOLS[tok]
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", tok)
    if m and m.group(1) in SYMBOLS:
        f, d = SYMBOLS[m.group(1)]
        n = int(m.group(2))
        return f ** n, {k: v * n for k, v in d.items()}
    raise UnitError(f"unknown unit symbol {tok!r}")


def parse(unit):
    """Quantity of a registry unit string (see the module docstring)."""
    u = str(unit).strip().replace("^", "")
    if not u:
        raise UnitError("empty unit")
    parts = u.split("/")
    factor, dims, species = 1.0, {}, None

    def add(f, d, sign):
        nonlocal factor
        factor *= f if sign > 0 else 1.0 / f
        for k, v in d.items():
            dims[k] = dims.get(k, 0) + sign * v

    toks = parts[0].split()
    if toks and re.fullmatch(r"[-+]?\d+(\.\d*)?([eE][-+]?\d+)?", toks[0]) and len(toks) > 1:
        factor *= float(toks.pop(0))
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in CARBON:
            add(*SYMBOLS[CARBON[tok]], 1)
            species = _merge_species(species, "C", unit)
        elif tok in MASS and i + 1 < len(toks) and toks[i + 1] not in SYMBOLS:
            add(*SYMBOLS[tok], 1)
            species = _merge_species(species, _norm_species(toks[i + 1]), unit)
            i += 1
        else:
            add(*_term(tok), 1)
        i += 1
    for p in parts[1:]:
        toks = p.split()
        if len(toks) == 2 and toks[0] in MASS:     # e.g. K/Mt CO2 (species not tracked)
            toks = toks[:1]
        if len(toks) != 1:
            raise UnitError(f"{unit!r}: one term per denominator (use a/b/c)")
        if toks[0] in CARBON:                       # e.g. ppm/GtC (species not tracked)
            add(*SYMBOLS[CARBON[toks[0]]], -1)
        else:
            add(*_term(toks[0]), -1)
    return Quantity(factor, tuple(sorted((k, v) for k, v in dims.items() if v)), species)


def _merge_species(old, new, unit):
    if old is not None and old != new:
        raise UnitError(f"{unit!r}: more than one species")
    return new


def time_denominator(unit):
    """'step', 'yr', 's' or None: the time unit in the denominator of a unit string."""
    dens = [re.sub(r"\d+$", "", p.strip()) for p in str(unit).replace("^", "").split("/")[1:]]
    for t in ("step", "yr", "s"):
        if t in dens:
            return t
    return None


def _species_factor(a, b):
    """Mass of species b per mass of species a (same amount of the shared element)."""
    if a == b:
        return 1.0
    if a is None or b is None:
        raise UnitError(f"species {a!r} and {b!r} are not convertible")
    fa, fb = SPECIES.get(a), SPECIES.get(b)
    if fa is None or fb is None:
        raise UnitError(f"species {a!r} and {b!r} are not convertible")
    common = [e for e in fa if e in fb and e not in ("O", "H")]
    if len(common) != 1:
        raise UnitError(f"species {a!r} and {b!r} share no single element")
    e = common[0]
    moles_e = 1.0 / molar_mass(fa) * fa[e]           # mol element per g of a
    return moles_e / fb[e] * molar_mass(fb)         # g of b per g of a


def factor(a, b):
    """Multiply a value in unit a by this number to get it in unit b."""
    qa, qb = parse(a), parse(b)
    if qa.dims != qb.dims:
        raise UnitError(f"{a!r} and {b!r} have different dimensions ({qa.dims} vs {qb.dims})")
    return qa.factor / qb.factor * _species_factor(qa.species, qb.species)
