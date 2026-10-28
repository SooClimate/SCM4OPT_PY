# SPDX-License-Identifier: GPL-3.0-or-later
"""The variable registry: one row per output, report symbol and parameter.

registry/registry.csv (in the package data: registry/registry.csv) has the columns

  name, kind (output | report | parameter), module, dims, unit, time_basis
  (state | per_step | per_year | none), description, reference,
  report_order (position in the model report, empty if not reported),
  report_group (variable | parameter: how the RCMIP export treats a reported symbol)

and is the single source of truth for:
  report_symbols()      the report symbols in report order (report.py)
  report_parameters()   the reported symbols that are parameters (zero rows dropped)
  per_step_names()      outputs/report symbols stored per model step (the RCMIP export
                        sums their steps into yearly values)
  check_registry()      unit checks: unit grammar (units.py), time basis against the unit,
                        names, report order, and every RCMIP3 export conversion
                        (output_mapping.csv scale x steps per year) against the units
  check_outputs(out)    every array of a run is registered, with matching dimensions
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .units import STEPS_PER_YEAR, UnitError, factor, parse, time_denominator
from .config import DAT_DIR, PACKAGE_DATA_DIR, data_path

_DEV_CSV = Path(__file__).resolve().parent / "registry" / "registry.csv"
REGISTRY_CSV = _DEV_CSV if _DEV_CSV.exists() else PACKAGE_DATA_DIR / "registry" / "registry.csv"

KINDS = ("output", "report", "parameter")
TIME_BASES = ("state", "per_step", "per_year", "none")
REPORT_GROUPS = ("variable", "parameter")
RCMIP_TOLERANCE = 2e-3   # the export uses rounded molar-mass ratios (44/12, 44/28, 17/14, 2)

# Known RCMIP3 export unit problems, kept so the export stays identical to the SCM4OPT v4.1
# reference submission files (see the registry descriptions of these variables):
#   doe_heatcontent  1e22 J exported as ZJ (x10 missing); its interior part is summed
#                    without dt (6x too large)
#   doe_heatflux     a rate in 1e22 J/yr, exported as ZJ/yr (x10 missing) and summed over
#                    the steps of a year like a per-step amount (x6); its mixed-layer part
#                    is a per-step energy
KNOWN_RCMIP_ISSUES = frozenset(("doe_heatcontent", "doe_heatflux"))


@lru_cache(maxsize=None)
def load_registry(path=None):
    """The registry as a DataFrame of strings (empty cells as "")."""
    return pd.read_csv(path or REGISTRY_CSV, dtype=str, keep_default_na=False)


def variables(reg=None):
    """Rows of the outputs and report symbols, indexed by name (names are unique there)."""
    reg = load_registry() if reg is None else reg
    return reg[reg["kind"] != "parameter"].set_index("name")


def report_symbols(reg=None):
    """Report symbols in report order."""
    v = variables(reg)
    v = v[v["report_order"] != ""]
    return tuple(v.index[v["report_order"].astype(int).argsort(kind="stable")])


def report_parameters(reg=None):
    v = variables(reg)
    return frozenset(v.index[(v["report_order"] != "") & (v["report_group"] == "parameter")])


def per_step_names(reg=None):
    v = variables(reg)
    return frozenset(v.index[v["time_basis"] == "per_step"])


# ---------------------------------------------------------------- checks
def check_registry(reg=None, mapping=None, known=KNOWN_RCMIP_ISSUES):
    """Problems found in the registry: a list of strings (empty if none).

    mapping : the RCMIP3 output mapping (report.load_output_mapping()); its conversions
              are checked when given (known: see check_rcmip_mapping).
    """
    reg = load_registry() if reg is None else reg
    err = []
    for i, r in reg.iterrows():
        where = f"{r['module']}.{r['name']} ({r['kind']})"
        if r["kind"] not in KINDS:
            err.append(f"{where}: kind {r['kind']!r} not in {KINDS}")
        if r["time_basis"] not in TIME_BASES:
            err.append(f"{where}: time_basis {r['time_basis']!r} not in {TIME_BASES}")
        if not r["unit"]:
            err.append(f"{where}: no unit")
            continue
        try:
            parse(r["unit"])
        except UnitError as e:
            err.append(f"{where}: {e}")
            continue
        den = time_denominator(r["unit"])
        want = {"step": "per_step", "yr": "per_year"}.get(den)
        if want and r["time_basis"] != want:
            err.append(f"{where}: unit {r['unit']!r} needs time_basis {want}, not {r['time_basis']}")
        if not want and r["time_basis"] in ("per_step", "per_year"):
            err.append(f"{where}: time_basis {r['time_basis']} but unit {r['unit']!r} has no /step or /yr")
        if r["kind"] != "parameter" and r["time_basis"] == "none":
            err.append(f"{where}: an output needs time_basis state, per_step or per_year")
        if not r["description"]:
            err.append(f"{where}: no description")
    v = reg[reg["kind"] != "parameter"]
    for n in sorted(set(v["name"][v["name"].duplicated()])):
        err.append(f"{n}: registered more than once as output/report")
    p = reg[reg["kind"] == "parameter"]
    for (m, n) in sorted(set(map(tuple, p[["module", "name"]][p[["module", "name"]].duplicated()].values))):
        err.append(f"{m}.{n}: parameter registered more than once")
    rep = v[v["report_order"] != ""]
    order = rep["report_order"].astype(int)
    if sorted(order) != list(range(1, len(order) + 1)):
        err.append("report_order is not 1..n without gaps or repeats")
    bad = rep[~rep["report_group"].isin(REPORT_GROUPS)]
    err += [f"{n}: report_group {g!r} not in {REPORT_GROUPS}" for n, g in zip(bad["name"], bad["report_group"])]
    if mapping is not None:
        err += check_rcmip_mapping(mapping, reg, known=known)
    return err


def rcmip_conversions(mapping, reg=None):
    """One row per reported RCMIP3 variable: registry unit, RCMIP unit, the factor
    implied by the units and the factor the export applies (scale x steps per year)."""
    v = variables(reg)
    rows = []
    for _, mp in mapping.iterrows():
        name = mp["rpt_var"]
        unit_rcmip = str(mp["Unit"]).replace("MAGICC ", "")
        if name not in v.index:
            rows.append(dict(rpt_var=name, variable=mp["Variable"], unit=None, unit_rcmip=unit_rcmip,
                             factor_units=None, factor_export=None, problem="not in the registry"))
            continue
        r = v.loc[name]
        # the time scaling the export applies (report.load_output_mapping), else the registry's
        steps = (float(mp["time_scale"]) if "time_scale" in mp else
                 STEPS_PER_YEAR if r["time_basis"] == "per_step" else 1)
        applied = float(mp["rpt_var_scale"]) * steps
        try:
            f = factor(r["unit"], unit_rcmip)
            problem = "" if abs(applied / f - 1) <= RCMIP_TOLERANCE else "factor mismatch"
        except UnitError as e:
            f, problem = None, str(e)
        rows.append(dict(rpt_var=name, variable=mp["Variable"], unit=r["unit"], unit_rcmip=unit_rcmip,
                         factor_units=f, factor_export=applied, problem=problem))
    return pd.DataFrame(rows)


def check_rcmip_mapping(mapping, reg=None, produced=None, known=KNOWN_RCMIP_ISSUES):
    """Problems of the RCMIP3 export conversions (see rcmip_conversions).

    produced : names the report can produce; mapped variables outside it are skipped
               by the export and reported here only if registered.
    known    : variables whose problems are known and not reported (KNOWN_RCMIP_ISSUES);
               pass () to see them.
    """
    c = rcmip_conversions(mapping, reg)
    out = []
    for _, r in c[(c["problem"] != "") & ~c["rpt_var"].isin(known)].iterrows():
        if r["problem"] == "not in the registry" and (produced is None or r["rpt_var"] not in produced):
            continue                     # mapped, but the model does not report it
        detail = (f" (units give {r['factor_units']:.6g}, export applies {r['factor_export']:.6g})"
                  if r["problem"] == "factor mismatch" else "")
        out.append(f"RCMIP {r['variable']} <- {r['rpt_var']} [{r['unit']} -> {r['unit_rcmip']}]: "
                   f"{r['problem']}{detail}")
    return out


def check_outputs(out, reg=None):
    """Problems of a run's outputs against the registry: unregistered arrays and
    dimension mismatches (number of dimensions; dims 't,...' in the registry)."""
    v = variables(reg)
    err = []
    for name, a in out.items():
        if not hasattr(a, "ndim"):
            continue
        if name not in v.index:
            err.append(f"{name}: output not in the registry")
            continue
        dims = [d for d in v.loc[name, "dims"].split(",") if d]
        if dims and len(dims) != a.ndim:
            err.append(f"{name}: {a.ndim}-D array, registry dims {v.loc[name, 'dims']!r}")
    return err


def default_mapping():
    """The RCMIP3 output mapping with the reported rows, with the export's time scaling
    (report.load_output_mapping)."""
    from ..io.report import load_output_mapping
    return load_output_mapping(data_path(DAT_DIR, "RCMIP3_src", "output_mapping.csv"))


def main(argv=None):
    """python3 -m scm4opt.core.registry [--all]: run the registry checks and print a
    summary (--all also shows the known RCMIP3 export issues). Exit code 1 on problems."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    reg = load_registry()
    known = () if "--all" in argv else KNOWN_RCMIP_ISSUES
    err = check_registry(reg, default_mapping(), known=known)
    filled = {c: int((reg[c] != "").sum()) for c in ("unit", "time_basis", "description")}
    print(f"registry {REGISTRY_CSV}: {len(reg)} rows, filled {filled}, "
          f"{len(report_symbols(reg))} report symbols, {len(per_step_names(reg))} per-step variables")
    print(f"{len(err)} problems" + ("" if known == () else f" (known RCMIP3 issues not shown: "
                                    f"{', '.join(sorted(KNOWN_RCMIP_ISSUES))}; --all)"))
    for e in err:
        print("  " + e)
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
