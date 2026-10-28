# SPDX-License-Identifier: GPL-3.0-or-later
"""Model report and RCMIP3 submission rows.

With --rout3=<dir>, a run writes the model report and the RCMIP3 submission rows of
its ensemble member:

  build_report(out, inp, params)  the report symbols, named as the model variables
  write_report_csv(...)           <rout3>/<rname>_<case>_<ctag>.csv: one row per model
                                  step, one column per report symbol (2-D symbols such
                                  as rfc_HFC become rfc_HFC.<gas> columns)
  write_rcmip_csv(...)            the RCMIP3 template rows for this
                                  ensemble member (output_mapping.csv + template xlsx)

REPORT_SYMBOLS lists the report symbols in the reference report order and
REPORT_PARAMETERS the reported parameters, both generated from the variable registry
(varregistry.py); the RCMIP3 export sums the steps of the variables the registry lists
as per_step. NCYCLE_SYMBOLS are reported only with --use_ncycle (the N-cycle module is
not available here, so they are skipped).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import DAT_DIR, data_path
from ..core.constants import DT
from ..core.sets import HALO, HALO_HFC, HALO_ODS, HALO_PFC
from ..core.units import factor as unit_factor
from ..core.registry import (KNOWN_RCMIP_ISSUES, per_step_names, report_parameters, report_symbols,
                          variables)
from ..core.version import RCMIP_CLIMATE_MODEL, RCMIP_FILE_PREFIX

# the report symbols in report order and the reported parameters: generated from the
# registry (registry.csv columns report_order / report_group; varregistry.py)
REPORT_SYMBOLS = report_symbols()

NCYCLE_SYMBOLS = (
    "nit_plant", "nit_litter", "nit_soil", "nit_mineral",
    "nit_uptake", "nit_litterfall", "nit_mineralization", "nit_leaching",
    "nit_denit", "nit_soilresp", "nit_netmin", "nit_pureq",
    "nit_puavail_netmin", "nit_puavail_stock", "nit_puavail", "nit_uptake_netmin",
    "nit_uptake_stock", "nit_fixation_dyn", "nit_luc_loss", "nit_luc_loss_plant",
    "nit_luc_loss_litter", "nit_luc_loss_soil", "nit_temp_factor", "nit_supply_total",
    "nit_use_total", "nit_pudef", "nit_limitation", "nit_npp_scaler",
    "nit_pudef_ratio", "nit_stock_share", "nit_bnf_ratio", "nit_budget_resid",
)

# Report symbols that are parameters (the rest are variables), report_group in the
# registry. Deliberately kept identical to the SCM4OPT v4.1 reference submission files:
# a parameter that is zero everywhere gets no row.
REPORT_PARAMETERS = report_parameters()

CO2_GTC_TO_MT = 44.01 / 12.0107 * 1000      # GtC -> MtCO2
_SUBSETS = {"rfc_HFC": HALO_HFC, "rfc_PFC": HALO_PFC, "rfc_ODS": HALO_ODS}
_NONF = ("co2", "ch4", "n2o", "so2", "co", "voc", "nox", "bc", "oc", "nh3")


def build_report(out, inp, doe_params):
    """The report symbols from a coupled run.

    out        : scm4opt.run() output; inp : ScenarioInputs; doe_params : DoeclimParams
    Returns {symbol: array (n,)} for 1-D symbols and {symbol: (n, k) array} for
    rfc_HFC / rfc_PFC / rfc_ODS (columns in their halo-subset order).
    """
    t = inp.t
    y = t.tval
    co2 = inp.gi("co2")
    r = {}
    # derived quantities
    r["rfc_O3"] = out["rfc_O3t"] + out["rfc_O3s"]
    since_2017 = y >= 2017
    for name, series in (("eco2", inp.ems[:, co2]), ("eco2i", inp.eind[:, co2]),
                         ("eco2l", inp.elnd[:, co2])):
        r[f"{name}_accum_1850"] = np.cumsum(series) * CO2_GTC_TO_MT
        r[f"{name}_accum_2017"] = np.where(since_2017,
                                           np.cumsum(np.where(since_2017, series, 0.0)), 0.0) * CO2_GTC_TO_MT
    r["atmoSeaFlux_accum_1850"] = np.cumsum(out["scm_atmoSeaFlux"]) * CO2_GTC_TO_MT
    r["atmoLndFlux_accum_1850"] = np.cumsum(out["scm_atmoLndFlux"]) * CO2_GTC_TO_MT
    acc = r["eco2_accum_1850"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r["inst_TCRE"] = np.where(acc != 0, out["scm_tatm"] / acc, 0.0)
    r["doe_t2co_t"] = np.full(t.n, doe_params.climate_sensitivity)
    # reference behaviour (ref_compat): the interior heat gets frac_ocean_deep once more
    r["ocn_heat"] = out["doe_heat_mixed"] + (doe_params.frac_ocean_deep if doe_params.ref_compat
                                             else 1.0) * out["doe_heat_interior"]
    # per-gas series
    for j, g in enumerate(HALO):
        r[f"chm_D_HALO_{g}"] = out["chm_D_HALO"][:, j]
        r[f"rfc_HALO_{g}"] = out["rfc_HALO"][:, j]
        r[f"erf_HALO_{g}"] = out["erf_HALO"][:, j]
    for name, sub in _SUBSETS.items():
        r[name] = out["rfc_HALO"][:, [HALO.index(g) for g in sub]]
    # emissions as used by the model (dat_sce_ems_*)
    for g in inp.gas:
        col = inp.gi(g)
        r[f"dat_sce_ems_{g}i" if g in _NONF else f"dat_sce_ems_{g}"] = inp.eind[:, col]
    for g in _NONF:
        r[f"dat_sce_ems_{g}l"] = inp.elnd[:, inp.gi(g)]
        r[f"dat_sce_ems_{g}"] = r[f"dat_sce_ems_{g}i"] + r[f"dat_sce_ems_{g}l"]
    r["lnd_regrow"] = np.zeros(t.n)         # declared in the land module, no equation: level 0
    rep = {}
    for name in REPORT_SYMBOLS:
        if name in r:
            rep[name] = r[name]
        elif name in out:
            rep[name] = out[name]
    return rep


def report_columns(rep, t):
    """The report as a DataFrame: index t labels, one column per (flattened) symbol."""
    cols = {}
    for name in REPORT_SYMBOLS:
        if name not in rep:
            continue
        v = rep[name]
        if v.ndim == 1:
            cols[name] = v
        else:
            for k, g in enumerate(_SUBSETS[name]):
                cols[f"{name}.{g}"] = v[:, k]
    df = pd.DataFrame(cols, index=pd.Index(t.labels, name="t"))
    return df


def report_stem(cfg):
    """<rname>_<case>_<ctag>, the report file stem."""
    return f"{cfg.rname or cfg.psce}_{cfg.case or '000000'}_{cfg.ctag}"


def write_report_csv(rep, inp, rout3):
    """Write <rout3>/<rname>_<case>_<ctag>.csv (rout3: a folder, with or without a trailing
    "/"); returns the path."""
    path = Path(rout3) / (report_stem(inp.cfg) + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    report_columns(rep, inp.t).to_csv(path, float_format="%.17g")
    return path


# ---------------------------------------------------------------- RCMIP3 template CSV

SCENARIOMIP_CONC = ("scen7-H", "scen7-HL", "scen7-M", "scen7-ML", "scen7-L", "scen7-VL",
                    "scen7-LN")        # exported as <name>C
# exported as if stored per step (yearly sum of the steps) although the registry has
# another time basis; deliberately kept identical to the SCM4OPT v4.1 reference
# submission files (doe_heatflux is a rate, see its registry description)
PER_STEP_COMPAT = frozenset(("doe_heatflux",))


def load_output_mapping(path=None, ref_compat=True):
    """output_mapping.csv rows with rpt_out == 1 (the reported outputs).

    ref_compat : the export of the reference submission files. False: the variables of
                 varregistry.KNOWN_RCMIP_ISSUES (ocean heat content and uptake) get the
                 scale implied by their registry and RCMIP units, and are no longer
                 summed per year like per-step amounts (PER_STEP_COMPAT): a rate is
                 exported as its annual mean.
    """
    m = pd.read_csv(path or data_path(DAT_DIR, "RCMIP3_src", "output_mapping.csv"))
    m = m[(m["rpt_out"] == 1) & m["rpt_var"].notna()].copy()
    m["rpt_var"] = m["rpt_var"].astype(str).str.strip()   # some entries carry trailing blanks
    m["rpt_var0"] = pd.to_numeric(m["rpt_var0"], errors="coerce").fillna(0.0)
    m["rpt_var_scale"] = pd.to_numeric(m["rpt_var_scale"], errors="coerce").fillna(1.0)
    compat = PER_STEP_COMPAT if ref_compat else frozenset()
    per_step = m["rpt_var"].isin(per_step_names() | compat)   # time basis: registry
    m["time_scale"] = np.where(per_step, 1 / DT, 1.0)   # DT = 1/6 yr per step
    m["aggregate_to_year"] = per_step
    if not ref_compat:
        reg = variables()
        for i in m.index[m["rpt_var"].isin(KNOWN_RCMIP_ISSUES)]:
            name = m.at[i, "rpt_var"]
            m.at[i, "rpt_var_scale"] = unit_factor(reg.at[name, "unit"],
                                                   str(m.at[i, "Unit"]).replace("MAGICC ", ""))
            m.at[i, "aggregate_to_year"] = reg.at[name, "time_basis"] != "state"   # annual mean
    return m


def template_columns(path=None):
    """Header of the 'your_data' sheet of the RCMIP3 submission template.

    Default: its header row saved as a CSV (..._columns.csv), so no Excel reader
    (openpyxl) is needed; an .xlsx / .xls path is read with pandas.read_excel.
    """
    path = Path(path or data_path(DAT_DIR, "RCMIP3_src",
                                  "rcmip3-data-submission-template-v1-1-0_columns.csv"))
    if path.suffix.lower() in (".xlsx", ".xls"):
        return [str(c) for c in pd.read_excel(path, sheet_name="your_data", nrows=0).columns]
    return [str(c) for c in pd.read_csv(path, nrows=0).columns]


def _annual(series, t, aggregate, stored_only=False):
    """Yearly values: mean over each year's steps, or the value at the whole-year step.

    stored_only : average only the non-zero steps. Deliberately kept identical to the
                  SCM4OPT v4.1 reference submission files, whose yearly means of report
                  parameters divide by the number of non-zero steps (e.g. 5 instead
                  of 6 in the year an emission starts).
    """
    years = np.floor(t.tval + 1e-6).astype(int)
    if aggregate:
        v = pd.Series(series, dtype=float)
        if stored_only:
            v = v.where(v != 0)
        s = v.groupby(years).mean()
    else:
        whole = np.abs(t.tval - np.round(t.tval)) < 1e-6
        s = pd.Series(series[whole], index=np.round(t.tval[whole]).astype(int))
    full = np.arange(years.min(), years.max() + 1)
    return s.reindex(full).fillna(0.0)


def rcmip_rows(rep, inp, mapping=None, template=None, climate_model=None, r_compat=None):
    """RCMIP3 submission rows for this run (one ensemble member), as a DataFrame.

    climate_model : the rows' climate_model; default --climate_model, else
                    version.RCMIP_CLIMATE_MODEL (the original SCM4OPT submits as SCM4OPTv4.0)
    r_compat : as the SCM4OPT v4.1 reference submission files (default: --ref_compat):
               yearly means of report parameters skip zero-valued steps, and parameters
               that are zero everywhere get no row; False averages all steps and keeps
               every row.
    The mapping defaults to load_output_mapping(ref_compat=--ref_compat).
    """
    cfg = inp.cfg
    r_compat = cfg.ref_compat if r_compat is None else r_compat
    m = load_output_mapping(ref_compat=cfg.ref_compat) if mapping is None else mapping
    cols = template_columns() if template is None else template
    ens = cols.index("ensemble_member")
    meta_cols, year_cols = cols[:ens + 1], cols[ens + 1:]
    climate_model = climate_model or cfg.climate_model or RCMIP_CLIMATE_MODEL
    sce = cfg.rname or cfg.psce
    scenario = f"{sce}C" if sce in SCENARIOMIP_CONC else sce
    member = str(int(cfg.case or "0"))
    rows = []
    for _, mp in m.iterrows():
        v = rep.get(mp["rpt_var"])
        if v is None or v.ndim != 1:
            continue
        if r_compat and mp["rpt_var"] in REPORT_PARAMETERS and not np.any(v != 0):
            continue
        s = _annual(v, inp.t, mp["aggregate_to_year"],
                    stored_only=r_compat and mp["rpt_var"] in REPORT_PARAMETERS)
        s = (s + mp["rpt_var0"]) * mp["rpt_var_scale"] * mp["time_scale"]
        row = dict(climate_model=climate_model, model="unspecified", scenario=scenario,
                   region="World", variable=mp["Variable"],
                   unit=str(mp["Unit"]).replace("MAGICC ", ""), ensemble_member=member)
        row.update({str(y): val for y, val in s.items() if str(y) in year_cols})
        rows.append(row)
    df = pd.DataFrame(rows)
    years = sorted(int(c) for c in df.columns if str(c).isdigit())
    keep = meta_cols + [str(y) for y in range(years[0], years[-1] + 1) if str(y) in df.columns]
    return df.reindex(columns=keep).fillna(0)


def write_rcmip_csv(rep, inp, out_dir, **kw):
    """Write SCM4OPTv40_rcmip_phase3_<scenario>_<case>.csv in out_dir; returns the path."""
    df = rcmip_rows(rep, inp, **kw)
    path = Path(out_dir) / f"{RCMIP_FILE_PREFIX}{df['scenario'].iloc[0]}_{inp.cfg.case or '000000'}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
