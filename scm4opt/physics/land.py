# SPDX-License-Identifier: GPL-3.0-or-later
"""Land carbon cycle: plant, detritus and soil pools with CO2 and temperature feedbacks.

Three pools (plant P, detritus H, soil S) in GtC, MAGICC-style. All fluxes are
GtC per model step. Within step n:

  1. pools(n) from pools(n-1) and the rates of step n-1  (trapezoidal update)
  2. rates(n) from CO2(n), deltem(n) and land-use CO2 emissions(n)
  3. decay(n) = 2 * th(n) * pools(n)
  4. land flux(n) = sum of pool changes; positive = uptake from the atmosphere

Land feels temperature only through deltem(n) = 2*tatm(n-1) - tatm(n-2),
so a step never needs temperature at n.

Options (--name=value flags):
  --cm6mdl=<ESM>          parameters from land_gcm_est.csv, feedbacks x 0.7
  no --cm6mdl             MAGICC6 defaults
                          (cm6mdl=None; the full model always sets a default)
  --scen=tcre / tcre2     derived values recomputed from the raw parameters
  --rname=abrupt-0p5xCO2  sigmoid-only CO2 fertilisation
  --use_ncycle            NPP x nit_npp_scaler(t): LandDrivers.npp_scaler of land_step
Not included: the --cali_lnd calibration branch.
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from ..core.config import DAT_DIR, data_path
from ..core.constants import CO2_CONC_1850, CO2_NPP_ZERO_CONC, DT
from ..core.errors import ModelBoundsError
from ..core.config import RunConfig

# Result names, the model's lnd_* variables without the prefix
OUTPUTS = (
    "fert_log", "fert_r", "fert_b", "fert_sig", "fert_fact",
    "fg", "fr", "fh", "fs",
    "npp", "nppP", "nppH", "nppS", "resp",
    "dpElu", "dhElu", "dsElu", "sumPElu", "sumHElu", "sumSElu", "sumElu",
    "tauP", "tauH", "tauS", "regrowP", "regrowH", "regrowS",
    "grossDP", "grossDH", "grossDS", "grossD",
    "thp", "thh", "ths", "dcyP", "dcyH", "dcyS",
    "rh", "rhh", "h2s", "p2h", "p2s",
    "plnt_pool", "detr_pool", "soil_pool", "terrest_pool", "atmoLndFlux",
)

# Parameter name (land_gcm_est.csv column) -> LandParams field, raw (input) parameters
RAW = {
    "CO2_NPP_INITIAL": "npp_0_per_step",
    "CO2_RESPIRATION_INITIAL": "resp_0_per_step",
    "CO2_FERTILIZATION_SCALE": "fert_scale",
    "CO2_FERTILIZATION_FACTOR": "beta",
    "CO2_FERTILIZATION_METHOD": "fert_method",
    "CO2_FEEDBACKFACTOR_NPP": "k_npp",
    "CO2_FEEDBACKFACTOR_RESPIRATION": "k_resp",
    "CO2_FEEDBACKFACTOR_DETRITUS": "k_detr",
    "CO2_FEEDBACKFACTOR_SOIL": "k_soil",
    "CO2_PLANTPOOL_INITIAL": "pool_0_p",
    "CO2_DETRITUSPOOL_INITIAL": "pool_0_h",
    "CO2_SOILPOOL_INITIAL": "pool_0_s",
    "CO2_GIFFORD_CONC_FOR_ZERONPP": "gifford_conc",
    "CO2_NORGRWTH_FRAC_DEFO": "no_regrowth_frac",
    "CO2_FRACTION_NPP_2_PLANT": "f_npp_to_p",
    "CO2_FRACTION_NPP_2_DETRITUS": "f_npp_to_h",
    "CO2_FRACTION_PLANT_2_DETRITUS": "f_p_to_h",
    "CO2_FRACTION_DEFOREST_PLANT": "f_defo_p",
    "CO2_FRACTION_DEFOREST_DETRITUS": "f_defo_h",
    "CO2_FRACTION_DETRITUS_2_SOIL": "f_h_to_s",
    "CO2_PLNT_REGROW_INTER": "regrow_inter_p",
    "CO2_PLNT_REGROW_GRADI": "regrow_gradi_p",
    "CO2_DETR_REGROW_INTER": "regrow_inter_h",
    "CO2_DETR_REGROW_GRADI": "regrow_gradi_h",
    "CO2_SOIL_REGROW_INTER": "regrow_inter_s",
    "CO2_SOIL_REGROW_GRADI": "regrow_gradi_s",
}
# Parameter name -> LandParams field, for the derived parameters
DERIVED = {
    "CO2_FRACTION_NPP_2_SOIL": "f_npp_to_s",
    "CO2_FRACTION_PLANT_2_SOIL": "f_p_to_s",
    "CO2_FRACTION_DEFOREST_SOIL": "f_defo_s",
    "CO2_PLNT_DECAY_INITIAL": "decay_0_p",
    "CO2_DETR_DECAY_INITIAL": "decay_0_h",
    "CO2_SOIL_DECAY_INITIAL": "decay_0_s",
}


@dataclass(frozen=True)
class LandParams:
    """Land parameters; parameter names in RAW / DERIVED. Build with load_land_params()."""
    npp_0_per_step: float          # per step
    resp_0_per_step: float         # per step
    fert_scale: float
    beta: float
    fert_method: float   # 1 = log, 2 = sigmoid, in between = blend
    k_npp: float
    k_resp: float
    k_detr: float
    k_soil: float
    pool_0_p: float
    pool_0_h: float
    pool_0_s: float
    gifford_conc: float  # loaded but unused by the equations (they use CO2_NPP_ZERO_CONC)
    no_regrowth_frac: float
    f_npp_to_p: float
    f_npp_to_h: float
    f_p_to_h: float
    f_defo_p: float
    f_defo_h: float
    f_h_to_s: float
    regrow_inter_p: float
    regrow_gradi_p: float
    regrow_inter_h: float
    regrow_gradi_h: float
    regrow_inter_s: float
    regrow_gradi_s: float
    # derived (see _derive)
    f_npp_to_s: float
    f_p_to_s: float
    f_defo_s: float
    decay_0_p: float
    decay_0_h: float
    decay_0_s: float
    # options
    fert_sigmoid_only: bool = False   # --rname=abrupt-0p5xCO2
    co2_ref: float = CO2_CONC_1850
    co2_npp_zero: float = CO2_NPP_ZERO_CONC


def _derive(raw, form):
    """Derived land parameters, with the formulas of the configuration in use.

    form="tau1"   --cm6mdl, and --scen=tcre/tcre2, via the
                  assistant parameters CO2_FRACTION_*_TAU_1 = decay / pool
    form="tau"    MAGICC6 defaults, via
                  CO2_FRACTION_*_TAU = pool / decay
    Both give the same values up to rounding.
    """
    r = raw
    d = dict(f_npp_to_s=1 - r["f_npp_to_p"] - r["f_npp_to_h"],          # CO2_FRACTION_NPP_2_SOIL
             f_p_to_s=1 - r["f_p_to_h"],                              # CO2_FRACTION_PLANT_2_SOIL
             f_defo_s=1 - r["f_defo_p"] - r["f_defo_h"])           # CO2_FRACTION_DEFOREST_SOIL
    npp0, resp0 = r["npp_0_per_step"], r["resp_0_per_step"]
    if form == "tau1":
        # CO2_FRACTION_DETRITUS_TAU_1 (PLANT/SOIL_TAU_1 are not needed)
        detr_tau_1 = ((r["f_npp_to_h"] * npp0 + r["f_p_to_h"] * (r["f_npp_to_p"] * npp0 - resp0))
                      / r["pool_0_h"])
        d["decay_0_p"] = r["f_npp_to_p"] * npp0 - resp0
        d["decay_0_h"] = r["f_npp_to_h"] * npp0 + r["f_p_to_h"] * d["decay_0_p"]
        d["decay_0_s"] = npp0 - resp0 - (1 - r["f_h_to_s"]) * r["pool_0_h"] * detr_tau_1
    elif form == "tau":
        # CO2_FRACTION_DETRITUS_TAU (PLANT/SOIL_TAU are not needed)
        detr_tau = r["pool_0_h"] / (r["f_npp_to_h"] * npp0
                                   + r["f_p_to_h"] * (r["f_npp_to_p"] * npp0 - resp0))
        d["decay_0_p"] = r["f_npp_to_p"] * npp0 - resp0
        d["decay_0_h"] = r["f_npp_to_h"] * npp0 + r["f_p_to_h"] * d["decay_0_p"]
        d["decay_0_s"] = npp0 - resp0 - (1 - r["f_h_to_s"]) * r["pool_0_h"] / detr_tau
    else:
        raise ValueError(form)
    return d


# MAGICC6 defaults, used when --cm6mdl is not set (annual values)
# "MAGICC6 defaults" mode (no --cm6mdl): the MAGICC6 land parameters are read from
# data/MAGICC6/land_c4mip_bern.csv (values of the configuration files distributed with the
# MAGICC6 executable; licensed CC BY-NC-SA 3.0 by its authors, see data/MAGICC6/NOTICE.txt).
# SCM4OPT's own values of this mode are kept here.
MAGICC6_LAND_CSV = ("MAGICC6", "land_c4mip_bern.csv")
SCM4OPT_LAND_DEFAULTS = {
    "beta": 0.62295908638692,      # CO2 fertilisation factor of SCM4OPT (MAGICC6 C4MIP-BERN: 0.6485981)
    "fert_scale": 1.0,
}


def magicc6_land_defaults(dat_dir=DAT_DIR):
    """The land parameters of the MAGICC6-defaults mode (MAGICC6 file + SCM4OPT values)."""
    tab = pd.read_csv(data_path(dat_dir, *MAGICC6_LAND_CSV), float_precision="round_trip")
    raw = {r.parameter: float(r.value) for r in tab.itertuples()}
    raw.update(SCM4OPT_LAND_DEFAULTS)
    return raw

_FEEDBACK_SCALE = 0.7  # CMIP6 branch only


def load_land_params(cfg=None):
    """Land parameters of a run configuration."""
    cfg = cfg or RunConfig()
    return _land_params(cfg.cm6mdl, cfg.dat_dir, cfg.scen, cfg.rname)


def _land_params(cm6mdl, dat_dir=DAT_DIR, scen="contrib", rname=None, dt=DT):
    """Land parameters for one configuration.

    cm6mdl : CMIP6 model name (row of land_gcm_est.csv), or None for the MAGICC6
             defaults.
    """
    if cm6mdl is None:
        raw = magicc6_land_defaults(dat_dir)
        # MAGICC6 defaults: per-step NPP/respiration, log fertilisation,
        # re-fitted regrowth coefficients (SCM4OPT)
        raw["npp_0_per_step"] = raw["npp_0_per_step"] * dt
        raw["resp_0_per_step"] = raw["resp_0_per_step"] * dt
        raw["fert_method"] = 1
        raw.update(regrow_inter_p=2.232639577, regrow_gradi_p=-0.007118357,
                   regrow_inter_h=-0.042113268, regrow_gradi_h=0.003201409,
                   regrow_inter_s=1.057490e-01, regrow_gradi_s=-8.746865e-05)
        form = "tau1" if scen in ("tcre", "tcre2") else "tau"
    else:
        df = pd.read_csv(data_path(dat_dir, "CMIP6", "land_gcm_est.csv"), index_col=0)
        if cm6mdl not in df.index:
            raise ValueError(f"unknown --cm6mdl={cm6mdl!r}; options: {list(df.index)}")
        row = df.loc[cm6mdl]
        raw = {field: float(row[gname]) for gname, field in RAW.items()}
        for k in ("k_npp", "k_resp", "k_detr", "k_soil"):
            raw[k] = raw[k] * _FEEDBACK_SCALE
        form = "tau1"
    return LandParams(**raw, **_derive(raw, form),
                      fert_sigmoid_only=(rname == "abrupt-0p5xCO2"))


def deltem(tatm_m1, tatm_m2, n, deltT1850=0.0):
    """scm_deltem(n): linear extrapolation of tatm from n-1 and n-2; 0 for n < 2."""
    return 0.0 if n < 2 else 2 * tatm_m1 - tatm_m2 - deltT1850


@dataclass
class LandRates:
    """Everything the land module computes at one step, named as the lnd_* variables."""
    fert_log: float = 0.0
    fert_r: float = 0.0
    fert_b: float = 0.0
    fert_sig: float = 0.0
    fert_fact: float = 0.0
    fg: float = 0.0
    fr: float = 0.0
    fh: float = 0.0
    fs: float = 0.0
    npp: float = 0.0
    nppP: float = 0.0
    nppH: float = 0.0
    nppS: float = 0.0
    resp: float = 0.0
    dpElu: float = 0.0
    dhElu: float = 0.0
    dsElu: float = 0.0
    sumPElu: float = 0.0
    sumHElu: float = 0.0
    sumSElu: float = 0.0
    sumElu: float = 0.0
    tauP: float = 0.0
    tauH: float = 0.0
    tauS: float = 0.0
    regrowP: float = 0.0
    regrowH: float = 0.0
    regrowS: float = 0.0
    grossDP: float = 0.0
    grossDH: float = 0.0
    grossDS: float = 0.0
    grossD: float = 0.0
    thp: float = 0.0
    thh: float = 0.0
    ths: float = 0.0
    dcyP: float = 0.0
    dcyH: float = 0.0
    dcyS: float = 0.0
    rh: float = 0.0
    rhh: float = 0.0
    h2s: float = 0.0
    p2h: float = 0.0
    p2s: float = 0.0
    plnt_pool: float = 0.0
    detr_pool: float = 0.0
    soil_pool: float = 0.0
    terrest_pool: float = 0.0
    atmoLndFlux: float = 0.0


@dataclass(frozen=True)
class LandDrivers:
    """Inputs of one land step n."""
    co2: float                       # scm_atmoCO2(n) [ppm]
    deltem: float                    # scm_deltem(n) [K]
    elnd_co2: float                  # dat_elnd(n,"co2") [GtC/step]
    npp_scaler: float | None = None  # nit_npp_scaler(n) (--use_ncycle), or None


def land_step(p, prev, drivers):
    """One step of the land carbon cycle.

    p          : LandParams
    prev       : LandRates of step n-1, or None at the first step (1850)
    drivers    : LandDrivers of step n:
    co2        : scm_atmoCO2(n) [ppm]
    dtem       : scm_deltem(n) [K]
    elnd_co2   : dat_elnd(n,"co2") [GtC/step]
    npp_scaler : nit_npp_scaler(n) from the N cycle (--use_ncycle), or None
    Returns the LandRates of step n; its atmoLndFlux feeds the atmosphere at n+1.
    """
    co2, dtem, elnd_co2, npp_scaler = drivers.co2, drivers.deltem, drivers.elnd_co2, drivers.npp_scaler
    r = LandRates()

    # Pools (lnd_PLNTEQ / DETREQ / SOILEQ): right-hand side entirely at n-1
    if prev is None:
        r.plnt_pool, r.detr_pool, r.soil_pool = p.pool_0_p, p.pool_0_h, p.pool_0_s
    else:
        q = prev
        r.plnt_pool = ((q.plnt_pool * (1 - q.thp) + q.nppP - q.resp - q.grossDP)
                       / (1 + q.thp))
        r.detr_pool = ((q.detr_pool * (1 - q.thh) + q.nppH + q.dcyP * p.f_p_to_h - q.grossDH)
                       / (1 + q.thh))
        r.soil_pool = ((q.soil_pool * (1 - q.ths) + q.nppS + q.dcyP * p.f_p_to_s
                        + q.dcyH * p.f_h_to_s - q.grossDS)
                       / (1 + q.ths))

    # CO2 fertilisation (lnd_FERT*EQ); the sigmoid form is always evaluated
    r.fert_log = 1 + p.beta * np.log(co2 / p.co2_ref)
    r.fert_r = ((1. + p.beta * np.log(680. / p.co2_ref))
                / (1. + p.beta * np.log(340. / p.co2_ref)))
    z = p.co2_npp_zero
    r.fert_b = (((680 - z) - r.fert_r * (340 - z))
                / ((r.fert_r - 1) * (680 - z) * (340 - z)))
    r.fert_sig = ((1 / (p.co2_ref - z) + r.fert_b)
                  / (1 / (co2 - z) + r.fert_b))
    if p.fert_sigmoid_only:   # --rname=abrupt-0p5xCO2
        r.fert_fact = r.fert_sig / p.fert_scale
    else:
        r.fert_fact = (((2 - p.fert_method) * r.fert_log + (p.fert_method - 1) * r.fert_sig)
                       / p.fert_scale)

    # Temperature feedbacks (lnd_FGEQ ... lnd_FSEQ)
    r.fg = np.exp(p.k_npp * dtem)
    r.fr = np.exp(p.k_resp * dtem)
    r.fh = np.exp(p.k_detr * dtem)
    r.fs = np.exp(p.k_soil * dtem)

    # NPP and autotrophic respiration
    r.npp = p.npp_0_per_step * r.fert_fact * r.fg
    if npp_scaler is not None:   # --use_ncycle: lnd_nppEQ1 multiplies by nit_npp_scaler(t)
        r.npp = r.npp * npp_scaler
    r.nppP = r.npp * p.f_npp_to_p
    r.nppH = r.npp * p.f_npp_to_h
    r.nppS = r.npp * p.f_npp_to_s
    r.resp = p.resp_0_per_step * r.fert_fact * r.fr

    # Land-use emissions: allocation and running sum INCLUDING step n
    r.dpElu = p.f_defo_p * elnd_co2
    r.dhElu = p.f_defo_h * elnd_co2
    r.dsElu = p.f_defo_s * elnd_co2
    base = prev if prev is not None else LandRates()
    r.sumPElu = base.sumPElu + r.dpElu
    r.sumHElu = base.sumHElu + r.dhElu
    r.sumSElu = base.sumSElu + r.dsElu
    r.sumElu = r.sumPElu + r.sumHElu + r.sumSElu

    # Turnover times, regrowth and gross deforestation
    r.tauP = (p.pool_0_p - p.no_regrowth_frac * r.sumPElu) / p.decay_0_p
    r.tauH = (p.pool_0_h - p.no_regrowth_frac * r.sumHElu) / p.decay_0_h
    r.tauS = (p.pool_0_s - p.no_regrowth_frac * r.sumSElu) / p.decay_0_s
    r.regrowP = p.regrow_inter_p + p.regrow_gradi_p * r.tauP
    r.regrowH = p.regrow_inter_h + p.regrow_gradi_h * r.tauH
    r.regrowS = p.regrow_inter_s + p.regrow_gradi_s * r.tauS
    r.grossDP = r.dpElu + r.regrowP
    r.grossDH = r.dhElu + r.regrowH
    r.grossDS = r.dsElu + r.regrowS
    r.grossD = r.grossDP + r.grossDH + r.grossDS

    # Decay coefficients and rates
    r.thp = 0.5 / r.tauP
    r.thh = 0.5 / r.tauH * r.fh
    r.ths = 0.5 / r.tauS * r.fs
    r.dcyP = r.thp * 2 * r.plnt_pool
    r.dcyH = r.thh * 2 * r.detr_pool
    r.dcyS = r.ths * 2 * r.soil_pool

    # Diagnostics
    r.rh = r.resp + r.dcyH * (1 - p.f_h_to_s) + r.dcyS
    r.rhh = r.dcyH * (1 - p.f_h_to_s)
    r.h2s = r.dcyH * p.f_h_to_s
    r.p2h = r.dcyP * p.f_p_to_h
    r.p2s = base.dcyP * p.f_p_to_s  # lnd_P2SEQ uses dcyP(t-1); 0 at the first step

    r.terrest_pool = r.plnt_pool + r.detr_pool + r.soil_pool
    r.atmoLndFlux = 0.0 if prev is None else r.terrest_pool - prev.terrest_pool

    _check_bounds(r)
    return r


def _check_bounds(r):
    """Admissible ranges of the land variables. If one binds, the model is no longer a
    plain recursion and a time-stepped result would silently differ."""
    for name in ("tauP", "tauH", "tauS"):
        if getattr(r, name) < 1e-3:
            raise ModelBoundsError(f"lnd_{name}", getattr(r, name), ">= 1e-3")
    for name in ("plnt_pool", "detr_pool", "soil_pool"):
        if not 5 <= getattr(r, name) <= 1e8:
            raise ModelBoundsError(f"lnd_{name}", getattr(r, name), "[5, 1e8]")


def run_land(p, co2, tatm, elnd_co2, npp_scaler=None):
    """Run the land module offline over prescribed series (all length N).

    co2 : scm_atmoCO2 [ppm]; tatm : scm_tatm [K]; elnd_co2 : dat_elnd(t,"co2");
    npp_scaler : nit_npp_scaler series (--use_ncycle) or None.
    Returns {variable name without the lnd_ prefix: array}, plus "deltem".
    """
    n = len(co2)
    out = {name: np.empty(n) for name in OUTPUTS}
    out["deltem"] = np.empty(n)
    prev = None
    for i in range(n):
        dtem = deltem(tatm[i - 1] if i >= 1 else 0.0, tatm[i - 2] if i >= 2 else 0.0, i)
        prev = land_step(p, prev, LandDrivers(co2[i], dtem, elnd_co2[i],
                                              None if npp_scaler is None else npp_scaler[i]))
        out["deltem"][i] = dtem
        for name in OUTPUTS:
            out[name][i] = getattr(prev, name)
    return out


assert set(OUTPUTS) == {f.name for f in fields(LandRates)}
