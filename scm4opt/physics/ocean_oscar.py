# SPDX-License-Identifier: GPL-3.0-or-later
"""Ocean carbon: OSCAR-type ocean carbon cycle (mixed-layer boxes).

The surface ocean is split into 8 boxes (fractions p_circ and timescales
tau_circ of the chosen box model) that exchange CO2 with the atmosphere and
export carbon to the deep ocean. Each model step is integrated with `nsub`
explicit Euler sub-steps (--ocn_nsub, default 4). Carbon is in GtC, fluxes in
GtC/step.

Order within step n (needs scm_D_sst(n), so the ocean runs AFTER DOECLIM):
  sub-step s=1   starts from the end state of n-1 and uses its outflows
  sub-step s>1   uses the outflows evaluated at sub-step s-1
  sea flux(n)    = -sum_s sum_b (FOUT_sub(n,b,s) - FIN(n-1,b)) * dt_sub
  FIN(n)         = from scm_D_CO2(n); used by the sub-steps of n+1

Reference behaviour (ref_compat, default; README: Reference compatibility): the sea flux sums FOUT over
sub-steps 1..nsub of n, while the box integration uses FOUT from the end of n-1 and
sub-steps 1..nsub-1, so ocean and atmosphere carbon are not exactly mass-consistent
(identical to the SCM4OPT v4.1 reference results). With --ref_compat=0 the sea flux
uses the outflows that drive the integration: flux = change of the surface boxes +
export to the deep ocean.

Options (config.OceanOptions, RunConfig.cm6mdl / rname):
  --cm6mdl=<ESM>           alpha_mld..sst_0 from ocean_gcm_osc_est.csv
  no --cm6mdl              surf_layer(v_osn_model) + tbl_OSNKtrans(mod_OSNKtrans)
  v_osn_model              box model: HILDA | BD-model | 2D-model | 3D-model
  mod_OSNKchem, --pco2_form, pco2_w_*   pCO2 emulator: Poly | Pade | Power, weighted
  --ocn_nsub               number of sub-steps
  --rname=abrupt-0p5xCO2   sst_0 = max(17.1, sst_0)
Derived values keep their reference formulas.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from ..core.config import DAT_DIR, OceanOptions, data_path
from ..core.constants import CO2_0, DT
from ..core.tables import pick
from ..core.errors import ModelBoundsError
from ..core.config import RunConfig
from .ocean_models import OceanDrivers

OBOX = tuple(f"ob{i}" for i in range(1, 9))
INF = np.inf

# =============================================================================
# Option tables, in the reference order
# =============================================================================

# Surface-layer parameters by box model: v_fg [1/yr], A_ocean [m2], mld_0 [m], sst_0 [degC]
SURF_LAYER = {
    "HILDA":    dict(gas_exchange_rate=1 / 9.06, area_ocean=3.62E14, mld_0=75, sst_0=18.2),
    "BD-model": dict(gas_exchange_rate=1 / 7.80, area_ocean=3.62E14, mld_0=75, sst_0=17.7),
    "2D-model": dict(gas_exchange_rate=1 / 7.46, area_ocean=3.54E14, mld_0=50, sst_0=18.3),
    "3D-model": dict(gas_exchange_rate=1 / 7.66, area_ocean=3.55E14, mld_0=50.9, sst_0=17.7),
}
# Mixed-layer response by transport model: alpha_mld, gamma_mld
TBL_OSNKTRANS = {
    "IPSL-CM5A-LR": (0.46600997, -0.10120396),
    "MPI-ESM-LR":   (0.4596155, -0.41559181),
    "CESM1-BGC":    (0.22717431, -0.18772954),
}
# Box fractions and timescales [yr] by box model
P_CIRC_TBL = {
    "HILDA":    (0.43196702, 0.24278, 0.13963, 0.089318, 0.03782, 0.035549, 0.022936, 0),
    "BD-model": (0.52443999, 0.16851, 0.11803, 0.076817, 0.050469, 0.010469, 0.031528, 0.019737),
    "2D-model": (0.815938, 0.06738, 0.036608, 0.026994, 0.026933, 0.012456, 0.013691, 0),
    "3D-model": (0, 0.61125302, 0.24966, 0.066485, 0.038344, 0.019439, 0.014819, 0),
}
TAU_CIRC_TBL = {
    "HILDA":    (0.33333334, 1.26789999, 5.25279999, 18.60099983, 68.73600006, 232.30000305, INF, INF),
    "BD-model": (0.33333334, 1.63880002, 4.87020016, 14.17199993, 43.50600052, 148.77000427, 215.71000671, INF),
    "2D-model": (0.5, 10.51500034, 11.67700005, 38.94599915, 107.56999969, 331.54000854, INF, INF),
    "3D-model": (INF, 0.70177001, 2.34879994, 15.28100014, 65.35900116, 347.54998779, INF, INF),
}

ALPHA_CO2 = 0.1765 * 12.0   # GtC per ppm inside the ocean module (NOT PPM2GTC)
ALPHA_SOL = 1.722e17        # {umol/kg * m3 / ppm}, Joos et al. (1996)
PREIND_SURFACE_C = 900.0    # GtC, IPCC AR6 Fig. 5.12
PREIND_INTERDEEP_C = 37100.0
FT = 1                      # fT, multiplies scm_D_sst




# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class OceanParams:
    """Ocean parameters, named as in the model; build with load_ocean_params()."""
    alpha_mld: float
    gamma_mld: float
    gas_exchange_rate: float            # gas-exchange coefficient [1/yr]
    area_ocean: float         # [m2]
    mld_0: float           # mixed-layer depth [m]
    sst_0: float           # preindustrial SST [degC]
    p_circ: np.ndarray     # (8,) box fractions
    tau_circ: np.ndarray   # (8,) box timescales [yr]; inf = no export
    w_poly: float          # ocn_pCO2_switch, normalised
    w_pade: float
    w_power: float
    n_substeps: int
    alpha_co2: float = ALPHA_CO2
    alpha_sol: float = ALPHA_SOL
    co2_0: float = CO2_0
    f_temp: float = FT
    dt: float = DT
    flux_bounds: bool = True   # check the scm_atmoSeaFlux bounds (off with --load_conc)
    ref_compat: bool = True    # sea flux as in the reference results (see the module docstring)

    @property
    def ocn_dt_sub(self):
        return self.dt / self.n_substeps

    @property
    def alpha_dic(self):
        return self.alpha_sol / self.alpha_co2 / self.area_ocean / self.mld_0

    @property
    def inv_tau(self):
        return 1.0 / self.tau_circ   # 1/inf = 0: no export

    # Reference DIC and emulator coefficients at sst_0
    @property
    def ocn_dic_0_pade(self):
        s, r = self.sst_0, self.co2_0 / 380
        return (30015.6 * (1 - 0.0226536 * (s - 15) + 0.000167105 * (s - 15)**2) * r
                / (1 + 13.4574 * (1 - 0.019829 * (s - 15) + 0.000113872 * (s - 15)**2) * r
                   - 0.243121 * (1 + 0.000443511 * (s - 15) - 0.000473227 * (s - 15)**2) * r**2))

    @property
    def ocn_dic_0_power(self):
        s = self.sst_0
        return (2160.156 * (1 - 0.00347063 * (s - 15) - 0.0000250016 * (s - 15)**2)
                * ((self.co2_0 / 380) - 0.318665 * (1 - 0.00151292 * (s - 15) - 0.000198978 * (s - 15)**2))
                ** (0.0595961 * (1 + 0.0200328 * (s - 15) + 0.000192084 * (s - 15)**2)))


def pco2_switch(o):
    """ocn_pCO2_switch(Poly, Pade, Power), normalised."""
    w = {"Poly": 0.0, "Pade": 0.0, "Power": 0.0}
    w[{"CO2SysPade": "Pade", "CO2SysPower": "Power"}.get(o.mod_OSNKchem, "Poly")] = 1.0
    if o.pco2_form is not None:
        if o.pco2_form not in w:
            raise ValueError("Unknown pco2_form. Use Poly, Pade or Power.")
        w = {k: float(k == o.pco2_form) for k in w}
    for key, val in (("Poly", o.pco2_w_poly), ("Pade", o.pco2_w_pade), ("Power", o.pco2_w_power)):
        if val is not None:
            w[key] = float(val)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Invalid pCO2 switch: sum of weights must be positive.")
    return w["Poly"] / total, w["Pade"] / total, w["Power"] / total


def load_ocean_params(cfg=None):
    """OSCAR ocean parameters of a run configuration."""
    cfg = cfg or RunConfig()
    return replace(_ocean_params(cfg.cm6mdl, cfg.ocean, cfg.dat_dir, cfg.rname),
                   flux_bounds=not cfg.load_conc, ref_compat=cfg.ref_compat)


def _ocean_params(cm6mdl, opts=None, dat_dir=DAT_DIR, rname=None):
    """Ocean parameters for one configuration.

    cm6mdl : CMIP6 model (row of ocean_gcm_osc_est.csv), or None for the
             surf_layer / tbl_OSNKtrans defaults.
    """
    o = opts or OceanOptions()
    if cm6mdl is not None:
        df = pd.read_csv(data_path(dat_dir, "CMIP6", "ocean_gcm_osc_est.csv"), index_col=0)
        if cm6mdl not in df.index:
            raise ValueError(f"unknown --cm6mdl={cm6mdl!r}; options: {list(df.index)}")
        row = df.loc[cm6mdl]
        cols = {"alpha_mld": "alpha_mld", "gamma_mld": "gamma_mld", "gas_exchange_rate": "v_fg",
                "area_ocean": "A_ocean", "mld_0": "mld_0", "sst_0": "sst_0"}   # field: CSV column
        kw = {k: float(row[c]) for k, c in cols.items()}
    else:
        alpha_mld, gamma_mld = pick(TBL_OSNKTRANS, o.mod_OSNKtrans, "mod_OSNKtrans")
        kw = dict(alpha_mld=alpha_mld, gamma_mld=gamma_mld,
                  **pick(SURF_LAYER, o.v_osn_model, "v_osn_model"))
    if rname == "abrupt-0p5xCO2":
        kw["sst_0"] = max(17.1, kw["sst_0"])
    w_poly, w_pade, w_power = pco2_switch(o)
    return OceanParams(
        **kw,
        p_circ=np.array(pick(P_CIRC_TBL, o.v_osn_model, "v_osn_model"), dtype=float),
        tau_circ=np.array(TAU_CIRC_TBL[o.v_osn_model], dtype=float),
        w_poly=w_poly, w_pade=w_pade, w_power=w_power, n_substeps=int(o.ocn_nsub))


# =============================================================================
# Carbonate chemistry and DIC
# =============================================================================

def f_dic(p, D_CSURF, D_mld):
    """DIC anomaly [umol/kg] from surface carbon [GtC] (f_dic macro)."""
    return p.alpha_dic * D_CSURF / (1 + D_mld / p.mld_0)


def dic_attribution(p, D_CSURF, D_mld):
    """dic_df_1 (carbon), dic_df_2 (mixed-layer depth), dic_df_tot, df_dic1, df_dic2."""
    d1 = p.alpha_dic / (1 + D_mld / p.mld_0) * D_CSURF
    d2 = -p.alpha_dic * D_CSURF / p.mld_0 / (1 + D_mld / p.mld_0)**2 * D_mld
    tot = d1 + d2
    return d1, d2, tot, d1 / tot, d2 / tot


def _pade_coeffs(sst):
    """ocn_a0/a1/a2 at an SST [degC]."""
    x = sst - 15
    return (30015.6 * (1 - 0.0226536 * x + 0.000167105 * x**2),
            13.4574 * (1 - 0.019829 * x + 0.000113872 * x**2),
            -0.243121 * (1 + 0.000443511 * x - 0.000473227 * x**2))


def _power_coeffs(sst):
    """ocn_p0/p1/p2 at an SST [degC]."""
    x = sst - 15
    return (2160.156 * (1 - 0.00347063 * x - 0.0000250016 * x**2),
            0.0595961 * (1 + 0.0200328 * x + 0.000192084 * x**2),
            0.318665 * (1 - 0.00151292 * x - 0.000198978 * x**2))


def f_pCO2_pade(p, D_dic, D_sst):
    """CO2Sys Pade approximant (f_pCO2_pade macro), pCO2 anomaly [ppm]."""
    a0, a1, a2 = _pade_coeffs(p.sst_0 + D_sst)
    a0_0, a1_0, a2_0 = _pade_coeffs(p.sst_0)
    d0 = p.ocn_dic_0_pade
    d = d0 + D_dic
    return (380 * (a0 - a1 * d - np.sqrt((a0 - a1 * d)**2 - 4 * a2 * d**2)) / (2 * a2 * d)
            - 380 * (a0_0 - a1_0 * d0 - np.sqrt((a0_0 - a1_0 * d0)**2 - 4 * a2_0 * d0**2)) / (2 * a2_0 * d0))


def f_pCO2_power(p, D_dic, D_sst):
    """CO2Sys power law (f_pCO2_power macro), pCO2 anomaly [ppm]."""
    p0, p1, p2 = _power_coeffs(p.sst_0 + D_sst)
    p0_0, p1_0, p2_0 = _power_coeffs(p.sst_0)
    d0 = p.ocn_dic_0_power
    return (380 * (p2 + ((d0 + D_dic) / p0) ** (1 / p1))
            - 380 * (p2_0 + (d0 / p0_0) ** (1 / p1_0)))


def f_pCO2_poly(p, D_dic, D_sst):
    """Joos (2001) polynomial used in OSCARv3 (f_pCO2_poly macro), pCO2 anomaly [ppm]."""
    s = p.sst_0
    return ((p.co2_0 + ((1.5568 - 1.3993E-2 * s) * D_dic
                        + (7.4706 - 0.20207 * s) * 1E-3 * D_dic**2
                        - (1.2748 - 0.12015 * s) * 1E-5 * D_dic**3
                        + (2.4491 - 0.12639 * s) * 1E-7 * D_dic**4
                        - (1.5468 - 0.15326 * s) * 1E-10 * D_dic**5))
            * np.exp(0.0423 * D_sst) - p.co2_0)


def pco2_anomaly(p, D_dic, D_sst):
    """Weighted pCO2 emulator, as in ocn_D_pCO2_sub_EQ (zero-weight forms are skipped)."""
    out = 0.0
    if p.w_poly:
        out += p.w_poly * f_pCO2_poly(p, D_dic, D_sst)
    if p.w_pade:
        out += p.w_pade * f_pCO2_pade(p, D_dic, D_sst)
    if p.w_power:
        out += p.w_power * f_pCO2_power(p, D_dic, D_sst)
    return out


# =============================================================================
# Time stepping
# =============================================================================

@dataclass
class OceanStep:
    """One step of the ocean module, named as the ocn_D_* / ocn_* variables."""
    D_mld: float
    D_FIN: np.ndarray        # (8,)
    D_FOUT: np.ndarray       # (8,)  last sub-step
    D_FCIRC: np.ndarray      # (8,)  last sub-step
    D_CSURF: np.ndarray      # (8,)  last sub-step
    D_CSURF_all: float
    D_dic: float
    D_pCO2: float
    D_dic_sub: np.ndarray    # (nsub,)
    D_pCO2_sub: np.ndarray   # (nsub,)
    D_FOUT_sub: np.ndarray   # (8, nsub)
    D_FCIRC_sub: np.ndarray  # (8, nsub)
    D_CSURF_sub: np.ndarray  # (8, nsub)
    atmoSeaFlux: float
    surface_cpool: float
    total_cpool: float


def ocean_step(p, prev, drivers):
    """One step of the ocean carbon module.

    prev    : OceanStep of step n-1, or None at the first step (1850)
    drivers : OceanDrivers; uses d_co2 = scm_D_CO2(n) [ppm] and d_sst = scm_D_sst(n) [K]
              (DOECLIM at the same step). p.flux_bounds: check the scm_atmoSeaFlux bounds.
    Returns the OceanStep of step n; its atmoSeaFlux feeds the atmosphere at n+1.
    """
    d_co2, d_sst, flux_bounds = drivers.d_co2, drivers.d_sst, p.flux_bounds
    nb, ns = len(p.p_circ), p.n_substeps
    exch = p.p_circ * p.gas_exchange_rate * p.alpha_co2          # (8,) uptake per ppm
    sst = p.f_temp * d_sst
    d_mld = p.mld_0 * p.alpha_mld * (np.exp(p.gamma_mld * sst) - 1)
    fin = exch * d_co2

    dic_sub = np.zeros(ns)
    pco2_sub = np.zeros(ns)
    fout_sub = np.zeros((nb, ns))
    fcirc_sub = np.zeros((nb, ns))
    csurf_sub = np.zeros((nb, ns))
    flux = 0.0

    if prev is not None:  # all sub-step variables are 0 at the first step
        # outflows at the end of n-1 drive the first sub-step
        c, fout, fcirc = prev.D_CSURF, exch * prev.D_pCO2, prev.D_CSURF * p.inv_tau
        fout_drive = 0.0                    # sum of the outflows driving the sub-steps
        for s in range(ns):
            fout_drive = fout_drive + fout
            c = c + p.ocn_dt_sub * (prev.D_FIN - fout - fcirc)
            csurf_sub[:, s] = c
            dic_sub[s] = f_dic(p, c.sum(), d_mld)
            pco2_sub[s] = pco2_anomaly(p, dic_sub[s], sst)
            fout = exch * pco2_sub[s]
            fcirc = c * p.inv_tau
            fout_sub[:, s] = fout
            fcirc_sub[:, s] = fcirc
        if p.ref_compat:
            flux = -((fout_sub - prev.D_FIN[:, None]).sum() * p.ocn_dt_sub)
        else:                               # mass-consistent with the box integration
            flux = -((fout_drive - ns * prev.D_FIN).sum() * p.ocn_dt_sub)

    surface = PREIND_SURFACE_C if prev is None else prev.surface_cpool + flux
    step = OceanStep(
        D_mld=d_mld, D_FIN=fin,
        D_FOUT=fout_sub[:, -1].copy(), D_FCIRC=fcirc_sub[:, -1].copy(),
        D_CSURF=csurf_sub[:, -1].copy(), D_CSURF_all=float(csurf_sub[:, -1].sum()),
        D_dic=dic_sub[-1], D_pCO2=pco2_sub[-1],
        D_dic_sub=dic_sub, D_pCO2_sub=pco2_sub,
        D_FOUT_sub=fout_sub, D_FCIRC_sub=fcirc_sub, D_CSURF_sub=csurf_sub,
        atmoSeaFlux=flux, surface_cpool=surface,
        total_cpool=surface + PREIND_INTERDEEP_C)
    _check_bounds(step, flux_bounds)
    return step


def _check_bounds(s, flux_bounds=True):
    """Bounds: ocn_D_dic in [-1000, 5000] (for ssp1..5 too: they replace the narrower
    [-100, 2000]); scm_atmoSeaFlux in [-100, 100]."""
    if not -1000 <= s.D_dic <= 5000:
        raise ModelBoundsError("ocn_D_dic", s.D_dic, "[-1000, 5000]")
    if flux_bounds and not -100 <= s.atmoSeaFlux <= 100:
        raise ModelBoundsError("ocn_atmoSeaFlux", s.atmoSeaFlux, "[-100, 100]")


def collect_ocean(steps):
    """OceanStep list -> {ocn_<field>: array over t} (per-box (N, 8), sub-steps (N, ..., nsub))."""
    return {"ocn_" + f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(OceanStep)}


def run_ocean(p, d_co2, d_sst):
    """Run the ocean offline over prescribed scm_D_CO2 and scm_D_sst series.

    Returns {variable name without the ocn_ prefix: array}; per-box variables have
    shape (N, 8) and sub-step ones (N, nsub) or (N, 8, nsub).
    """
    steps = []
    prev = None
    for i in range(len(d_co2)):
        prev = ocean_step(p, prev, OceanDrivers(d_co2=d_co2[i], d_sst=d_sst[i]))
        steps.append(prev)
    return {f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(OceanStep)}
