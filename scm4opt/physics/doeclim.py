# SPDX-License-Identifier: GPL-3.0-or-later
"""DOECLIM temperature response: two-box energy balance with a diffusive ocean.

Two-box energy balance (land-air TL, sea-surface TS) coupled to a diffusive
ocean (Kriegler 2005; Tanaka & Kriegler 2007). The Crank-Nicolson system
B*T(n) = Q(n) + A*T(n-1) + DPAST(n) is solved with the precomputed inverse of
B, so each step is an explicit update once the forcing of step n is known.
DPAST2(n) is the ocean-diffusion memory: a convolution of the whole TS history
with the kernel doe_Ker.

Lagged (t-1) terms are 0 at the first step: Q(-1) = TL(-1) = TS(-1) = 0, so
the 1850 forcing acts as a step from zero.

Options (--name=value flags, RunConfig):
  --cm6mdl=<ESM>          parameters from doe_gcm_est_etminan.csv / _myhre.csv
                          (--ch4rfc) or _ssp534.csv (--use_ssp534)
  no --cm6mdl             hard-coded defaults
  --scen=ssp1..ssp5       hard-coded defaults, not the CSV
  --clim_sensi=<ECS>      overrides doe_t2co (ssp1..5 and --cm6mdl branches)
  --scen=tcre / tcre2     fixed parameter set
  --co2_only              DOECLIM forced by rfc_CO2 instead of erf_tot
Derived values are computed from the raw parameters.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..core.config import DAT_DIR, data_path
from ..core.constants import DT
from ..core.errors import ModelBoundsError
from ..core.config import RunConfig, n_steps

DOE_PI = 3.14159265358979323846264338327950288

# Hard-coded DOECLIM parameters
DEFAULTS = dict(
    land_sea_hx_slope=0.31,             # slope in climate feedback - land-sea heat exchange linear relationship
    land_sea_hx_offset=1.59,             # offset in that relationship, W/m2/K
    heatcap_seawater=0.13,            # specific heat capacity of seawater, W*yr/m3/K
    area_earth=5100656E8,  # m2
    diffusivity_unit_conv=3155.8,         # conversion from cm2/s to m2/yr
    rf_2xco2=3.7,            # radiative forcing for CO2 doubling, W/m2
    land_sens_ratio=1.43,           # land/sea-surface climate sensitivity ratio
    secs_per_year=31556926.0,
    depth_bottom=4000.0,         # bottom depth of diffusive ocean, m
    sst_air_factor=1.3,             # marine surface air warming over SST (sea-ice retreat)
    heatcap_land=0.52,            # heat capacity of land-troposphere system, W*yr/m2/K
    heatcap_mixed=7.80,            # heat capacity of mixed layer-troposphere system, W*yr/m2/K
    frac_land=0.29,           # fractional land area
    frac_ocean_deep=0.95,            # ocean fractional area below 60m
    climate_sensitivity=3.0,            # equilibrium climate sensitivity, K
    diffusivity=0.55,          # ocean vertical diffusivity, cm2/s
    tas_0=0.0,
    sst_0=0.0,
)
# --scen=tcre / tcre2
TCRE = dict(land_sea_hx_slope=0.0015, land_sea_hx_offset=7.9251, land_sens_ratio=0.0003, sst_air_factor=1.3000, heatcap_seawater=0.1300, heatcap_land=0.5200,
            heatcap_mixed=7.8000, rf_2xco2=3.7000, climate_sensitivity=2.6122, diffusivity=0.0100)
_CSV_COLS = {"land_sea_hx_slope": "doe_ak", "land_sea_hx_offset": "doe_bk", "heatcap_seawater": "doe_csw", "rf_2xco2": "doe_q2co",
             "land_sens_ratio": "doe_rlam", "sst_air_factor": "doe_bsi", "heatcap_land": "doe_cal", "heatcap_mixed": "doe_cas",
             "climate_sensitivity": "doe_t2co", "diffusivity": "doe_kappa", "tas_0": "T0", "sst_0": "ST0"}


def erf(x):
    """erf evaluated as 2*ndtr(x*sqrt(2)) - 1, ndtr = standard normal CDF."""
    return 2 * ndtr(x * np.sqrt(2)) - 1


@dataclass(frozen=True)
class DoeclimParams:
    """DOECLIM parameters, named as the doe_* parameters without the prefix."""
    land_sea_hx_slope: float
    land_sea_hx_offset: float
    heatcap_seawater: float
    area_earth: float
    diffusivity_unit_conv: float
    rf_2xco2: float
    land_sens_ratio: float
    secs_per_year: float
    depth_bottom: float
    sst_air_factor: float
    heatcap_land: float
    heatcap_mixed: float
    frac_land: float
    frac_ocean_deep: float
    climate_sensitivity: float
    diffusivity: float
    tas_0: float
    sst_0: float
    nsteps: int
    dt: float
    # derived
    area_ocean: float
    c_num: float
    c_den: float
    feedback_land: float
    feedback_sea: float
    land_sea_hx: float
    diffusivity_eff: float
    tau_bottom: float
    power_to_heat: float
    tau_feedback_sea: float
    tau_feedback_land: float
    tau_diffusion: float
    tau_hx_sea: float
    tau_hx_land: float
    # kernel terms (N,), in time-step order, and their sum doe_Ker
    ker_t0: np.ndarray
    ker_ta1: np.ndarray
    ker_tb1: np.ndarray
    ker_ta2: np.ndarray
    ker_tb2: np.ndarray
    ker_ta3: np.ndarray
    ker_tb3: np.ndarray
    ker: np.ndarray
    ker_last: float
    # matrices (mtx_idx 1..4 -> index 0..3)
    mat_c: np.ndarray
    mat_b: np.ndarray
    mat_a: np.ndarray
    temp_d: float
    temp: float
    mat_b_inv: np.ndarray
    t_lo: float = -5.0         # lower bound of tas / temp_sst [K]
    ref_compat: bool = True    # heat diagnostics as in the reference results (doeclim_step)


def _derive(r, nsteps, dt):
    """Derived parameters, kernel and matrices, computed from the raw parameters."""
    d = {}
    d["area_ocean"] = (1.0 - r["frac_land"]) * r["area_earth"]
    d["c_num"] = r["land_sens_ratio"] * r["frac_land"] + r["sst_air_factor"] * (1.0 - r["frac_land"])
    d["c_den"] = r["land_sens_ratio"] * r["frac_land"] - r["land_sea_hx_slope"] * (r["land_sens_ratio"] - r["sst_air_factor"])
    d["feedback_land"] = (r["frac_land"] * d["c_num"] / d["c_den"] * r["rf_2xco2"] / r["climate_sensitivity"]
                - r["land_sea_hx_offset"] * (r["land_sens_ratio"] - r["sst_air_factor"]) / d["c_den"])
    d["feedback_sea"] = ((r["land_sens_ratio"] * r["frac_land"] - r["land_sea_hx_slope"] / (1.0 - r["frac_land"]) * (r["land_sens_ratio"] - r["sst_air_factor"]))
                * d["c_num"] / d["c_den"] * r["rf_2xco2"] / r["climate_sensitivity"]
                + r["land_sens_ratio"] * r["frac_land"] / (1.0 - r["frac_land"]) * r["land_sea_hx_offset"] * (r["land_sens_ratio"] - r["sst_air_factor"]) / d["c_den"])
    d["land_sea_hx"] = (r["land_sea_hx_offset"] * r["land_sens_ratio"] * r["frac_land"] / d["c_den"]
                - r["land_sea_hx_slope"] * r["frac_land"] * d["c_num"] / d["c_den"] * r["rf_2xco2"] / r["climate_sensitivity"])
    d["diffusivity_eff"] = r["diffusivity_unit_conv"] * r["diffusivity"]
    d["tau_bottom"] = r["depth_bottom"]**2 / d["diffusivity_eff"]
    d["power_to_heat"] = d["area_ocean"] * r["secs_per_year"] / 10.0**22
    d["tau_feedback_sea"] = r["heatcap_mixed"] / d["feedback_sea"]
    d["tau_feedback_land"] = r["heatcap_land"] / d["feedback_land"]
    d["tau_diffusion"] = r["heatcap_mixed"]**2 / r["heatcap_seawater"]**2 * DOE_PI / d["diffusivity_eff"]
    d["tau_hx_sea"] = (1.0 - r["frac_land"]) * r["heatcap_mixed"] / d["land_sea_hx"]
    d["tau_hx_land"] = r["frac_land"] * r["heatcap_land"] / d["land_sea_hx"]

    # Kernel of the diffusion integral, third-order bottom correction
    # (Eq. A.25 EK05 / 2.3.23 TK07). It is evaluated per step index; the last
    # step (t.val = last_year) has its own closed form.
    N, tb = nsteps, d["tau_bottom"] / dt
    o = np.arange(1, N + 1, dtype=float)          # ord(t)
    m, mp, mm = N + 1 - o, N + 1 + 1 - o, N + 1 - 1 - o
    with np.errstate(divide="ignore", invalid="ignore"):
        K = {
            "ker_t0": 4.0 * m**0.5 - 2.0 * mp**0.5 - 2.0 * mm**0.5,
            "ker_ta1": (-8.0 * m**0.5 * np.exp(-tb / m) + 4.0 * mp**0.5 * np.exp(-tb / mp)
                     + 4.0 * mm**0.5 * np.exp(-tb / mm)),
            "ker_tb1": 4.0 * (DOE_PI * tb)**0.5 * (erf((tb / mm)**0.5) + erf((tb / mp)**0.5)
                                                - 2.0 * erf((tb / m)**0.5)),
            "ker_ta2": (8.0 * m**0.5 * np.exp(-4.0 * tb / m) - 4.0 * mp**0.5 * np.exp(-4.0 * tb / mp)
                     - 4.0 * mm**0.5 * np.exp(-4.0 * tb / mm)),
            "ker_tb2": -8.0 * (DOE_PI * tb)**0.5 * (erf(2.0 * (tb / mm)**0.5) + erf(2.0 * (tb / mp)**0.5)
                                                 - 2.0 * erf(2.0 * (tb / m)**0.5)),
            "ker_ta3": (-8.0 * m**0.5 * np.exp(-9.0 * tb / m) + 4.0 * mp**0.5 * np.exp(-9.0 * tb / mp)
                     + 4.0 * mm**0.5 * np.exp(-9.0 * tb / mm)),
            "ker_tb3": 12.0 * (DOE_PI * tb)**0.5 * (erf(3.0 * (tb / mm)**0.5) + erf(3.0 * (tb / mp)**0.5)
                                                 - 2.0 * erf(3.0 * (tb / m)**0.5)),
        }
    last = {  # t.val = last_year
        "ker_t0": 4.0 - 2.0 * 2.0**0.5,
        "ker_ta1": -8.0 * np.exp(-tb) + 4.0 * 2.0**0.5 * np.exp(-0.5 * tb),
        "ker_tb1": 4.0 * (DOE_PI * tb)**0.5 * (1.0 + erf((0.5 * tb)**0.5) - 2.0 * erf(tb**0.5)),
        "ker_ta2": 8.0 * np.exp(-4.0 * tb) - 4.0 * 2.0**0.5 * np.exp(-2.0 * tb),
        "ker_tb2": -8.0 * (DOE_PI * tb)**0.5 * (1.0 + erf((2.0 * tb)**0.5) - 2.0 * erf(2.0 * tb**0.5)),
        "ker_ta3": -8.0 * np.exp(-9.0 * tb) + 4.0 * 2.0**0.5 * np.exp(-4.5 * tb),
        "ker_tb3": 12.0 * (DOE_PI * tb)**0.5 * (1.0 + erf((4.5 * tb)**0.5) - 2.0 * erf(3.0 * tb**0.5)),
    }
    for k in K:
        K[k][-1] = last[k]
    d.update(K)
    d["ker"] = K["ker_t0"] + K["ker_ta1"] + K["ker_tb1"] + K["ker_ta2"] + K["ker_tb2"] + K["ker_ta3"] + K["ker_tb3"]
    d["ker_last"] = float(d["ker"][-1])

    # Correction terms against oscillations from short-term forcing (Eq. 2.3.27, TK07)
    tcl, tkl, tcs, tks, bsi = d["tau_feedback_land"], d["tau_hx_land"], d["tau_feedback_sea"], d["tau_hx_sea"], r["sst_air_factor"]
    c = np.array([
        1.0 / tcl**2 + 1.0 / tkl**2 + 2.0 / tcl / tkl + bsi / tkl / tks,
        -1 * bsi / tkl**2 - bsi / tcl / tkl - bsi / tcs / tkl - bsi**2 / tkl / tks,
        -1 * bsi / tks**2 - 1.0 / tcs / tks - 1.0 / tcl / tks - 1.0 / tkl / tks,
        1.0 / tcs**2 + bsi**2 / tks**2 + 2.0 * bsi / tcs / tks + bsi / tkl / tks,
    ]) * (dt**2 / 12.0)
    # Matrices of B*T(i+1) = Q(i) + A*T(i), T = (TL, TS) (Eqs. 2.3.24, 2.3.27, TK07)
    fso_dif = r["frac_ocean_deep"] * (dt / d["tau_diffusion"])**0.5
    b = np.array([1.0 + dt / (2.0 * tcl) + dt / (2.0 * tkl),
                  -dt / (2.0 * tkl) * bsi,
                  -dt / (2.0 * tks),
                  1.0 + dt / (2.0 * tcs) + dt / (2.0 * tks) * bsi + 2.0 * r["frac_ocean_deep"] * (dt / d["tau_diffusion"])**0.5])
    a = np.array([1.0 - dt / (2.0 * tcl) - dt / (2.0 * tkl),
                  dt / (2.0 * tkl) * bsi,
                  dt / (2.0 * tks),
                  1.0 - dt / (2.0 * tcs) - dt / (2.0 * tks) * bsi + d["ker_last"] * fso_dif])
    d["mat_c"], d["mat_b"], d["mat_a"] = c, b + c, a + c
    B = d["mat_b"]
    d["temp_d"] = B[0] * B[3] - B[1] * B[2]
    d["temp"] = 1 / d["temp_d"]
    d["mat_b_inv"] = np.array([d["temp"] * B[3], d["temp"] * (-1) * B[1],
                           d["temp"] * (-1) * B[2], d["temp"] * B[0]])
    return d


def load_doeclim_params(cfg=None, nsteps=None):
    """DOECLIM parameters of a run configuration; nsteps (the diffusion kernel length)
    defaults to the configuration's number of steps."""
    cfg = cfg or RunConfig()
    nsteps = n_steps(cfg) if nsteps is None else nsteps
    return replace(_doeclim_params(nsteps, cfg.cm6mdl, cfg.scen, cfg.ch4rfc, cfg.use_ssp534,
                                   cfg.clim_sensi, cfg.dat_dir), ref_compat=cfg.ref_compat)


def _doeclim_params(nsteps, cm6mdl="NorESM2-MM", scen="contrib", ch4rfc="etminan",
                        use_ssp534=False, clim_sensi=None, dat_dir=DAT_DIR, dt=DT):
    """DOECLIM parameters for one configuration.

    nsteps : number of model steps (card(t)); the kernel depends on it.
    """
    r = dict(DEFAULTS)
    if scen in ("ssp1", "ssp2", "ssp3", "ssp4", "ssp5"):
        if clim_sensi is not None:
            r["climate_sensitivity"] = clim_sensi
    elif cm6mdl is not None:
        name = ("doe_gcm_est_ssp534.csv" if use_ssp534 else
                "doe_gcm_est_etminan.csv" if ch4rfc == "etminan" else "doe_gcm_est_myhre.csv")
        df = pd.read_csv(data_path(dat_dir, "CMIP6", name), index_col=0)
        if cm6mdl not in df.index:
            raise ValueError(f"unknown --cm6mdl={cm6mdl!r} in {name}; options: {list(df.index)}")
        row = df.loc[cm6mdl]
        for k, col in _CSV_COLS.items():
            r[k] = float(row[col]) if col in row.index else 0.0   # missing columns -> 0
        if clim_sensi is not None:
            r["climate_sensitivity"] = clim_sensi
    if scen in ("tcre", "tcre2"):
        r.update(TCRE)
    return DoeclimParams(**r, nsteps=nsteps, dt=dt, **_derive(r, nsteps, dt))


@dataclass
class DoeclimStep:
    """One DOECLIM step, named as the doe_* variables."""
    QL: float
    QO: float
    DelQL: float
    DelQO: float
    QC1: float
    QC2: float
    DQ1: float
    DQ2: float
    DPAST1: float
    DPAST2: float
    DTEAUX1: float
    DTEAUX2: float
    temp_landair: float
    temp_sst: float
    tas: float
    heatflux_mixed: float
    heatflux_interior: float
    heat_mixed: float
    heat_interior: float
    heatflux: float
    heatcontent: float


@dataclass(frozen=True)
class DoeclimDrivers:
    """Input of one DOECLIM step n."""
    ftot: float                 # scm_ftot(n): global forcing [W/m2]


@dataclass(frozen=True)
class DoeclimState:
    """DOECLIM state after step n-1.

    out  : DoeclimStep of that step (the doe_* outputs)
    n    : number of steps done
    hist : temp_sst history of the run, used by the diffusion integral. The buffer is
           shared by the states of one run: a step from state(n) writes hist[n] and reads
           only hist[:n], so a trial step (computed and discarded) does not disturb the
           run, and the next real step from state(n) overwrites hist[n].
    """
    out: "DoeclimStep"
    n: int
    hist: np.ndarray


def doeclim_step(p, prev, drivers):
    """One DOECLIM step: prev = DoeclimState after n-1 (None at the first step),
    drivers = DoeclimDrivers; returns the DoeclimState after step n (outputs in .out).
    Raises ModelBoundsError if tas or temp_sst leaves [p.t_lo, 50].
    """
    q = prev.out if prev is not None else None
    n = prev.n if prev is not None else 0
    TS_hist = prev.hist if prev is not None else np.zeros(p.nsteps)
    N, dt = p.nsteps, p.dt
    if n >= N:
        raise IndexError("DOECLIM was set up for nsteps steps")
    Q = drivers.ftot
    QL = QO = Q                                    # land and ocean forcing = global
    QL1, QO1 = (q.QL, q.QO) if q else (0.0, 0.0)
    TL1, TS1 = (q.temp_landair, q.temp_sst) if q else (0.0, 0.0)
    DelQL, DelQO = QL - QL1, QO - QO1
    QC1 = ((DelQL / p.heatcap_land * (1.0 / p.tau_feedback_land + 1.0 / p.tau_hx_land)
            - p.sst_air_factor * DelQO / p.heatcap_mixed / p.tau_hx_land) * dt**2 / 12.0)
    QC2 = ((DelQO / p.heatcap_mixed * (1.0 / p.tau_feedback_sea + p.sst_air_factor / p.tau_hx_sea)
            - DelQL / p.heatcap_land / p.tau_hx_sea) * dt**2 / 12.0)
    DQ1 = 0.5 * dt / p.heatcap_land * (QL + QL1) + QC1
    DQ2 = 0.5 * dt / p.heatcap_mixed * (QO + QO1) + QC2
    hist = TS_hist[:n]
    # sum over t1 < t of TS(t1) * Ker(t1 + (nsteps - ord(t)))
    DPAST2 = float(hist @ p.ker[N - n - 1:N - 1]) * p.frac_ocean_deep * (dt / p.tau_diffusion)**0.5
    DPAST1 = 0.0                                    # doe_DPAST1.fx = 0
    DTEAUX1 = p.mat_a[0] * TL1 + p.mat_a[1] * TS1
    DTEAUX2 = p.mat_a[2] * TL1 + p.mat_a[3] * TS1
    TL = p.mat_b_inv[0] * (DQ1 + DPAST1 + DTEAUX1) + p.mat_b_inv[1] * (DQ2 + DPAST2 + DTEAUX2)
    TS = p.mat_b_inv[2] * (DQ1 + DPAST1 + DTEAUX1) + p.mat_b_inv[3] * (DQ2 + DPAST2 + DTEAUX2)
    tas = p.frac_land * TL + (1.0 - p.frac_land) * p.sst_air_factor * TS
    # ocean heat uptake (Eq. 2.7 EK05 / 2.3.13 TK07). Reference behaviour (p.ref_compat):
    # hf_mixed is the mixed-layer heat taken up during the step (W yr/m2, not divided by
    # dt), so heat_mixed is dt (1/6) times too small, and heatcontent sums heatflux
    # without dt (its interior part 6x too large). --ref_compat=0: hf_mixed in W/m2,
    # heatcontent = heat_mixed + heat_interior. The frac_ocean_deep factors of the
    # interior terms are those of the DOECLIM equations in both cases.
    hf_mixed = p.heatcap_mixed * (TS - TS1) / (1.0 if p.ref_compat else dt)
    # sum over t1 < t of TS(t1) * Ker(t1 + (nsteps - ord(t) + 1))
    hf_interior = (p.heatcap_mixed * p.frac_ocean_deep / (p.tau_diffusion * dt)**0.5
                   * (2.0 * TS - float(hist @ p.ker[N - n:N])))
    heat_mixed = (q.heat_mixed if q else 0.0) + hf_mixed * (p.power_to_heat * dt)
    heat_interior = (q.heat_interior if q else 0.0) + hf_interior * (p.frac_ocean_deep * p.power_to_heat * dt)
    heatflux = (hf_mixed + p.frac_ocean_deep * hf_interior) * p.power_to_heat
    if p.ref_compat:
        heatcontent = 0.0 if q is None else q.heatcontent + heatflux
    else:
        heatcontent = heat_mixed + heat_interior    # = step sum of heatflux * dt
    s = DoeclimStep(QL=QL, QO=QO, DelQL=DelQL, DelQO=DelQO, QC1=QC1, QC2=QC2, DQ1=DQ1,
                    DQ2=DQ2, DPAST1=DPAST1, DPAST2=DPAST2, DTEAUX1=DTEAUX1, DTEAUX2=DTEAUX2,
                    temp_landair=TL, temp_sst=TS, tas=tas, heatflux_mixed=hf_mixed,
                    heatflux_interior=hf_interior, heat_mixed=heat_mixed,
                    heat_interior=heat_interior, heatflux=heatflux, heatcontent=heatcontent)
    if not (p.t_lo <= tas <= 50 and p.t_lo <= TS <= 50):   # scm_tatm / scm_sst bounds
        raise ModelBoundsError("doe_tas / doe_temp_sst", tas if not p.t_lo <= tas <= 50 else TS,
                              f"[{p.t_lo}, 50]")
    TS_hist[n] = TS                 # history buffer shared along a run (see DoeclimState)
    return DoeclimState(out=s, n=n + 1, hist=TS_hist)




def glue(p, s):
    """Temperature links of the full model: scm_D_tatm, scm_tatm, scm_D_sst, scm_sst.

    +0 is added where doe_T0 / doe_ST0 were once used (no effect on any value).
    """
    return {"scm_D_tatm": s.tas, "scm_tatm": s.tas + 0, "scm_D_sst": s.temp_sst,
            "scm_sst": s.temp_sst + 0}


def delsst(sst_m1, sst_m2, n):
    """scm_delsst(n) = 2*sst(n-1) - sst(n-2), 0 for n < 2 (not used by other equations)."""
    return 0.0 if n < 2 else 2 * sst_m1 - sst_m2


def run_doeclim(p, ftot):
    """Run DOECLIM offline over a prescribed scm_ftot series.

    Returns {doe_* name without the prefix: array} plus the scm_* temperature links.
    """
    steps, state = [], None
    for q in ftot:
        state = doeclim_step(p, state, DoeclimDrivers(q))
        steps.append(state.out)
    out = {f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(DoeclimStep)}
    for k in ("scm_D_tatm", "scm_tatm", "scm_D_sst", "scm_sst"):
        out[k] = np.array([glue(p, s)[k] for s in steps])
    sst = out["scm_sst"]
    out["scm_delsst"] = np.array([delsst(sst[i - 1] if i else 0, sst[i - 2] if i > 1 else 0, i)
                                  for i in range(len(sst))])
    return out


def scm_ftot(f, co2_only=False):
    """scm_ftot(n), DOECLIM's forcing: erf_tot, or rfc_CO2 with --co2_only.

    f : the forcing.ForcingStep of step n.
    """
    return f.rfc_CO2 if co2_only else f.erf_tot
