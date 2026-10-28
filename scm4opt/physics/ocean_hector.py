# SPDX-License-Identifier: GPL-3.0-or-later
"""Hector v1.0 ocean carbon module.

Hartin, C. A., P. Patel, A. Schwarber, R. P. Link and B. P. Bond-Lamberty (2015),
"A simple object-oriented and open-source model for scientific and policy analyses of
the global climate system - Hector v1.0", Geosci. Model Dev. 8, 939-955.

Four boxes: surface low latitude (ob_LL), surface high latitude (ob_HL), intermediate
(ob_IO) and deep (ob_DO). The two surface boxes have carbonate chemistry (DIC + fixed
alkalinity -> [H+], pCO2, pH, Omega, Revelle) and exchange CO2 with the atmosphere
(Takahashi et al. 2009); carbon moves between boxes by advection (thermohaline and
high-latitude overturning) and two-way exchange.

It is an alternative to the default OSCAR ocean, selected with --ocean_model=hector, and
can also be run standalone. Its equations are explicit in time like the rest of SCM4OPT,
so one step is (step n):

  carbon(n)   = carbon(n-1) + carbon_flux(n-1)          (carbon_initial at n = 0)
  Tbox(n)     = scm_tatm(n) * warmingfactor + 15 + deltaT
  chemistry   LL/HL: DIC from carbon(n); [H+] from the alkalinity equation (the only
              implicit relation: solved by brentq inside the bounds of hct_H)
  atmos_flux  (scm_atmoCO2(n) - pCO2o * cpoolscale) * Tr * As * 12 / 1e15 * scm_dt
  carbon_flux(n) = transport/exchange in - out + atmos_flux(n)

Options (load_hector_params):
  cm6mdl       --cm6mdl: transports, initial carbon, cpoolscale, warming factors, U and S
               from data/CMIP6/ocean_gcm_est.csv; None: Hector defaults
  scen         --scen=tcre|tcre2: transports and initial carbon overridden. Reference
               behaviour (ref_compat, default): the tcre transport overrides do not change
               the k-rates; --ref_compat=0 or --hector_tcre_rates recomputes them
  same_alk     --same_alk: all boxes use the mean of the LL/HL alkalinities
Calibration hooks (--cali / --gene) are not included. The Hector reference output
(data/hector/ocean26.csv, rcp_out and the *_hct parameters, used only for comparison) is
available through load_hector_reference.

Usage (offline, prescribed temperature and CO2):
  from scm4opt.physics.ocean_hector import load_hector_params, run_hector
  p = load_hector_params(RunConfig())           # or RunConfig(cm6mdl="ACCESS-CM2",
                                                #   ocean=OceanOptions(same_alk=True))
  out = run_hector(p, tatm, co2)                # arrays over steps -> dict of arrays
Coupled: hector_step(p, prev, OceanDrivers(co2=..., tatm=...)), via ocean_models.
Coupled: python3 -m scm4opt --ocean_model=hector [--same_alk=1] replaces the
OSCAR ocean by this module in scm4opt.py: step n uses scm_tatm(n) and
scm_atmoCO2(n), and s.atmoSeaFlux (GtC per step, positive into the ocean) enters the
atmosphere at n+1 like ocn_atmoSeaFlux. scm_ocn_surface_cpool / scm_ocn_total_cpool are
the LL+HL and the four-box carbon.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..core.config import DAT_DIR, data_path
from ..core.constants import DT
from ..core.errors import ModelBoundsError
from ..core.config import RunConfig
from .ocean_models import OceanDrivers

BOXES = ("ob_LL", "ob_HL", "ob_IO", "ob_DO")
CHEM_BOXES = ("ob_LL", "ob_HL")
LL, HL, IO, DO = range(4)
MEAN_GLOBAL_TEMP = 15.0
OCN_THOUSAND, OCN_MILLION, OCN_SCALE = 1.0e3, 1.0e6, 1.0e6
ALK_LL, ALK_HL = 0.0024352881020550661, 0.0023114639125907501   # mol/kg

# physical bounds of the state; a violation is an error (check_bounds=True)
BOUNDS = {"H": (0.001, 0.1), "dic": (1800.0, 3000.0), "co3_lo": 1.0, "tk_lo": 1.0,
          "Sc_lo": 1.0, "carbon_up": 1e8, "K_lo": 1e-6}


@dataclass(frozen=True)
class HectorParams:
    """Hector ocean parameters; arrays over BOXES."""
    circ_thermohaline: float = 7.2e7           # thermohaline circulation, m3/s
    circ_hl_overturning: float = 4.9e7           # high latitude overturning, m3/s
    exch_warm_inter: float = 1.25e7         # warm-intermediate exchange, m3/s
    exch_inter_deep: float = 2.0e8          # intermediate-deep exchange, m3/s
    exch_cold_deep: float = 0.0
    exch_warm_cold: float = 0.0
    exch_cold_inter: float = 0.0
    dt: float = DT              # scm_dt [yr]
    secs_per_step: float = 0.0           # seconds per step
    thick_ll: float = 100.0
    thick_hl: float = 100.0
    thick_inter: float = 0.0
    thick_deep: float = 0.0
    area: float = 3.6e14        # m2
    part_high: float = 0.15
    part_low: float = 0.0
    volume_ll: float = 0.0
    volume_hl: float = 0.0
    volume_io: float = 0.0
    volume_do: float = 0.0
    transport_k: np.ndarray = field(default=None)   # (4,4) fraction per step, from -> to
    exchange_k: np.ndarray = field(default=None)    # (4,4)
    delta_t: np.ndarray = field(default=None)
    salinity: np.ndarray = field(default=None)             # salinity
    volume: np.ndarray = field(default=None)        # m3
    area_surface: np.ndarray = field(default=None)            # surface area, m2
    wind_speed: np.ndarray = field(default=None)             # wind speed, m/s
    alk: np.ndarray = field(default=None)           # alkalinity * OCN_SCALE
    carbon_initial: np.ndarray = field(default=None)   # PgC
    cpool_scale: np.ndarray = field(default=None)    # over CHEM_BOXES
    warming_factor: np.ndarray = field(default=None)
    check_bounds: bool = True      # stop with ModelBoundsError outside the physical bounds


def _rates(p):
    """Transport and exchange k-rates."""
    T = np.zeros((4, 4))
    T[LL, HL] = (p.circ_thermohaline * p.secs_per_step) / p.volume_ll
    T[HL, DO] = ((p.circ_thermohaline + p.circ_hl_overturning) * p.secs_per_step) / p.volume_hl
    T[DO, IO] = ((p.circ_thermohaline + p.circ_hl_overturning) * p.secs_per_step) / p.volume_do
    T[IO, HL] = (p.circ_hl_overturning * p.secs_per_step) / p.volume_io
    T[IO, LL] = (p.circ_thermohaline * p.secs_per_step) / p.volume_io
    X = np.zeros((4, 4))
    X[IO, LL] = (p.exch_warm_inter * p.secs_per_step) / p.volume_io
    X[LL, IO] = (p.exch_warm_inter * p.secs_per_step) / p.volume_ll
    X[DO, IO] = (p.exch_inter_deep * p.secs_per_step) / p.volume_do
    X[IO, DO] = (p.exch_inter_deep * p.secs_per_step) / p.volume_io
    X[HL, DO] = (p.exch_cold_deep * p.secs_per_step) / p.volume_hl
    X[DO, HL] = (p.exch_cold_deep * p.secs_per_step) / p.volume_do
    X[HL, IO] = (p.exch_cold_inter * p.secs_per_step) / p.volume_hl
    X[IO, HL] = (p.exch_cold_inter * p.secs_per_step) / p.volume_io
    X[LL, HL] = (p.exch_warm_cold * p.secs_per_step) / p.volume_ll
    X[HL, LL] = (p.exch_warm_cold * p.secs_per_step) / p.volume_hl
    return T, X


TCRE = dict(circ_thermohaline=3.6000000E+7, circ_hl_overturning=2.4500000E+7, exch_warm_inter=1.1074378E+7, exch_inter_deep=1.0000000E+8,
            exch_cold_deep=0.0, exch_warm_cold=0.0, exch_cold_inter=0.0)
TCRE_CARBON = (803.6305, 139.1292, 8240.4350, 26149.9661)


# transport / exchange field: column suffix in ocean_gcm_est.csv (ocn_<suffix>)
HECTOR_CSV_COLS = {"circ_thermohaline": "tt", "circ_hl_overturning": "tu", "exch_warm_inter": "twi",
                   "exch_inter_deep": "tid", "exch_cold_deep": "tcd", "exch_warm_cold": "twc",
                   "exch_cold_inter": "tci"}


def load_hector_params(cfg=None):
    """Hector ocean parameters of a run configuration (--cm6mdl, --scen, --same_alk,
    --hector_tcre_rates)."""
    cfg = cfg or RunConfig()
    return _hector_params(cfg.cm6mdl, cfg.scen, cfg.ocean.same_alk, cfg.dat_dir,
                          cfg.ocean.hector_tcre_rates or not cfg.ref_compat)


def _hector_params(cm6mdl=None, scen=None, same_alk=False, dat_dir=None,
                       tcre_rates=False):
    """HectorParams for the given options, derived values kept as their formulas.

    tcre_rates : for --scen=tcre|tcre2, also recompute the k-rates from the overridden
                 transports (default False reproduces the SCM4OPT v4.1 reference results).
    """
    dat_dir = Path(dat_dir) if dat_dir else DAT_DIR
    v = dict(circ_thermohaline=7.2e7, circ_hl_overturning=4.9e7, exch_warm_inter=1.25e7, exch_inter_deep=2.0e8, exch_cold_deep=0.0, exch_warm_cold=0.0, exch_cold_inter=0.0)
    est = None
    if cm6mdl is not None:
        tab = pd.read_csv(data_path(dat_dir, "CMIP6", "ocean_gcm_est.csv"), index_col=0)
        if cm6mdl not in tab.index:
            raise KeyError(f"{cm6mdl} not in ocean_gcm_est.csv")
        est = tab.loc[cm6mdl]
        v = {k: float(est[f"ocn_{HECTOR_CSV_COLS[k]}"]) for k in v}
    dt = DT
    time = 60 * 60 * 24 * 365.25 * dt                           # seconds per step
    thick_LL = thick_HL = 100.0
    thick_inter = 1000 - thick_LL
    thick_deep = 3777 - thick_inter - thick_LL
    area, part_high = 3.6e14, 0.15
    part_low = 1 - part_high
    vol = dict(volume_ll=area * part_low * thick_LL, volume_hl=area * part_high * thick_HL,
               volume_io=area * thick_inter, volume_do=area * thick_deep)
    p = HectorParams(**v, dt=dt, secs_per_step=time, thick_ll=thick_LL, thick_hl=thick_HL,
                     thick_inter=thick_inter, thick_deep=thick_deep, area=area,
                     part_high=part_high, part_low=part_low, **vol)
    T, X = _rates(p)

    deltaT = np.array([7.0, -13.0, 0.0, 0.0])
    S = np.full(4, 34.5)
    volume = np.array([vol["volume_ll"], vol["volume_hl"], vol["volume_io"], vol["volume_do"]])
    As = np.array([area * part_low, area * part_high, 0.0, 0.0])
    U = np.array([6.7, 6.7, 0.0, 0.0])
    alk = np.array([ALK_LL, ALK_HL, 0.0, 0.0]) * OCN_SCALE
    alk[IO] = alk[DO] = (alk[LL] + alk[HL]) / 2
    if same_alk:
        alk[:] = (ALK_LL + ALK_HL) / 2 * OCN_SCALE
    cpoolscale = np.ones(2)
    warmingfactor = np.ones(4)
    if est is not None:
        carbon0 = np.array([est[f"carbon_{b}"] for b in BOXES], dtype=float)
        cpoolscale = np.array([est["cpoolscale_ob_LL"], est["cpoolscale_ob_HL"]], dtype=float)
        warmingfactor = np.array([est[f"ocn_warmingfactor_{b}"] for b in BOXES], dtype=float)
        U = np.full(4, float(est["ocn_U"]))
        S = np.full(4, float(est["ocn_S"]))
    else:
        carbon0 = np.array([750.0, 145.0, 10040.0, 27070.0])
    if scen in ("tcre", "tcre2"):
        p = replace(p, **TCRE)
        carbon0 = np.array(TCRE_CARBON)
        if tcre_rates:
            T, X = _rates(p)
    return replace(p, transport_k=T, exchange_k=X, delta_t=deltaT, salinity=S, volume=volume, area_surface=As,
                   wind_speed=U, alk=alk, carbon_initial=carbon0, cpool_scale=cpoolscale,
                   warming_factor=warmingfactor)


# ---------------------------------------------------------------- carbonate chemistry
@dataclass
class HectorChem:
    """Chemistry of one surface box (outputs hct_<field>, over CHEM_BOXES)."""
    dic: float
    tc: float
    tk: float
    lnK0: float
    K0: float
    Sc: float
    lnKw: float
    Kw: float
    nKhwe74: float
    Kh: float
    pK1mehr: float
    K1: float
    pK2mehr: float
    K2: float
    lnKb: float
    Kb: float
    log10Kspc: float
    Kspc: float
    log10Kspa: float
    Kspa: float
    bor: float
    H: float
    co2st: float
    hco3: float
    co3: float
    PCO2o: float
    pH: float
    Tr: float
    calcium: float
    OmegaCa: float
    OmegaAr: float
    Revelle: float


def convert_to_dic(carbon, volume):
    """DIC of a box from its carbon: PgC -> umol/kg (times OCN_SCALE)."""
    return (carbon * 1e15) * (1.0 / 12.01) * (1.0 / 1027.0) * (1.0 / volume) * OCN_SCALE


def constants(tc, S):
    """Equilibrium constants of a box at temperature tc [degC], salinity S."""
    tk = tc + 273.15
    lnK0 = (-58.0931 + 90.5069 * (100 / tk) + 22.2940 * np.log(tk / 100)
            + S * (0.027766 - 0.025888 * (tk / 100) + 0.0050578 * ((tk / 100) * (tk / 100))))
    Sc = 2073.1 - (125.62 * tc) + (3.6276 * tc * tc) - (0.043219 * tc * tc * tc)
    lnKw = (-13847.26 / tk + 148.96502 - 23.6521 * np.log(tk)
            + (118.67 / tk - 5.977 + 1.0495 * np.log(tk)) * np.sqrt(S) - 0.01615 * S)
    nKhwe74 = (9345.17 / tk - 60.2409 + 23.3585 * np.log(tk / 100)
               + S * (0.023517 - 0.00023656 * tk + 0.0047036e-4 * tk * tk))
    pK1mehr = (3633.86 / tk - 61.2172 + 9.6777 * np.log(tk) - 0.011555 * S
               + 0.0001152 * S * S)
    pK2mehr = (471.78 / tk + 25.9290 - 3.16967 * np.log(tk) - 0.01781 * S
               + 0.0001122 * S * S)
    lnKb = ((-8966.90 - 2890.53 * np.sqrt(S) - 77.942 * S + 1.728 * S ** (3.0 / 2.0)
             - 0.0996 * S * S) / tk
            + 148.0248 + 137.1942 * np.sqrt(S) + 1.62142 * S
            + (-24.4344 - 25.085 * np.sqrt(S) - 0.2474 * S) * np.log(tk)
            + 0.053105 * np.sqrt(S) * tk)
    log10Kspc = (-171.9065 - 0.077993 * tk + 2839.319 / tk + 71.595 * np.log10(tk)
                 + (-0.77712 + 0.0028426 * tk + 178.34 / tk) * np.sqrt(S)
                 - 0.07711 * S + 0.0041249 * S ** 1.5)
    log10Kspa = (-171.945 - 0.077993 * tk + 2903.293 / tk + 71.595 * np.log10(tk)
                 + (-0.068393 + 0.0017276 * tk + 88.135 / tk) * np.sqrt(S)
                 - 0.10018 * S + 0.0059415 * S ** 1.5)
    return dict(tc=tc, tk=tk, lnK0=lnK0, K0=np.exp(lnK0), Sc=Sc, lnKw=lnKw,
                Kw=np.exp(lnKw) * OCN_SCALE, nKhwe74=nKhwe74, Kh=np.exp(nKhwe74),
                pK1mehr=pK1mehr, K1=10 ** (-pK1mehr) * OCN_SCALE,
                pK2mehr=pK2mehr, K2=10 ** (-pK2mehr) * OCN_SCALE,
                lnKb=lnKb, Kb=np.exp(lnKb) * OCN_SCALE,
                log10Kspc=log10Kspc, Kspc=10.0 ** log10Kspc * OCN_SCALE,
                log10Kspa=log10Kspa, Kspa=10.0 ** log10Kspa * OCN_SCALE,
                bor=1 * (416.0 * (S / 35.0)) * 1.e-6 * OCN_SCALE)


def alk_residual(H, dic, alk, k):
    """Alkalinity balance, LHS - RHS, for [H+] (scaled by OCN_SCALE)."""
    K1, K2, Kb, Kw, bor = k["K1"], k["K2"], k["Kb"], k["Kw"], k["bor"]
    lhs = dic * (K1 / H + 2 * K1 * K2 / H ** 2)
    rhs = ((alk - Kb * bor / (Kb + H) - Kw / H * OCN_SCALE + H)
           * (1 + K1 / H + K1 * K2 / H ** 2))
    return lhs - rhs


def surface_chemistry(carbon, Tbox, S, U, volume, alk, H_bounds=BOUNDS["H"]):
    """Carbonate chemistry of one surface box from its carbon [PgC] and temperature."""
    dic = convert_to_dic(carbon, volume)
    k = constants(Tbox, S)
    lo, hi = H_bounds
    f = lambda h: alk_residual(h, dic, alk, k)
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        raise ModelBoundsError(
            "hct_H", None, f"[{lo}, {hi}]",
            f"no [H+] in the bounds [{lo}, {hi}] (dic={dic:.1f}, T={Tbox:.2f}): "
            "the alkalinity balance has no solution")
    H = brentq(f, lo, hi, xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=500)
    K1, K2, Kh = k["K1"], k["K2"], k["Kh"]
    co2st = (dic / (1 + K1 / H + K1 * K2 / H ** 2)) * OCN_MILLION / OCN_SCALE
    hco3 = (dic / (1.0 + H / K1 + K2 / H)) * OCN_MILLION / OCN_SCALE
    co3 = (dic / (1.0 + H / K2 + H * H / K1 / K2)) * OCN_MILLION / OCN_SCALE
    PCO2o = co2st / Kh
    pH = -np.log10(H / OCN_SCALE)
    Tr = 0.585 * k["K0"] * 1 / (k["Sc"] ** 0.5) * U * U          # gC m-2 month-1 uatm-1
    calcium = 0.02128 / 40.087 * (S / 1.80655) / OCN_MILLION
    OmegaCa = (co3 * calcium) / k["Kspc"] * OCN_SCALE
    OmegaAr = (co3 * calcium) / k["Kspa"] * OCN_SCALE
    Revelle = dic / co3
    return HectorChem(dic=dic, H=H, co2st=co2st, hco3=hco3, co3=co3, PCO2o=PCO2o, pH=pH,
                      Tr=Tr, calcium=calcium, OmegaCa=OmegaCa, OmegaAr=OmegaAr,
                      Revelle=Revelle, **k)


def calc_monthly_surface_flux(Ca, PCO2o, cpscale, Tr):
    """calc_monthly_surface_flux macro: gC m-2 month-1."""
    return (Ca - PCO2o * cpscale) * Tr


def calc_annual_surface_flux(Ca, PCO2o, cpscale, Tr, As):
    """calc_annual_surface_flux macro: PgC/yr."""
    return (calc_monthly_surface_flux(Ca, PCO2o, cpscale, Tr) * As * 12.0) / 1e15


# ---------------------------------------------------------------- time step
@dataclass
class HectorStep:
    """State of step n (outputs hct_<field>)."""
    Tbox: np.ndarray           # (4,) degC
    carbon: np.ndarray         # (4,) PgC
    chem: tuple                # (HectorChem LL, HectorChem HL)
    atmos_flux: np.ndarray     # (4,) PgC per step, into the ocean
    atmoSeaFlux: float
    transport: np.ndarray      # (4,4) from -> to
    exchange: np.ndarray       # (4,4)
    carbon_flux: np.ndarray    # (4,) net change to step n+1
    total_carbon: float

    @property
    def surface_cpool(self):
        """Carbon of the two surface boxes (LL + HL) [PgC]."""
        return self.carbon[LL] + self.carbon[HL]

    @property
    def total_cpool(self):
        """Carbon of all four boxes [PgC]."""
        return self.total_carbon


def _check(p, s):
    for b, c in zip(CHEM_BOXES, s.chem):
        lo, hi = BOUNDS["dic"]
        if not lo <= c.dic <= hi:
            raise ModelBoundsError(f"hct_dic({b})", c.dic, f"[{lo}, {hi}]")
        if c.co3 < BOUNDS["co3_lo"] or c.Sc < BOUNDS["Sc_lo"] or c.tk < BOUNDS["tk_lo"]:
            raise ModelBoundsError(f"hct_co3/Sc/tk({b})", None, "their lower bounds")
        if min(c.K0, c.K1, c.K2, c.Kh, c.Kb, c.Kspc, c.Kspa) < BOUNDS["K_lo"]:
            raise ModelBoundsError(f"equilibrium constants ({b})", None, ">= 1e-6")
        if max(c.lnK0, c.lnKw, c.nKhwe74, c.lnKb, c.log10Kspc, c.log10Kspa) > 0:
            raise ModelBoundsError(f"log equilibrium constants ({b})", None, "<= 0")
    if np.any(s.carbon > BOUNDS["carbon_up"]):
        raise ModelBoundsError("hct_carbon", float(np.max(s.carbon)), "<= 1e8")


def hector_step(p, prev, drivers):
    """One step: prev = HectorStep of n-1 (None at the first step); drivers =
    OceanDrivers, uses tatm = scm_tatm(n) [K anomaly] and co2 = scm_atmoCO2(n) [ppm]."""
    tatm, co2, check_bounds = drivers.tatm, drivers.co2, p.check_bounds
    carbon = p.carbon_initial.copy() if prev is None else prev.carbon + prev.carbon_flux
    Tbox = tatm * p.warming_factor + MEAN_GLOBAL_TEMP + p.delta_t
    chem = tuple(surface_chemistry(carbon[i], Tbox[i], p.salinity[i], p.wind_speed[i], p.volume[i], p.alk[i])
                 for i in (LL, HL))
    atmos = np.zeros(4)
    for j, (i, c) in enumerate(zip((LL, HL), chem)):
        atmos[i] = ((co2 - c.PCO2o * p.cpool_scale[j]) * c.Tr) * p.area_surface[i] * 12.0 / 1.0e15 * p.dt
    transport = carbon[:, None] * p.transport_k
    exchange = carbon[:, None] * p.exchange_k
    flux = (transport.sum(0) + exchange.sum(0) - transport.sum(1) - exchange.sum(1) + atmos)
    s = HectorStep(Tbox=Tbox, carbon=carbon, chem=chem, atmos_flux=atmos,
                   atmoSeaFlux=float(atmos.sum()), transport=transport, exchange=exchange,
                   carbon_flux=flux, total_carbon=float(carbon.sum()))
    if check_bounds:
        _check(p, s)
    return s


def run_hector(p, tatm, co2, check_bounds=True):
    """Offline run over prescribed scm_tatm and scm_atmoCO2; returns {model name: array}."""
    p = replace(p, check_bounds=check_bounds)
    steps, prev = [], None
    for T, c in zip(np.asarray(tatm, float), np.asarray(co2, float)):
        prev = hector_step(p, prev, OceanDrivers(co2=c, tatm=T))
        steps.append(prev)
    return collect_hector(steps)


def collect_hector(steps):
    """HectorStep list -> {model name: array}; chemistry arrays are (n, 2) over CHEM_BOXES.
    Output names carry the Hector prefix hct_. Also hct_surface_cpool (LL + HL carbon) and
    hct_total_cpool (all boxes), which the coupled run reports as scm_ocn_surface_cpool /
    scm_ocn_total_cpool."""
    out = {"hct_Tbox": np.array([s.Tbox for s in steps]),
           "hct_carbon": np.array([s.carbon for s in steps]),
           "hct_atmos_flux": np.array([s.atmos_flux for s in steps]),
           "hct_atmoSeaFlux": np.array([s.atmoSeaFlux for s in steps]),
           "hct_carbon_flux": np.array([s.carbon_flux for s in steps]),
           "hct_transport": np.array([s.transport for s in steps]),
           "hct_exchange": np.array([s.exchange for s in steps]),
           "hct_total_carbon": np.array([s.total_carbon for s in steps])}
    out["hct_tc"] = out["hct_Tbox"]
    out["hct_tk"] = out["hct_Tbox"] + 273.15
    for name in HectorChem.__dataclass_fields__:
        if name in ("tc", "tk"):
            continue
        out[f"hct_{name}"] = np.array([[getattr(c, name) for c in s.chem] for s in steps])
    out["hct_surface_cpool"] = out["hct_carbon"][:, LL] + out["hct_carbon"][:, HL]
    out["hct_total_cpool"] = out["hct_total_carbon"]
    return out


# ---------------------------------------------------------------- Hector reference
HCT_COLUMNS = {"hct_carbon": ("ocean_carbon_LL", "ocean_carbon_HL", "ocean_carbon_IO",
                              "ocean_carbon_DO"),
               "hct_dic": ("ocean_DIC_LL", "ocean_DIC_HL"),
               "hct_pH": ("ocean_pH_LL", "ocean_pH_HL"),
               "hct_PCO2o": ("ocean_PCO2_LL", "ocean_PCO2_HL"),
               "hct_Tr": ("ocean_OCN_Tr_LL", "ocean_OCN_Tr_HL"),
               "hct_atmos_flux": ("ocean_atm_ocean_flux_LL", "ocean_atm_ocean_flux_HL")}


def load_hector_reference(dat_dir=None, name="ocean26.csv"):
    """Hector v1.0 reference output rcp_out (data/hector/ocean26.csv):
    the raw table (index = t label) plus hct_<name>_ref arrays (Hector's own output for
    the hct_<name> variables), and the
    drivers simpleNbox_Ca (CO2, ppm) and temperature_Tgav (K)."""
    dat_dir = Path(dat_dir) if dat_dir else DAT_DIR
    tab = pd.read_csv(data_path(dat_dir, "hector", name), dtype={"year": str}).set_index("year")
    ref = {"rcp_out": tab}
    for k, cols in HCT_COLUMNS.items():
        ref[f"{k}_ref"] = tab[list(cols)].to_numpy()
    c = ref["hct_carbon_ref"]
    ref["hct_carbon_mean_ref"] = (c + np.vstack([np.zeros((1, 4)), c[:-1]])) / 2   # (t + t-1)/2
    ref["hct_ocean_c_ref"] = tab["ocean_ocean_c"].to_numpy()
    ref["hct_ocean_c_sum_ref"] = c.sum(1)
    ref["hct_atmos_flux_total_ref"] = tab["ocean_atm_ocean_flux"].to_numpy()
    ref["co2"] = tab["simpleNbox_Ca"].to_numpy()
    ref["tatm"] = tab["temperature_Tgav"].to_numpy()
    return ref
