# SPDX-License-Identifier: GPL-3.0-or-later
"""Radiative forcing: greenhouse gases, ozone, aerosols, land cover and natural terms.

Every term is evaluated at the same step n from the concentrations of n:
  CO2, CH4, N2O       Etminan (default) or Myhre formulas (--ch4rfc)
  strat. H2O          from the lagged CH4
  halogens, O3, aerosol direct effects   radiative efficiency x anomaly
  aerosol-cloud       log(1 + AERh/AERh_0) + BC semi-direct adjustment
  BC on snow          from BC emissions
  land-cover albedo   regional albedo coefficients x land-cover change
  contrails, volcanic, solar, dust       exogenous (inputs.py)
ERF = RF x efficacy; erf_tot drives DOECLIM.

All forcing model choices are available: the option tables below are keyed by the
option names and are selected with config.ForcingOptions (the forcing model options,
--..._mdl flags). Also included: the --rname list that switches Etminan CO2 forcing to the
simple log form, and rfc_LCC = 0 for the natural-forcing group "zero".

--scen=contrib_rfc (components read from data): forcing_from_data.
Not included: --cali_chem and --cali_rf (efficacies become calibration variables).
"""

from dataclasses import dataclass, fields

import numpy as np

from ..core.config import ForcingOptions, nat_grp_of
from ..core.constants import DT
from .chemistry import BASELINES
from .forcing_data import ALPHA_LCC_PARAM, ALPHA_LCC_REG_PARAM, RADEFF_HALO
from ..scenarios.rcmip3_data import LOG_CO2_RNAMES
from ..core.sets import FHALO, HALO, HALO_CFC, HALO_HFC, HALO_ODS, HALO_PFC, LAND_TYPE, MHALO, REGION_11
from ..core.tables import halo_mask, pick
from ..core.errors import ModelBoundsError
from ..core.config import RunConfig, nat_grp_of

# =============================================================================
# Option tables
# =============================================================================

# Tropospheric O3 radiative efficiency [W/m2/DU] (O3Tradeff_mdl)
O3T_RFEF = {
    "IPCC-AR5": 0.042, "IPCC-AR4": 0.032,
    "CESM-CAM-superfast": 0.446 / 10.0, "CICERO-OsloCTM2": 0.401 / 9.3, "CMAM": 0.322 / 7.6,
    "EMAC": 0.460 / 10.8, "GEOSCCM": 0.387 / 8.7, "GFDL-AM3": 0.423 / 10.3,
    "GISS-E2-R": 0.314 / 8.3, "HadGEM2": 0.303 / 7.3, "LMDzORINCA": 0.351 / 8.2,
    "MIROC-CHEM": 0.402 / 9.2, "MOCAGE": 0.219 / 4.8, "NCAR-CAM-35": 0.433 / 10.2,
    "STOC-HadAM3": 0.437 / 10.5, "UM-CAM": 0.376 / 8.7, "TM5": 0.422 / 10.0,
}
# Stratospheric O3 radiative efficiency (O3Sradeff_mdl)
O3S_RFEF = {"IPCC-AR4": 0.004, "ULAQ": -0.059 / (-12.6), "DLR-E39C": -0.027 / (-16.1),
            "NCAR-MACCM": -0.019 / (-12.7), "CHASER": -0.126 / (-14.1)}

# Aerosol direct radiative efficiencies [W/m2/Tg], AeroCom2 [mW/m2/(mg/m2)]
# converted with 1E12/510072E9 (Earth surface area)
_AEROCOM = 1E12 / 510072E9
SO4_RFEF = {k: _AEROCOM * v for k, v in {
    "mean-AeroCom2": -185, "BCC": -108, "CAM4-Oslo": -173, "CAM-51": -104, "GEOS-CHEM": -123,
    "GISS-MATRIX": -196, "GISS-modelE": -307, "GMI": -195, "GOCART": -238, "HadGEM2": -193,
    "IMPACT-Umich": -113, "INCA": -180, "MPIHAM": -125, "NCAR-CAM-35": -354, "OsloCTM2": -192,
    "SPRINTARS": -172}.items()}
POA_RFEF = {k: _AEROCOM * v for k, v in {
    "mean-AeroCom2": -113, "BCC": -97, "CAM4-Oslo": -118, "CAM-51": -69, "GEOS-CHEM": -95,
    "GISS-MATRIX": -129, "GISS-modelE": -76, "GMI": -189, "GOCART": -144, "HadGEM2": -145,
    "IMPACT-Umich": -141, "INCA": -76, "MPIHAM": -41, "NCAR-CAM-35": -48, "OsloCTM2": -165,
    "SPRINTARS": -102}.items()}
BC_RFEF = {k: _AEROCOM * v for k, v in {
    "mean-AeroCom2": 1438, "BCC": 650, "CAM4-Oslo": 1763, "CAM-51": 2661, "GEOS-CHEM": 1067,
    "GISS-MATRIX": 2484, "GISS-modelE": 1253, "GMI": 1208, "GOCART": 874, "HadGEM2": 612,
    "IMPACT-Umich": 1467, "INCA": 1160, "MPIHAM": 1453, "NCAR-CAM-35": 1364, "OsloCTM2": 2161,
    "SPRINTARS": 1322}.items()}
NO3_RFEF = {k: _AEROCOM * v for k, v in {
    "mean-AeroCom2": -166, "GEOS-CHEM": -136, "GISS-MATRIX": -240, "GMI": -103, "HadGEM2": -249,
    "IMPACT-Umich": -155, "INCA": -110, "NCAR-CAM-35": -91, "OsloCTM2": -173}.items()}
SOA_RFEF = {k: _AEROCOM * v for k, v in {
    "mean-AeroCom2": -122, "CAM-51": -45, "GEOS-CHEM": -45, "IMPACT-Umich": -218,
    "MPIHAM": -139, "OsloCTM2": -161}.items()}
RADEFF_DUST = _AEROCOM * 0
RADEFF_SALT = _AEROCOM * 0

# BC semi-direct adjustment (BCadjust_mdl)
BCADJUST_RFEF = {
    "Boucher2013": -0.1 / 0.6,
    "CSIRO":   -0.35 * (-0.1 / 0.6) / (-0.111),
    "GISS":    -0.225 * (-0.1 / 0.6) / (-0.111),
    "HadGEM2": -0.13 * (-0.1 / 0.6) / (-0.111),
    "ECHAM5":  0.05 * (-0.1 / 0.6) / (-0.111),
    "ECMWF":   0.12 * (-0.1 / 0.6) / (-0.111),
}
CLOUD_SCALE = 4.7 * 1.05
AERH_SCALE = 1 / 4
# Aerosol-cloud ERF (CLOUDerf_mdl)
CLOUD_EFRF = {k: v * (-0.45) / (-0.84) for k, v in {
    "mean-ACCMIP": -0.84, "CSIRO-Mk360": -0.99, "GFDL-AM3": -0.82, "GISS-E2-R": -0.61,
    "HadGEM2": -0.89, "LMDzORINCA": -0.21, "MIROC-CHEM": -1.12, "NCAR-CAM-51": -1.22}.items()}
# Hydrophilic aerosol reference burdens (CLOUDerf_mdl): AER_ref0, AER_ref1
AER_EFRF = {
    "mean-ACCMIP": (6.493315, 9.124497), "CSIRO-Mk360": (8.630909, 11.215797),
    "GFDL-AM3": (3.7277446, 5.859241), "GISS-E2-R": (7.1631308, 10.560671),
    "HadGEM2": (7.775085, 9.570654), "LMDzORINCA": (3.8228593, 5.533892),
    "MIROC-CHEM": (3.495258, 5.2857947), "NCAR-CAM-51": (5.2824764, 6.660064),
}
WARMEFF_VOLC = 0.6
# BC on snow (ALBBCrf_mdl)
_BCSNOW_NORM = (0.04 / 4.8) / (0.0146 / (7.9 - 3.2))
ALBBCRF_RFEF = {
    "mean-ACCMIP":     0.0146 / (7.9 - 3.2) * _BCSNOW_NORM,
    "CICERO-OsloCTM2": 0.0131 / (7.8 - 3.1) * _BCSNOW_NORM,
    "GFDL-AM3":        0.0130 / (7.8 - 3.1) * _BCSNOW_NORM,
    "GISS-E2-R":       0.0142 / (8.8 - 4.0) * _BCSNOW_NORM,
    "GISS-E2-R-TOMAS": 0.0175 / (7.8 - 3.1) * _BCSNOW_NORM,
    "HadGEM2":         0.0133 / (7.8 - 3.1) * _BCSNOW_NORM,
    "MIROC-CHEM":      0.0173 / (7.7 - 3.0) * _BCSNOW_NORM,
    "NCAR-CAM-35":     0.0143 / (7.8 - 3.1) * _BCSNOW_NORM,
    "NCAR-CAM-51":     0.0141 / (7.8 - 3.1) * _BCSNOW_NORM,
}
ALBBCWARM_RFEF = {"median": 3., "low": 2., "high": 4.}
# Land-cover albedo efficacy (ALBLCwarm_mdl)
ALBLCWARM_RFEF = {"Hansen2005": 1.02, "Davin2007": 0.5, "Davin2010": 0.78, "Jones2013": 0.79}
# Atmospheric absorption fractions (PRECradfact_mdl)
PRECRADFACT_RFEF = {
    "Andrews2010": dict(p_atm_co2=0.8, p_atm_no_co2=0.5, p_atm_o3t=-0.3, p_atm_strat=0.0,
                        p_atm_scatter=0.0, p_atm_absorb=2.5, p_atm_cloud=0.0, p_atm_alb=0.0,
                        p_atm_solar=0.2),
    "Kvalevag2013": dict(p_atm_co2=2.0 / 3.6, p_atm_no_co2=0.5 / 1.8, p_atm_o3t=0.0,
                         p_atm_strat=0.0, p_atm_scatter=0.6 / (-1.4), p_atm_absorb=3.1 / 0.5,
                         p_atm_cloud=0.0, p_atm_alb=0.0, p_atm_solar=0.4 / 3.5),
}
# Efficacies; the unused ones are kept for completeness
RF_EFFICACY = {
    "CO2": 1.000001, "CH4": 1.012385, "CH4OXSTRATH2O": 1.0, "N2O": 1.004492,
    "STRATOZ": 1.25, "TROPOZ": 0.85, "AER_DIR": 1.3, "CLOUD_ALBEDO": 1.5621921368015046,
    "CLOUD_COVER": 0.9, "SOLAR": 0.85, "VOLC": 0.95, "LANDUSE": 1.0, "BCSNOW": 1.7,
    "HALO": 1.0, "MHALO": 1.0, "QXTRA": 1.0, "AIRH2O": 1.0, "CONTRAIL": 1.0, "CIRRUS": 1.0,
    "NON_CO2": 0.623660013463149, "CRETC": 1.0,
}
RFC_F2X = 3.71






# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class ForcingParams:
    """Forcing parameters (model names); build with load_forcing_params()."""
    ch4rfc: str
    co2_log: bool              # rfc_CO2 = 5.35*log(1 + D_CO2/CO2_init) (Myhre or listed --rname)
    lcc_zero: bool             # nat_grp == "zero": rfc_LCC = 0
    co2_init: float
    co2_0: float
    ch4_init: float
    ch4_0: float
    n2o_init: float
    n2o_0: float
    rf_2xco2: float
    rf_2xco2_etminan: float
    scale_rf_co2: float
    radeff_halo: np.ndarray    # (48,) W/m2/ppt
    radeff_o3t: float
    radeff_o3s: float
    radeff_so4: float
    radeff_poa: float
    radeff_bc: float
    radeff_no3: float
    radeff_soa: float
    radeff_dust: float
    radeff_salt: float
    k_bc_adjust: float
    cloud_scale: float
    aerh_scale: float
    rf_ref1: float
    aer_ref0: float
    aer_ref1: float
    phi_0: float
    aerh_0: float
    warmeff_volc: float
    radeff_bc_snow: float
    warmeff_bc_snow: float
    alpha_lcc: np.ndarray      # (5,) global coefficients (computed, unused by the equations)
    alpha_lcc_reg: np.ndarray  # (11, 5) regional coefficients used by rfc_LCC_rgn
    warmeff_lcc: float
    p_atm_co2: float
    p_atm_no_co2: float
    p_atm_o3t: float
    p_atm_strat: float
    p_atm_scatter: float
    p_atm_absorb: float
    p_atm_cloud: float
    p_atm_alb: float
    p_atm_solar: float
    efficacy: dict                  # RF_EFFICACY_*
    dt: float = DT


def load_forcing_params(cfg=None, efficacy=None, baselines=None):
    """Forcing parameters of a run configuration (efficacy: RF efficacy overrides;
    baselines: CO2/CH4/N2O baseline overrides)."""
    cfg = cfg or RunConfig()
    return _forcing_params(cfg.forcing, cfg.ch4rfc, cfg.rname, nat_grp_of(cfg), efficacy,
                           baselines)


def _forcing_params(opts=None, ch4rfc="etminan", rname=None, nat_grp="default",
                        efficacy=None, baselines=None):
    """Forcing parameters for one set of model choices.

    efficacy  : optional {name: value} overriding RF_EFFICACY (e.g. calibrated values).
    baselines : optional CO2_init/CH4_0/... dict replacing chemistry's BASELINES[ch4rfc]
    """
    o = opts or ForcingOptions()
    b = baselines or pick(BASELINES, ch4rfc, "--ch4rfc")
    F2x_etminan = (-2.4e-7 * b["co2_init"]**2 + 7.2e-4 * b["co2_init"]
                   - 2.1e-4 * b["n2o_init"] + 5.36) * np.log(2)
    RF_ref1 = pick(CLOUD_EFRF, o.CLOUDerf_mdl, "--CLOUDerf_mdl")
    AER_ref0, AER_ref1 = AER_EFRF[o.CLOUDerf_mdl]
    Phi_0 = RF_ref1 / np.log(AER_ref1 / AER_ref0)          # Phi_0_rfef
    AERh_0 = pick({"median": AER_ref0 * np.exp(1 * (1.42 - 1.30) / Phi_0),   # AERh_0_rfef
                    "high":   AER_ref0 * np.exp(0 * (1.42 - 1.30) / Phi_0),
                    "low":    AER_ref0 * np.exp(2 * (1.42 - 1.30) / Phi_0)},
                   o.CLOUDpreind_mdl, "--CLOUDpreind_mdl")
    lcc_key = (o.ALBLCflux_mdl, o.ALBLCalb_mdl, o.ALBLCcover_mdl)
    eff = dict(RF_EFFICACY)
    eff.update(efficacy or {})
    return ForcingParams(
        ch4rfc=ch4rfc,
        co2_log=(ch4rfc != "etminan") or (rname in LOG_CO2_RNAMES),
        lcc_zero=(nat_grp == "zero"),
        co2_init=b["co2_init"], co2_0=b["co2_0"], ch4_init=b["ch4_init"], ch4_0=b["ch4_0"],
        n2o_init=b["n2o_init"], n2o_0=b["n2o_0"],
        rf_2xco2=RFC_F2X, rf_2xco2_etminan=F2x_etminan, scale_rf_co2=RFC_F2X / F2x_etminan,
        radeff_halo=np.array([RADEFF_HALO[g] for g in HALO]) * 1E-3,
        radeff_o3t=pick(O3T_RFEF, o.O3Tradeff_mdl, "--O3Tradeff_mdl"),
        radeff_o3s=pick(O3S_RFEF, o.O3Sradeff_mdl, "--O3Sradeff_mdl"),
        radeff_so4=pick(SO4_RFEF, o.SO4radeff_mdl, "--SO4radeff_mdl"),
        radeff_poa=pick(POA_RFEF, o.POAradeff_mdl, "--POAradeff_mdl"),
        radeff_bc=pick(BC_RFEF, o.BCradeff_mdl, "--BCradeff_mdl"),
        radeff_no3=pick(NO3_RFEF, o.NO3radeff_mdl, "--NO3radeff_mdl"),
        radeff_soa=pick(SOA_RFEF, o.SOAradeff_mdl, "--SOAradeff_mdl"),
        radeff_dust=RADEFF_DUST, radeff_salt=RADEFF_SALT,
        k_bc_adjust=pick(BCADJUST_RFEF, o.BCadjust_mdl, "--BCadjust_mdl"),
        cloud_scale=CLOUD_SCALE, aerh_scale=AERH_SCALE,
        rf_ref1=RF_ref1, aer_ref0=AER_ref0, aer_ref1=AER_ref1, phi_0=Phi_0, aerh_0=AERh_0,
        warmeff_volc=WARMEFF_VOLC,
        radeff_bc_snow=pick(ALBBCRF_RFEF, o.ALBBCrf_mdl, "--ALBBCrf_mdl"),
        warmeff_bc_snow=pick(ALBBCWARM_RFEF, o.ALBBCwarm_mdl, "--ALBBCwarm_mdl"),
        alpha_lcc=np.array(pick(ALPHA_LCC_PARAM, lcc_key, "--ALBLCflux/alb/cover_mdl")),
        alpha_lcc_reg=np.array(ALPHA_LCC_REG_PARAM[lcc_key]),
        warmeff_lcc=pick(ALBLCWARM_RFEF, o.ALBLCwarm_mdl, "--ALBLCwarm_mdl"),
        **pick(PRECRADFACT_RFEF, o.PRECradfact_mdl, "--PRECradfact_mdl"),
        efficacy=eff,
    )


# =============================================================================
# Forcing functions
# =============================================================================

def f_RF_CO2(p, D_CO2):
    return 5.35 * np.log(1 + D_CO2 / p.co2_init)


def f_RF_CH4(p, D_CH4):
    return 0.036 * ((p.ch4_0 + D_CH4)**0.5 - p.ch4_0**0.5)


def f_RF_H2Os(p, D_CH4_lag):
    return 0.15 * 0.036 * ((p.ch4_0 + D_CH4_lag)**0.5 - p.ch4_0**0.5)


def f_RF_N2O(p, D_N2O):
    return 0.12 * ((p.n2o_0 + D_N2O)**0.5 - p.n2o_0**0.5)


def f_RF_overlap(p, D_CH4, D_N2O):
    """CH4-N2O band overlap relative to the baseline (Myhre et al. 2013)."""
    M, N = p.ch4_0 + D_CH4, p.n2o_0 + D_N2O
    M0, N0 = p.ch4_0, p.n2o_0
    return ((0.47 * np.log(1 + 2.01E-5 * (M * N)**0.75 + 5.31E-15 * M * (M * N)**1.52))
            - (0.47 * np.log(1 + 2.01E-5 * (M0 * N0)**0.75 + 5.31E-15 * M0 * (M0 * N0)**1.52)))


# =============================================================================
# One step
# =============================================================================

_GROUPS = {"HFC": halo_mask(HALO_HFC), "PFC": halo_mask(HALO_PFC), "ODS": halo_mask(HALO_ODS),
           "CFC": halo_mask(HALO_CFC)}
_FHALO, _MHALO = halo_mask(FHALO), halo_mask(MHALO)


@dataclass
class ForcingStep:
    """All forcing variables of one step, named as the rfc_* / erf_* variables."""
    rfc_CO2_Cbar: float = np.nan   # Etminan only
    rfc_CH4_Mbar: float = np.nan
    rfc_N2O_Nbar: float = np.nan
    rfc_CO2: float = 0.0
    rfc_CH4: float = 0.0
    rfc_H2Os: float = 0.0
    rfc_N2O: float = 0.0
    rfc_HALO: np.ndarray = None    # (48,)
    rfc_FHFC: float = 0.0
    rfc_FPFC: float = 0.0
    rfc_FODS: float = 0.0
    rfc_FCFC: float = 0.0
    rfc_CFGS: float = 0.0
    rfc_CMNT: float = 0.0
    rfc_CHALO: float = 0.0
    rfc_O3t: float = 0.0
    rfc_O3s: float = 0.0
    rfc_SO4: float = 0.0
    rfc_POA: float = 0.0
    rfc_BC: float = 0.0
    rfc_NO3: float = 0.0
    rfc_SOA: float = 0.0
    rfc_SALT: float = 0.0
    rfc_DUST: float = 0.0
    rfc_cloud: float = 0.0
    rfc_BCsnow: float = 0.0
    rfc_LCC_rgn: np.ndarray = None  # (11,)
    rfc_LCC_lnd: np.ndarray = None  # (5,)
    rfc_LCC: float = 0.0
    rfc_con: float = 0.0
    rfc_volc: float = 0.0
    rfc_solar: float = 0.0
    rfc_co2ch4n2o: float = 0.0
    rfc_other_wmghg: float = 0.0
    rfc_ghg: float = 0.0
    rfc_aero: float = 0.0
    rfc_aero_excloud: float = 0.0
    rfc_anthro: float = 0.0
    rfc_anthro_other: float = 0.0
    rfc_natural: float = 0.0
    rfc_tot: float = 0.0
    rfc_atm: float = 0.0
    erf_CO2: float = 0.0
    erf_CH4: float = 0.0
    erf_H2Os: float = 0.0
    erf_N2O: float = 0.0
    erf_HALO: np.ndarray = None
    erf_CFGS: float = 0.0
    erf_CMNT: float = 0.0
    erf_CHALO: float = 0.0
    erf_FODS: float = 0.0
    erf_FHFC: float = 0.0
    erf_FPFC: float = 0.0
    erf_FCFC: float = 0.0
    erf_SO4: float = 0.0
    erf_POA: float = 0.0
    erf_BC: float = 0.0
    erf_NO3: float = 0.0
    erf_SOA: float = 0.0
    erf_SALT: float = 0.0
    erf_O3t: float = 0.0
    erf_O3s: float = 0.0
    erf_AerDir: float = 0.0
    erf_cloud: float = 0.0
    erf_BCsnow: float = 0.0
    erf_LCC: float = 0.0
    erf_con: float = 0.0
    erf_volc: float = 0.0
    erf_solar: float = 0.0
    erf_DUST: float = 0.0
    erf_co2ch4n2o: float = 0.0
    erf_other_wmghg: float = 0.0
    erf_ghg: float = 0.0
    erf_aero: float = 0.0
    erf_aero_excloud: float = 0.0
    erf_anthro: float = 0.0
    erf_anthro_other: float = 0.0
    erf_natural: float = 0.0
    erf_tot: float = 0.0


@dataclass(frozen=True)
class ForcingDrivers:
    """Inputs of one forcing step n."""
    d_co2: float              # scm_D_CO2(n) [ppm]
    chem: object              # chemistry state of step n (ChemStep or any object with its fields)
    ems_bc: float             # dat_ems(n,"bc") [per step]
    land_cover: object        # land_cover_WORLD(n), (11, 5) [Mha]
    con: float                # rfc_con(n) [W/m2]
    volc: float               # rfc_volc(n)
    solar: float              # rfc_solar(n)
    dust: float               # rfc_DUST(n)
    d_ebb_bc: float = 0.0     # biomass-burning BC anomaly


def forcing_step(p, drivers):
    """Radiative forcing and ERF of one step (forcing has no state of its own).

    drivers    : ForcingDrivers of step n:
    d_co2      : scm_D_CO2(n) [ppm]
    chem       : the chemistry state of step n; any object with D_CH4, D_N2O, D_CH4_lag,
                 D_HALO (48,), D_O3t, D_O3s, D_SO4, D_POA, D_BC, D_NO3, D_SOA, D_SALT, D_AERh
    ems_bc     : dat_ems(n,"bc") [per step]
    land_cover : land_cover_WORLD(n) (11, 5) [Mha]
    con, volc, solar, dust : rfc_con / rfc_volc / rfc_solar / rfc_DUST(n) [W/m2]
    d_ebb_bc   : biomass-burning BC anomaly added to ems_bc for BC on snow
    Returns a ForcingStep; its erf_tot is DOECLIM's forcing at n.
    """
    D_CO2, c, ems_bc, land_cover, con, volc, solar, dust, d_ebb_bc = (
        drivers.d_co2, drivers.chem, drivers.ems_bc, drivers.land_cover, drivers.con,
        drivers.volc, drivers.solar, drivers.dust, drivers.d_ebb_bc)
    f = ForcingStep(rfc_con=con, rfc_volc=volc, rfc_solar=solar, rfc_DUST=dust)

    # CO2, CH4, N2O
    if p.ch4rfc == "etminan":
        f.rfc_CO2_Cbar = (D_CO2 + p.co2_0 + p.co2_init) * 0.5
        f.rfc_CH4_Mbar = (c.D_CH4 + p.ch4_0 + p.ch4_init) * 0.5
        f.rfc_N2O_Nbar = (c.D_N2O + p.n2o_0 + p.n2o_init) * 0.5
        if p.co2_log:
            f.rfc_CO2 = 5.35 * np.log(1 + D_CO2 / p.co2_init)
        else:
            C = D_CO2 + p.co2_0
            f.rfc_CO2 = ((-2.4e-7 * (C - p.co2_init)**2 + 7.2e-4 * (C - p.co2_init)
                          - 2.1e-4 * f.rfc_N2O_Nbar + 5.36)
                         * np.log(C / p.co2_init) * p.scale_rf_co2)
        f.rfc_CH4 = ((-1.3e-6 * f.rfc_CH4_Mbar - 8.2e-6 * f.rfc_N2O_Nbar + 0.043)
                     * ((c.D_CH4 + p.ch4_0)**0.5 - p.ch4_init**0.5))
        f.rfc_N2O = ((-8.0e-6 * f.rfc_CO2_Cbar + 4.2e-6 * f.rfc_N2O_Nbar
                      - 4.9e-6 * f.rfc_CH4_Mbar + 0.117)
                     * ((c.D_N2O + p.n2o_0)**0.5 - p.n2o_init**0.5))
    else:   # Myhre et al. (2013)
        f.rfc_CO2 = 5.35 * np.log(1 + D_CO2 / p.co2_init)
        f.rfc_CH4 = (0.036 * ((p.ch4_0 + c.D_CH4)**0.5 - p.ch4_0**0.5)
                     - (f_RF_overlap(p, c.D_CH4, c.D_N2O) - f_RF_overlap(p, 0, c.D_N2O)))
        f.rfc_N2O = (0.12 * ((p.n2o_0 + c.D_N2O)**0.5 - p.n2o_0**0.5)
                     - (f_RF_overlap(p, c.D_CH4, c.D_N2O) - f_RF_overlap(p, c.D_CH4, 0)))
    f.rfc_H2Os = 0.15 * 0.036 * ((p.ch4_0 + c.D_CH4_lag)**0.5 - p.ch4_0**0.5)

    # Halogens
    f.rfc_HALO = p.radeff_halo * c.D_HALO
    _halo_sums(f)

    # Ozone and aerosols
    f.rfc_O3t = p.radeff_o3t * c.D_O3t
    f.rfc_O3s = p.radeff_o3s * c.D_O3s
    f.rfc_SO4 = p.radeff_so4 * c.D_SO4
    f.rfc_POA = p.radeff_poa * c.D_POA
    f.rfc_BC = p.radeff_bc * c.D_BC
    f.rfc_NO3 = p.radeff_no3 * c.D_NO3
    f.rfc_SOA = p.radeff_soa * c.D_SOA
    f.rfc_SALT = p.radeff_salt * c.D_SALT
    f.rfc_cloud = ((p.k_bc_adjust * f.rfc_BC
                    + p.aerh_scale * p.phi_0 * np.log(1 + c.D_AERh / p.aerh_0)) * p.cloud_scale)
    f.rfc_BCsnow = p.radeff_bc_snow * (ems_bc + d_ebb_bc) / p.dt

    # Land-cover albedo
    contrib = p.alpha_lcc_reg * land_cover          # (11, 5)
    f.rfc_LCC_rgn = contrib.sum(axis=1)
    f.rfc_LCC_lnd = contrib.sum(axis=0)
    f.rfc_LCC = 0.0 if p.lcc_zero else float(f.rfc_LCC_rgn.sum())
    return _finish(p, f)


def _halo_sums(f):
    """Group sums of rfc_HALO (rfc_FHFC ... rfc_CHALO)."""
    f.rfc_FHFC = float(f.rfc_HALO[_GROUPS["HFC"]].sum())
    f.rfc_FPFC = float(f.rfc_HALO[_GROUPS["PFC"]].sum())
    f.rfc_FODS = float(f.rfc_HALO[_GROUPS["ODS"]].sum())
    f.rfc_FCFC = float(f.rfc_HALO[_GROUPS["CFC"]].sum())
    f.rfc_CFGS = float(f.rfc_HALO[_FHALO].sum())
    f.rfc_CMNT = float(f.rfc_HALO[_MHALO].sum())
    f.rfc_CHALO = float(f.rfc_HALO.sum())


def forcing_from_data(p, comp, halo, con):
    """--scen=contrib_rfc: forcing components prescribed, one step.

    comp : {name: value} for CO2, CH4, H2Os, N2O, O3t, O3s, SO4, POA, BC, BCsnow, NO3,
           SOA, DUST, volc, solar, cloud, LCC [W/m2]; halo : rfc_HALO (48,); con : rfc_con.
    Aggregates, ERF and rfc_atm are computed exactly as in forcing_step.
    """
    f = ForcingStep(rfc_con=con, rfc_volc=comp["volc"], rfc_solar=comp["solar"],
                    rfc_DUST=comp["DUST"])
    f.rfc_CO2, f.rfc_CH4, f.rfc_N2O, f.rfc_H2Os = comp["CO2"], comp["CH4"], comp["N2O"], comp["H2Os"]
    f.rfc_HALO = np.asarray(halo, dtype=float)
    _halo_sums(f)
    f.rfc_O3t, f.rfc_O3s = comp["O3t"], comp["O3s"]
    f.rfc_SO4, f.rfc_POA, f.rfc_BC = comp["SO4"], comp["POA"], comp["BC"]
    f.rfc_NO3, f.rfc_SOA, f.rfc_SALT = comp["NO3"], comp["SOA"], 0.0
    f.rfc_cloud, f.rfc_BCsnow, f.rfc_LCC = comp["cloud"], comp["BCsnow"], comp["LCC"]
    return _finish(p, f)


def _finish(p, f):
    """RF aggregates, ERF and rfc_atm from the components in f."""
    e = p.efficacy
    # RF aggregates
    f.rfc_co2ch4n2o = f.rfc_CO2 + f.rfc_CH4 + f.rfc_N2O
    f.rfc_other_wmghg = f.rfc_CHALO
    f.rfc_ghg = f.rfc_co2ch4n2o + f.rfc_CHALO
    f.rfc_aero_excloud = (f.rfc_SO4 + f.rfc_POA + f.rfc_BC + f.rfc_NO3 + f.rfc_SOA
                          + f.rfc_DUST + f.rfc_SALT)
    f.rfc_aero = f.rfc_aero_excloud + f.rfc_cloud
    f.rfc_anthro = (f.rfc_ghg + f.rfc_aero + f.rfc_H2Os + f.rfc_O3t + f.rfc_O3s
                    + f.rfc_BCsnow + f.rfc_LCC + f.rfc_con)
    f.rfc_anthro_other = (f.rfc_BCsnow + f.rfc_H2Os + f.rfc_con + f.rfc_other_wmghg
                          + f.rfc_O3s + f.rfc_O3t)
    f.rfc_natural = f.rfc_volc + f.rfc_solar
    f.rfc_tot = f.rfc_anthro + f.rfc_natural

    # ERF
    f.erf_CO2 = f.rfc_CO2 * e["CO2"]
    f.erf_CH4 = f.rfc_CH4 * e["CH4"]
    f.erf_H2Os = f.rfc_H2Os * e["CH4OXSTRATH2O"]
    f.erf_N2O = f.rfc_N2O * e["N2O"]
    f.erf_HALO = f.rfc_HALO * e["HALO"]
    f.erf_CFGS = f.rfc_CFGS * e["HALO"]
    f.erf_CMNT = f.rfc_CMNT * e["MHALO"]
    f.erf_CHALO = f.rfc_CHALO * e["HALO"]
    f.erf_FODS = f.rfc_FODS * e["MHALO"]
    f.erf_FHFC = f.rfc_FHFC * e["HALO"]
    f.erf_FPFC = f.rfc_FPFC * e["HALO"]
    f.erf_FCFC = f.rfc_FCFC * e["HALO"]
    f.erf_SO4 = f.rfc_SO4 * e["AER_DIR"]
    f.erf_POA = f.rfc_POA * e["AER_DIR"]
    f.erf_BC = f.rfc_BC * e["AER_DIR"]
    f.erf_NO3 = f.rfc_NO3 * e["AER_DIR"]
    f.erf_SOA = f.rfc_SOA * e["AER_DIR"]
    f.erf_SALT = f.rfc_SALT * e["AER_DIR"]
    f.erf_O3t = f.rfc_O3t * e["TROPOZ"]
    f.erf_O3s = f.rfc_O3s * e["STRATOZ"]
    f.erf_AerDir = f.rfc_aero_excloud * e["AER_DIR"]
    f.erf_cloud = f.rfc_cloud * e["CLOUD_ALBEDO"]
    f.erf_BCsnow = p.warmeff_bc_snow * f.rfc_BCsnow
    f.erf_LCC = p.warmeff_lcc * f.rfc_LCC
    f.erf_con = f.rfc_con * e["CONTRAIL"]
    f.erf_volc = p.warmeff_volc * f.rfc_volc
    f.erf_solar = f.rfc_solar * e["SOLAR"]
    f.erf_DUST = f.rfc_DUST * e["AER_DIR"]
    f.erf_co2ch4n2o = f.erf_CO2 + f.erf_CH4 + f.erf_N2O
    f.erf_other_wmghg = f.erf_CHALO
    f.erf_ghg = f.erf_co2ch4n2o + f.erf_CHALO
    f.erf_aero = f.erf_AerDir + f.erf_cloud
    f.erf_aero_excloud = f.erf_AerDir
    f.erf_anthro = (f.erf_ghg + f.erf_aero + f.erf_H2Os + f.erf_O3t + f.erf_O3s
                    + f.erf_BCsnow + f.erf_LCC + f.erf_con)
    f.erf_anthro_other = (f.erf_BCsnow + f.erf_H2Os + f.erf_con + f.erf_other_wmghg
                          + f.erf_O3s + f.erf_O3t)
    f.erf_natural = f.erf_volc + f.erf_solar
    f.erf_tot = f.erf_anthro + f.erf_natural

    # Atmospheric absorption (precipitation response)
    f.rfc_atm = (p.p_atm_co2 * f.rfc_CO2
                 + p.p_atm_no_co2 * (f.rfc_CH4 + f.rfc_N2O + f.rfc_CFGS + f.rfc_CMNT)
                 + p.p_atm_o3t * f.rfc_O3t + p.p_atm_strat * (f.rfc_O3s + f.rfc_H2Os)
                 + p.p_atm_scatter * (f.rfc_SO4 + f.rfc_POA + f.rfc_NO3 + f.rfc_SOA
                                      + f.rfc_DUST + f.rfc_SALT + f.rfc_volc)
                 + p.p_atm_absorb * f.rfc_BC + p.p_atm_cloud * (f.rfc_cloud + f.rfc_con)
                 + p.p_atm_alb * (f.rfc_BCsnow + f.rfc_LCC) + p.p_atm_solar * f.rfc_solar)

    if not -50 <= f.rfc_tot <= 50:   # bounds of rfc_tot
        raise ModelBoundsError("rfc_tot", f.rfc_tot, "[-50, 50]")
    return f


class _Chem:
    """Chemistry state of one step for forcing_step, from arrays (offline runs)."""
    def __init__(self, chem, i):
        for k, v in chem.items():
            setattr(self, k, v[i])


def run_forcing(p, D_CO2, chem, ems_bc, land_cover, con, volc, solar, dust):
    """Run forcing offline over prescribed series.

    chem : {name: array} with the fields listed in forcing_step (D_HALO is (N, 48)).
    Returns {variable name: array}.
    """
    steps = [forcing_step(p, ForcingDrivers(D_CO2[i], _Chem(chem, i), ems_bc[i], land_cover[i],
                                            con[i], volc[i], solar[i], dust[i]))
             for i in range(len(D_CO2))]
    return {f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(ForcingStep)}
