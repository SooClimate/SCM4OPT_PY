# SPDX-License-Identifier: GPL-3.0-or-later
"""Atmospheric chemistry: CH4, N2O, halogenated gases, ozone and aerosols.

Concentrations are anomalies (D_*) on top of baselines (CH4_0, N2O_0, HALO_0).
CH4, N2O and the 48 halogenated gases are stocks updated by explicit Euler with
the sinks of the previous step; ozone and aerosols are diagnostics.

A step is split in two because the OH and photolysis sinks need temperature at
the SAME step, which DOECLIM only provides after forcing:

  chemistry_step(n)    stocks from n-1 sinks; EESC, O3t, O3s, aerosols (tatm at n-1)
  ... forcing(n) -> DOECLIM(n) -> tatm(n) ...
  chemistry_finish(n)  kOH, hv (tatm at n) -> sinks(n), emeth(n)   [used at n+1]

All chemistry model choices are available: the option tables below are keyed by
the option names and are selected with config.ChemOptions (the chemistry model
options, --..._mdl flags). The default options give the SCM4OPT v4.1 defaults.

--load_conc (prescribed concentrations) is supported through ChemDrivers(prescribed=...).
Not included: the --cali_chem calibration branch.
"""

from dataclasses import dataclass, fields

import numpy as np

from ..core.config import ChemOptions
from ..core.constants import DT
from .halo_data import COLUMNS as _HCOLS, FRACREL, HALO_TABLE
from ..core.sets import FHALO, HALO, HALO_CFC, HALO_HFC, HALO_ODS, HALO_PFC, MHALO
from ..core.tables import halo_mask, pick
from ..core.config import RunConfig

INF = np.inf

CH4_FOSSFUELFRACTION = 0.18
CH4_OXIDATION_COEFF = 0.003

# =============================================================================
# Option tables
# =============================================================================

# Baselines by CH4 forcing pattern (--ch4rfc)
BASELINES = {
    "etminan": dict(co2_init=279., ch4_init=729, n2o_init=273., co2_0=284, ch4_0=729,
                    n2o_0=270, d_ch4_1850=67.84303284, d_n2o_1850=5.0650513),
    "myhre":   dict(co2_init=279., co2_0=284, ch4_init=722, ch4_0=731.4059957,
                    n2o_init=273, n2o_0=270, d_ch4_1850=76.84303284, d_n2o_1850=5.0650513),
}

# Age of air (HVSNKcirc_mdl)
CHM_GAMMA_AGE = {"CAM-35": 0.091375, "CMAM": 0.090625, "Niwa-SOCOL": 0.0683125,
                 "SOCOL": 0.1441875, "ULAQ": 0.11775, "UMUKCA-UCAM": 0.086125}

# Tropospheric O3 vs emissions (O3Temis_mdl): chi_O3t_CH4, _CO, _VOC, _NOX
CHI_O3T_RFEF = {
    "mean-OxComp":     (5.0, 0.0025666666, 0.0033, 0.125),
    "CICERO-OsloCTM2": (3.8634875, 0.0027303882, 0.019194925, 0.1493967),
    "NCAR-CAM-35":     (3.8942387, 0.00453162, 0.009061918, 0.15242542),
    "STOC-HadAM3":     (4.0814595, 0.0028336823, 0.007932475, 0.15485853),
    "UM-CAM":          (2.3116956, 0.0023664278, 0.008332977, 0.16836293),
}
SCALE_O3T = 0.43 / 0.55

# Tropospheric O3 vs climate (O3Tclim_mdl)
CHM_GAMMA_O3T = {"CESM-CAM-superfast": -0.35975, "GFDL-AM3": 0.095125, "GISS-E2-R": -0.28025,
                 "MIROC-CHEM": -0.201, "MOCAGE": -0.6781875, "NCAR-CAM-35": -0.6023125,
                 "STOC-HadAM3": -0.6823125, "UM-CAM": 0.101625}

# Stratospheric O3 (O3Strans_mdl): chi_O3s_EESC, Gamma_O3s
CHI_O3S_RFEF = {
    "AMTRAC": (-0.0137, -0.743034), "CCSR-NIES": (-0.0079, 0.69275606),
    "CMAM": (-0.0081, -4.8533874), "CNRM-ACM": (-0.0232, 0.33077526),
    "LMDZrepro": (-0.0106, 3.082569), "MRI": (-0.0203, -1.9066534),
    "Niwa-SOCOL": (-0.0052, 6.874074), "SOCOL": (-0.0105, 3.0497866),
    "ULAQ": (-0.0154, 2.234234), "UMSLIMCAT": (-0.0106, 0.29934123),
    "UMUKCA-UCAM": (-0.0115, 1.7112298),
}
# Stratospheric O3 vs N2O (O3Strans_mdl, O3Snitrous_mdl): chi_O3s_N2O, EESC_x
CHI_O3S2_PARAM = {
    (m, "Daniel2010-sat"): (-0.0, INF) for m in CHI_O3S_RFEF
} | {
    ("AMTRAC", "Daniel2010-lin"): (-0.06730806, INF),
    ("CCSR-NIES", "Daniel2010-lin"): (-0.03881268, INF),
    ("CMAM", "Daniel2010-lin"): (-0.039795283, INF),
    ("CNRM-ACM", "Daniel2010-lin"): (-0.11398154, INF),
    ("LMDZrepro", "Daniel2010-lin"): (-0.052077774, INF),
    ("MRI", "Daniel2010-lin"): (-0.09973385, INF),
    ("Niwa-SOCOL", "Daniel2010-lin"): (-0.025547586, INF),
    ("SOCOL", "Daniel2010-lin"): (-0.051586475, INF),
    ("ULAQ", "Daniel2010-lin"): (-0.07566016, INF),
    ("UMSLIMCAT", "Daniel2010-lin"): (-0.052077774, INF),
    ("UMUKCA-UCAM", "Daniel2010-lin"): (-0.056499474, INF),
}

# CH4 OH lifetime (OHSNK_mdl)
TAU_CH4_OH0 = {
    "Prather2012": 11.2000, "CESM-CAM-superfast": 9.6990, "CICERO-OsloCTM2": 11.5464,
    "CMAM": 10.8536, "EMAC": 10.5072, "GEOSCCM": 11.0845, "GDFL-AM3": 10.8536,
    "GISS-E2-R": 12.2392, "GISS-E2-R-TOMAS": 10.6227, "HadGEM2": 13.3938,
    "LMDzORINCA": 12.1237, "MIROC-CHEM": 10.0454, "MOCAGE": 8.1979,
    "NCAR-CAM-35": 10.6227, "STOC-HadAM3": 10.5072, "TM5": 11.4309, "UM-CAM": 16.1649,
}
# OH sensitivities (OHSNKtrans_mdl): OH_Tatm, OH_Qatm, OH_O3, OH_CH4
OHSNK_SENS = {
    "Holmes2013":  (3.0, 0.32, -0.55, -0.31),
    "UCI-CTM":     (3.9, 0.32, -0.66, -0.363),
    "Oslo-CTM3":   (2.8, 0.29, -0.43, -0.307),
    "GEOS-Chem":   (2.2, 0.34, -0.61, -0.274),
    "mean-OxComp": (0., 0., 0., -0.32),
}
# OH sensitivities to ozone precursors (OHSNKtrans_mdl, OHSNKfct_mdl): NOX, CO, VOC
OHSNK_OZ_SENS = {
    ("Holmes2013", "log"):  (-0.14, 0.06, 0.04),
    ("UCI-CTM", "log"):     (-0.15, 0.066, 0.04),
    ("Oslo-CTM3", "log"):   (-0.10, 0.05, 0.04),
    ("GEOS-Chem", "log"):   (-0.16, 0.065, 0.04),
    ("mean-OxComp", "log"): (-0.137, 0.11, 0.047),
    ("Holmes2013", "lin"):  (0.0042 * (-0.14) / (-0.137), -1.05E-4 * 0.06 / 0.11 * 28 / 12., -3.14E-4 * 0.04 / 0.047),
    ("UCI-CTM", "lin"):     (0.0042 * (-0.15) / (-0.137), -1.05E-4 * 0.066 / 0.11 * 28 / 12., -3.14E-4 * 0.04 / 0.047),
    ("Oslo-CTM3", "lin"):   (0.0042 * (-0.10) / (-0.137), -1.05E-4 * 0.05 / 0.11 * 28 / 12., -3.14E-4 * 0.04 / 0.047),
    ("GEOS-Chem", "lin"):   (0.0042 * (-0.16) / (-0.137), -1.05E-4 * 0.065 / 0.11 * 28 / 12., -3.14E-4 * 0.04 / 0.047),
    ("mean-OxComp", "lin"): (0.0042, -1.05E-4 * 28 / 12., -3.14E-4),
}

# N2O photolysis lifetime (HVSNK_mdl)
TAU_N2O_HV = {
    "Prather2015": 123.,
    "GMI": 137.4 * 123. / 132.5,
    "GEOSCCM": 120.2 * 123. / 132.5,
    "G2d-M": 127.0 * 123. / 132.5,
    "G2d": 129.5 * 123. / 132.5,
    "Oslo-c29": 126.1 * 123. / 132.5,
    "Oslo-c36": 146.7 * 123. / 132.5,
    "UCI-c29": 126.2 * 123. / 132.5,
    "UCI-c36": 146.2 * 123. / 132.5,
}
# Photolysis sensitivities (HVSNKtrans_mdl): hv_N2O, hv_EESC, hv_age
EESC_HV, EESC_HV0 = 1729.4532, 420.0
HVSNK_SENS = {
    "Prather2012": (0.08, 0, 0),
    "Prather2015": (0.065, 0.04 / np.log(EESC_HV / EESC_HV0), 0),
    "G2d":         (0.018 / np.log(321 / 270.), 0.048 / np.log(EESC_HV / EESC_HV0), 0.011 / np.log(4.0 / 4.5)),
    "Oslo-c29":    (0.010 / np.log(321 / 270.), 0.033 / np.log(EESC_HV / EESC_HV0), 0),
    "UCI-c29":     (0.012 / np.log(321 / 270.), 0.029 / np.log(EESC_HV / EESC_HV0), 0),
}

# OC -> POM conversion (POAconv_mdl)
ALPHA_POM_PARAM = {"default": 1.4, "GFDL": 1.6, "CSIRO": 1.3}
# Aerosol load tables
SO4_PARAM = {   # tau_SO2, tau_DMS, Gamma_SO4
    "CSIRO-Mk360": (0.010229539, 0.0022402848, 0.037603702),
    "GFDL-AM3":    (0.0058695273, 0.078276515, 0.11960271),
    "GISS-E2-R":   (0.0031671403, 0.011171886, 0.000894944),
    "MIROC-CHEM":  (0.0039869417, 0.039507672, 0.051096383),
}
POA_PARAM = {   # tau_OMff, tau_OMbb, Gamma_POA
    "CSIRO-Mk360": (0.02991815, 0.0011926149, -0.01278927),
    "GFDL-AM3":    (0.041511822, 0.037062675, 0.09961289),
    "GISS-E2-R":   (0.023955109, 0.011572403, 0.03176226),
    "MIROC-CHEM":  (0.017742313, 0.015973175, 0.061647136),
}
BC_PARAM = {    # tau_BCff, tau_BCbb, Gamma_BC
    "CSIRO-Mk360": (0.025810905, 0.013132318, 0.0028200152),
    "GFDL-AM3":    (0.014927125, 0.026311196, 0.008938937),
    "GISS-E2-R":   (0.016818512, 0.0131583065, 0.0040882262),
    "MIROC-CHEM":  (0.013574581, 0.021029513, 0.0059852498),
}
NO3_PARAM = {   # tau_NOX, tau_NH3, Gamma_NO3
    "Bellouin2011":     (0.00038767335, 0.0020994716, 0.0014586062),
    "Hauglustaine2014": (0.0006427187, 0.0007864764, 0.0),
}
SOA_PARAM = {   # tau_VOC, tau_BVOC, Gamma_SOA
    "GFDL-AM3":  (5.667277e-05, 0.013705156, 0.011087768),
    "GISS-E2-R": (0.002873391, 8.02826e-08, 0.001463341),
}
DUST_PARAM = {  # tau_DUST, Gamma_DUST
    "CSIRO-Mk360": (0.00980371, 1.1433227),
    "GFDL-AM3":    (0.010827399, 0.35639888),
    "GISS-E2-R":   (0.01173677, -0.5630213),
    "MIROC-CHEM":  (0.004776219, 0.045209166),
}
SALT_PARAM = {  # tau_SALT, Gamma_SALT
    "GFDL-AM3":   (0.0005241309, -0.04872044),
    "GISS-E2-R":  (0.0034472656, 5.2246094e-05),
    "MIROC-CHEM": (0.0006817889, 0.0048052827),
}
SOLUB_PARAM = { # solub_SO4, _POA, _BC, _NO3, _SOA, _DUST, _SALT
    "Hansen2005":   (1., 0.8, 0.7, 1., 0.8, 0., 1.),
    "Lamarque2011": (1., 0.86, 0.80, 1., 1., 0.12, 0.05),
}




# =============================================================================
# Parameters
# =============================================================================

def _halo_col(name, missing):
    j = _HCOLS.index(name)
    return np.array([missing if HALO_TABLE[g][j] is None else HALO_TABLE[g][j] for g in HALO])




@dataclass(frozen=True)
class ChemParams:
    """All chemistry parameters (model names); build with load_chem_params()."""
    # baselines (--ch4rfc)
    co2_init: float
    ch4_init: float
    n2o_init: float
    co2_0: float
    ch4_0: float
    n2o_0: float
    d_ch4_1850: float
    d_n2o_1850: float
    # CH4
    alpha_ch4: float
    scale_oh: float
    tau_ch4_oh: float
    tau_ch4_hv: float
    tau_ch4_soil: float
    tau_ch4_ocean: float
    # OH
    oh_sink_form: str
    chi_oh_tatm: float
    chi_oh_qatm: float
    chi_oh_o3: float
    chi_oh_ch4: float
    chi_oh_nox: float
    chi_oh_co: float
    chi_oh_voc: float
    k_tatm: float
    k_qatm: float
    tatm_0: float
    k_svp: float
    t_svp: float
    o3s_0: float
    e_nox_oh: float
    e_co_oh: float
    e_voc_oh: float
    # N2O and photolysis
    alpha_n2o: float
    tau_n2o_hv: float          # also named chi_tau_N2O_hv
    chi_hv_n2o: float
    chi_hv_eesc: float
    chi_hv_age: float
    gamma_age: float
    tau_lag: float
    # ozone
    chi_o3t_ch4: float
    chi_o3t_co: float
    chi_o3t_voc: float
    chi_o3t_nox: float
    gamma_o3t: float
    chi_o3s_eesc: float
    gamma_o3s: float
    chi_o3s_n2o: float
    eesc_x: float
    alpha_br: float
    # 1850 anomalies of the diagnostics
    d_o3t_1850: float
    d_o3s_1850: float
    d_so4_1850: float
    d_poa_1850: float
    d_bc_1850: float
    d_no3_1850: float
    d_soa_1850: float
    # aerosols
    alpha_so4: float
    alpha_no3: float
    alpha_pom: float
    scale_pom: float
    scale_bc: float
    tau_so2: float
    tau_dms: float
    gamma_so4: float
    tau_om_ff: float
    tau_om_bb: float
    gamma_poa: float
    tau_bc_ff: float
    tau_bc_bb: float
    gamma_bc: float
    tau_nox: float
    tau_nh3: float
    gamma_no3: float
    tau_voc: float
    tau_bvoc: float
    gamma_soa: float
    tau_dust: float
    gamma_dust: float
    tau_salt: float
    gamma_salt: float
    solub_so4: float
    solub_poa: float
    solub_bc: float
    solub_no3: float
    solub_soa: float
    solub_dust: float
    solub_salt: float
    # biomass-burning and wetland anomalies (all 0)
    d_ebb_ch4: float
    d_ebb_n2o: float
    d_ebb_nh3: float
    d_e_wet: float
    d_ebb_co: float
    d_ebb_so2: float
    d_ebb_oc: float
    d_ebb_bc: float
    d_ebb_nox: float
    d_ebb_voc: float
    # halogens, HALO order
    alpha_halo: np.ndarray
    tau_halo_oh: np.ndarray
    tau_halo_hv: np.ndarray
    tau_halo_other: np.ndarray
    tau_halo_total: np.ndarray
    halo_0: np.ndarray
    f_fracrel: np.ndarray      # 0 outside halo_ods
    n_cl: np.ndarray
    n_br: np.ndarray
    gwp100: np.ndarray         # def_gwp100_oa2022_ems
    eesc_0: float
    dt: float = DT

    @property
    def eesc_weight(self):
        """f_fracrel * (n_Cl + alpha_Br * n_Br), the EESC weight of each gas."""
        return self.f_fracrel * (self.n_cl + self.alpha_br * self.n_br)


def load_chem_params(cfg=None):
    """Chemistry parameters of a run configuration."""
    cfg = cfg or RunConfig()
    return _chem_params(cfg.chem, cfg.ch4rfc, cfg.scen)


def _chem_params(opts=None, ch4rfc="etminan", scen="contrib"):
    """Chemistry parameters for one set of model choices."""
    o = opts or ChemOptions()
    b = pick(BASELINES, ch4rfc, "--ch4rfc")

    # scale_OH depends on the scenario
    scale_OH = 0.4 if scen in ("ssp1", "ssp2", "ssp3", "ssp4", "ssp5") else 0.64
    oh_T, oh_Q, oh_O3, oh_CH4 = pick(OHSNK_SENS, o.OHSNKtrans_mdl, "--OHSNKtrans_mdl")
    oh_NOX, oh_CO, oh_VOC = pick(OHSNK_OZ_SENS, (o.OHSNKtrans_mdl, o.OHSNKfct_mdl),
                                  "--OHSNKtrans_mdl/--OHSNKfct_mdl")
    hv_N2O, hv_EESC, hv_age = pick(HVSNK_SENS, o.HVSNKtrans_mdl, "--HVSNKtrans_mdl")
    o3t = [c * SCALE_O3T for c in pick(CHI_O3T_RFEF, o.O3Temis_mdl, "--O3Temis_mdl")]
    o3s_EESC, o3s_Gamma = pick(CHI_O3S_RFEF, o.O3Strans_mdl, "--O3Strans_mdl")
    o3s_N2O, eesc_x = pick(CHI_O3S2_PARAM, (o.O3Strans_mdl, o.O3Snitrous_mdl),
                            "--O3Strans_mdl/--O3Snitrous_mdl")
    so4 = pick(SO4_PARAM, o.SO4load_mdl, "--SO4load_mdl")
    poa = pick(POA_PARAM, o.POAload_mdl, "--POAload_mdl")
    bc = pick(BC_PARAM, o.BCload_mdl, "--BCload_mdl")
    no3 = pick(NO3_PARAM, o.NO3load_mdl, "--NO3load_mdl")
    soa = pick(SOA_PARAM, o.SOAload_mdl, "--SOAload_mdl")
    dust = pick(DUST_PARAM, o.DUSTload_mdl, "--DUSTload_mdl")
    salt = pick(SALT_PARAM, o.SALTload_mdl, "--SALTload_mdl")
    solub = pick(SOLUB_PARAM, o.solub_mdl, "--solub_mdl")

    # Halogens
    HALO_0 = np.zeros(len(HALO))
    for g, v in (("cf4", 34.05), ("ch2cl2", 6.9), ("ch3br", 5.3), ("ch3cl", 457), ("chcl3", 6.)):
        HALO_0[HALO.index(g)] = v                       # RCMIP3 values, 2024/07/20
    frac_tab = pick(FRACREL, o.O3Sfracrel_mdl, "--O3Sfracrel_mdl")
    ods = halo_mask(HALO_ODS)
    f_fracrel = np.where(ods, np.array([frac_tab.get(g, 0.0) for g in HALO]), 0.0)
    n_Cl, n_Br = _halo_col("n_cl", 0.0), _halo_col("n_br", 0.0)
    alpha_Br = 60.
    EESC_0 = float(np.sum((f_fracrel * (n_Cl + alpha_Br * n_Br) * HALO_0)[ods]))

    return ChemParams(
        **b,
        alpha_ch4=0.1765 * 12,
        scale_oh=scale_OH,
        tau_ch4_oh=scale_OH * pick(TAU_CH4_OH0, o.OHSNK_mdl, "--OHSNK_mdl"),
        tau_ch4_hv=1.06 * 120, tau_ch4_soil=150, tau_ch4_ocean=200,
        oh_sink_form=_check_fct(o.OHSNKfct_mdl),
        chi_oh_tatm=oh_T, chi_oh_qatm=oh_Q, chi_oh_o3=oh_O3, chi_oh_ch4=oh_CH4,
        chi_oh_nox=oh_NOX, chi_oh_co=oh_CO, chi_oh_voc=oh_VOC,
        k_tatm=0.94, k_qatm=1.5, tatm_0=251., k_svp=17.67, t_svp=243.5 - 273.15,
        o3s_0=280., e_nox_oh=13., e_co_oh=180., e_voc_oh=39. + 220. + 175,
        alpha_n2o=0.1765 * 28.0,
        tau_n2o_hv=pick(TAU_N2O_HV, o.HVSNK_mdl, "--HVSNK_mdl"),
        chi_hv_n2o=hv_N2O, chi_hv_eesc=hv_EESC, chi_hv_age=hv_age,
        gamma_age=pick(CHM_GAMMA_AGE, o.HVSNKcirc_mdl, "--HVSNKcirc_mdl"),
        tau_lag=3.,
        chi_o3t_ch4=o3t[0], chi_o3t_co=o3t[1], chi_o3t_voc=o3t[2], chi_o3t_nox=o3t[3],
        gamma_o3t=pick(CHM_GAMMA_O3T, o.O3Tclim_mdl, "--O3Tclim_mdl"),
        chi_o3s_eesc=o3s_EESC, gamma_o3s=o3s_Gamma, chi_o3s_n2o=o3s_N2O, eesc_x=eesc_x,
        alpha_br=alpha_Br,
        d_o3t_1850=0., d_o3s_1850=0., d_so4_1850=0., d_poa_1850=0., d_bc_1850=0.,
        d_no3_1850=0., d_soa_1850=0.,
        alpha_so4=96 / 32., alpha_no3=62 / 14.,
        alpha_pom=pick(ALPHA_POM_PARAM, o.POAconv_mdl, "--POAconv_mdl"),
        scale_pom=0.09 / 0.292, scale_bc=1.,
        tau_so2=so4[0], tau_dms=so4[1], gamma_so4=so4[2],
        tau_om_ff=poa[0], tau_om_bb=poa[1], gamma_poa=poa[2],
        tau_bc_ff=bc[0], tau_bc_bb=bc[1], gamma_bc=bc[2],
        tau_nox=no3[0], tau_nh3=no3[1], gamma_no3=no3[2],
        tau_voc=soa[0], tau_bvoc=soa[1], gamma_soa=soa[2],
        tau_dust=dust[0], gamma_dust=dust[1], tau_salt=salt[0], gamma_salt=salt[1],
        solub_so4=solub[0], solub_poa=solub[1], solub_bc=solub[2], solub_no3=solub[3],
        solub_soa=solub[4], solub_dust=solub[5], solub_salt=solub[6],
        d_ebb_ch4=0., d_ebb_n2o=0., d_ebb_nh3=0., d_e_wet=0., d_ebb_co=0.,
        d_ebb_so2=0., d_ebb_oc=0., d_ebb_bc=0., d_ebb_nox=0., d_ebb_voc=0.,
        alpha_halo=_halo_col("molar_mass", 0.0) * 0.1765,
        tau_halo_oh=_halo_col("tau_OH", INF),
        tau_halo_hv=_halo_col("tau_hv_raw", INF) * 1.06,
        tau_halo_other=_halo_col("tau_othr", INF),
        tau_halo_total=_halo_col("tau_total", INF),
        halo_0=HALO_0, f_fracrel=f_fracrel,
        n_cl=n_Cl, n_br=n_Br, gwp100=_halo_col("gwp100", 0.0),
        eesc_0=EESC_0,
    )


def _check_fct(fct):
    if fct not in ("log", "lin"):
        raise ValueError(f"unknown --OHSNKfct_mdl={fct!r}; options: ['log', 'lin']")
    return fct


# GWP-weighted reporting sums: name -> HALO mask
GWP_SUMS = {"CHFC": halo_mask(HALO_HFC), "CPFC": halo_mask(HALO_PFC), "CODS": halo_mask(HALO_ODS),
            "CCFC": halo_mask(HALO_CFC), "CFGS": halo_mask(FHALO), "CMNT": halo_mask(MHALO),
            "CHALO": halo_mask(HALO)}

# =============================================================================
# OH and photolysis functions
# =============================================================================

def _kOH_exponent(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC):
    """Exponent of f_kOH; NOx/CO/VOC enter as log(1+E/E_oh) or linearly (--OHSNKfct_mdl)."""
    if p.oh_sink_form == "log":
        precursors = (p.chi_oh_nox * np.log(1 + ENOX / p.e_nox_oh)
                      + p.chi_oh_co * np.log(1 + ECO / p.e_co_oh)
                      + p.chi_oh_voc * np.log(1 + EVOC / p.e_voc_oh))
    else:
        precursors = p.chi_oh_nox * ENOX + p.chi_oh_co * ECO + p.chi_oh_voc * EVOC
    return (p.chi_oh_ch4 * np.log(1 + D_CH4 / p.ch4_0)
            + p.chi_oh_o3 * np.log(1 + D_O3s / p.o3s_0)
            + precursors
            + p.chi_oh_tatm * np.log(1 + p.k_tatm * T / p.tatm_0)
            + p.chi_oh_qatm * np.log(1 + p.k_qatm * (np.exp(p.k_svp * p.k_tatm * T
                                                             / (p.tatm_0 + p.t_svp)) - 1)))


def f_kOH(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC):
    """Relative change of OH."""
    return np.exp(_kOH_exponent(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC)) - 1


def df_kOH(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC, ref_compat=True):
    """Partial derivatives of f_kOH, keyed CH4, O3s, gst, ENOX, ECO, EVOC.

    ref_compat : the gst entry as in the reference results (see below); False gives the
                 true derivative. Used by kOH_attribution only (no model output).
    """
    e = np.exp(_kOH_exponent(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC))
    a = p.k_svp * p.k_tatm / (p.tatm_0 + p.t_svp)
    d = {
        "CH4": p.chi_oh_ch4 / (p.ch4_0 + D_CH4) * e,
        "O3s": p.chi_oh_o3 / (p.o3s_0 + D_O3s) * e,
        # Reference behaviour (ref_compat): df_kOH_dgst is NOT the true derivative: it
        # uses chi_OH_Tatm where the humidity term needs chi_OH_Qatm, and drops the
        # direct-temperature term chi_OH_Tatm*k_Tatm/(Tatm_0+k_Tatm*T).
        "gst": ((p.chi_oh_tatm * p.k_qatm * a * np.exp(a * T)
                 / (1 + p.k_qatm * (np.exp(a * T) - 1)) * e) if ref_compat else
                ((p.chi_oh_tatm * p.k_tatm / (p.tatm_0 + p.k_tatm * T)
                  + p.chi_oh_qatm * p.k_qatm * a * np.exp(a * T)
                  / (1 + p.k_qatm * (np.exp(a * T) - 1))) * e)),
    }
    if p.oh_sink_form == "log":
        d |= {"ENOX": p.chi_oh_nox / (p.e_nox_oh + ENOX) * e,
              "ECO": p.chi_oh_co / (p.e_co_oh + ECO) * e,
              "EVOC": p.chi_oh_voc / (p.e_voc_oh + EVOC) * e}
    else:
        d |= {"ENOX": p.chi_oh_nox * e, "ECO": p.chi_oh_co * e, "EVOC": p.chi_oh_voc * e}
    return d


def kOH_attribution(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC, ref_compat=True):
    """df_kOH_1..6 (derivative x driver), df_kOH_tot and df_kOH_pct_1..6, as dicts."""
    d = df_kOH(p, D_CH4, D_O3s, T, ENOX, ECO, EVOC, ref_compat)
    x = {"CH4": D_CH4, "O3s": D_O3s, "gst": T, "ENOX": ENOX, "ECO": ECO, "EVOC": EVOC}
    contrib = {k: d[k] * x[k] for k in d}
    tot = sum(contrib.values())
    return contrib, tot, {k: v / tot for k, v in contrib.items()}


def f_hv(p, D_N2O, D_EESC, T):
    """Relative change of stratospheric photolysis."""
    return np.exp(p.chi_hv_n2o * np.log(1 + D_N2O / p.n2o_0)
                  + p.chi_hv_eesc * np.log(1 + D_EESC / p.eesc_0)
                  - p.chi_hv_age * np.log(1 + p.gamma_age * T)) - 1


def df_hv(p, D_N2O, D_EESC, T, ref_compat=True):
    """Partial derivatives of f_hv (df_hv_d*), keyed N2O, EESC, gst (ref_compat: the gst
    entry as in the reference results, see below; False gives the true derivative)."""
    e = f_hv(p, D_N2O, D_EESC, T) + 1
    sign = 1.0 if ref_compat else -1.0
    return {"N2O": p.chi_hv_n2o / (p.n2o_0 + D_N2O) * e,
            "EESC": p.chi_hv_eesc / (p.eesc_0 + D_EESC) * e,
            # Reference behaviour (ref_compat): df_hv_dgst has the opposite sign of the
            # true derivative (f_hv uses -chi_hv_age*log(1+gamma_age*T)).
            "gst": sign * p.chi_hv_age * p.gamma_age / (1 + p.gamma_age * T) * e}


def hv_attribution(p, D_N2O, D_EESC, T, ref_compat=True):
    """df_hv_1..3, df_hv_tot and df_hv_pct_1..3, as dicts."""
    d = df_hv(p, D_N2O, D_EESC, T, ref_compat)
    x = {"N2O": D_N2O, "EESC": D_EESC, "gst": T}
    contrib = {k: d[k] * x[k] for k in d}
    tot = sum(contrib.values())
    return contrib, tot, {k: v / tot for k, v in contrib.items()}


# =============================================================================
# Time stepping
# =============================================================================

@dataclass
class ChemStep:
    """Chemistry at one step, named as the chm_* variables (HALO arrays in HALO order)."""
    D_CH4: float
    D_N2O: float
    D_HALO: np.ndarray
    D_CH4_lag: float
    D_N2O_lag: float
    D_HALO_lag: np.ndarray
    D_EESC: float
    D_O3t: float
    D_O3s: float
    D_SO4: float
    D_POA: float
    D_BC: float
    D_NO3: float
    D_SOA: float
    D_DUST: float
    D_SALT: float
    D_AERh: float
    # filled by chemistry_finish once tatm(n) is known
    D_kOH: float = np.nan
    D_hv: float = np.nan
    D_OHSNK_CH4: float = np.nan
    D_HVSNK_CH4: float = np.nan
    D_XSNK_CH4: float = np.nan
    D_HVSNK_N2O: float = np.nan
    D_OHSNK_HALO: np.ndarray = None
    D_HVSNK_HALO: np.ndarray = None
    D_XSNK_HALO: np.ndarray = None
    emeth: float = np.nan


def _ozone_precursors(p, e):
    """Annual NOx, CO (as C) and VOC emission rates used by kOH and O3t."""
    dt = p.dt
    return ((e("nox") + p.d_ebb_nox) / dt,
            (e("co") / 28 * 12 + p.d_ebb_co) / dt,
            (e("voc") + p.d_ebb_voc) / dt)


@dataclass(frozen=True)
class ChemDrivers:
    """Inputs of the first part of a chemistry step n (chemistry_step)."""
    ems_row: object             # dat_ems(n, gas), all gases, per step
    ems_1850: object            # dat_ems("1850", gas)
    gi: object                  # gas name -> column index
    halo_idx: object            # columns of the HALO gases
    tatm_prev: float            # scm_D_tatm(n-1) (ignored at the first step)
    prescribed: object = None   # (D_CH4, D_N2O, D_HALO, halo_mask) of step n, or None


@dataclass(frozen=True)
class ChemFinishDrivers:
    """Inputs of the second part of a chemistry step n (chemistry_finish)."""
    ems_row: object             # dat_ems(n, gas)
    gi: object                  # gas name -> column index
    tatm: float                 # scm_D_tatm(n), from DOECLIM at the same step


def chemistry_step(p, prev, drivers):
    """Chemistry stocks and diagnostics of step n (everything not needing tatm(n));
    chemistry_finish completes the step once tatm(n) is known.

    prev       : ChemStep of n-1 (finished), or None at the first step
    drivers    : ChemDrivers of step n:
    ems_row    : dat_ems(n, gas) for all gases, per step
    ems_1850   : dat_ems("1850", gas)
    gi         : gas name -> column index; halo_idx : columns of the HALO gases
    tatm_prev  : scm_D_tatm(n-1) (ignored at the first step)
    prescribed : (D_CH4, D_N2O, D_HALO (48,), halo_mask) of step n replacing the stock
                 equations (--load_conc, 1850 included); any entry may be None to
                 keep the computed value
    """
    ems_row, ems_1850, gi, halo_idx, tatm_prev, prescribed = (
        drivers.ems_row, drivers.ems_1850, drivers.gi, drivers.halo_idx, drivers.tatm_prev,
        drivers.prescribed)
    def e(g):
        return ems_row[gi(g)]

    def e0(g):
        return ems_1850[gi(g)]

    dt = p.dt
    first = prev is None
    if first:
        D_CH4, D_N2O, D_HALO = p.d_ch4_1850, p.d_n2o_1850, np.zeros(len(halo_idx))
        CH4_lag, N2O_lag, HALO_lag = p.d_ch4_1850, p.d_n2o_1850, np.zeros(len(halo_idx))
    else:
        q = prev
        D_CH4 = q.D_CH4 + (1 / p.alpha_ch4) * (
            e("ch4") / 16 * 12 + p.d_ebb_ch4 + p.d_e_wet
            + (q.D_OHSNK_CH4 + q.D_HVSNK_CH4 + q.D_XSNK_CH4) * dt)
        D_N2O = q.D_N2O + (1 / p.alpha_n2o) * (e("n2o") + p.d_ebb_n2o + q.D_HVSNK_N2O * dt)
        D_HALO = q.D_HALO + (1 / p.alpha_halo) * (
            ems_row[halo_idx] - ems_1850[halo_idx]
            + (q.D_OHSNK_HALO + q.D_HVSNK_HALO + q.D_XSNK_HALO) * dt)
        CH4_lag = q.D_CH4_lag + ((1 / p.tau_lag) * q.D_CH4 - (1 / p.tau_lag) * q.D_CH4_lag) * dt
        N2O_lag = q.D_N2O_lag + ((1 / p.tau_lag) * q.D_N2O - (1 / p.tau_lag) * q.D_N2O_lag) * dt
        HALO_lag = q.D_HALO_lag + ((1 / p.tau_lag) * q.D_HALO - (1 / p.tau_lag) * q.D_HALO_lag) * dt

    if prescribed is not None:              # None entries keep the computed value
        p_ch4, p_n2o, halo_c, mask = prescribed
        D_CH4 = D_CH4 if p_ch4 is None else p_ch4
        D_N2O = D_N2O if p_n2o is None else p_n2o
        if halo_c is not None:
            D_HALO = np.where(mask, halo_c, D_HALO)

    EESC = float(p.eesc_weight @ HALO_lag)

    if first:
        O3t, O3s = p.d_o3t_1850, p.d_o3s_1850
        SO4, POA, BC = p.d_so4_1850, p.d_poa_1850 * p.scale_pom, p.scale_bc * p.d_bc_1850
        NO3, SOA, DUST, SALT = p.d_no3_1850, p.d_soa_1850, 0.0, 0.0
    else:
        q, T = prev, tatm_prev
        ENOX, ECO, EVOC = _ozone_precursors(p, e)
        O3t = (p.chi_o3t_ch4 * np.log(1 + q.D_CH4 / p.ch4_0) + p.gamma_o3t * T
               + p.chi_o3t_nox * ENOX + p.chi_o3t_co * ECO + p.chi_o3t_voc * EVOC)
        O3s = (p.chi_o3s_eesc * q.D_EESC
               + p.chi_o3s_n2o * q.D_N2O_lag * (1 - q.D_EESC / p.eesc_x)
               + p.gamma_o3s * T)
        # Each aerosol is exactly 0 when its emission is <= 0
        SO4 = (p.alpha_so4 * p.tau_so2 * (e("so2") + p.d_ebb_so2 - e0("so2")) / dt
               + p.alpha_so4 * p.tau_dms * 0 + p.gamma_so4 * T) if e("so2") > 0 else 0.0
        POA = ((p.tau_om_ff * p.alpha_pom * (e("oc") - e0("oc")) / dt
                + p.tau_om_bb * p.alpha_pom * p.d_ebb_oc / dt + p.gamma_poa * T)
               if e("oc") > 0 else 0.0) * p.scale_pom
        BC = (p.scale_bc * (p.tau_bc_ff * (e("bc") - e0("bc")) / dt
                            + p.tau_bc_bb * p.d_ebb_bc / dt + p.gamma_bc * T)
              if e("bc") > 0 else 0.0)
        NO3 = (p.alpha_no3 * p.tau_nox * (e("nox") + p.d_ebb_nox - e0("nox")) / dt
               + p.alpha_no3 * p.tau_nh3 * (e("nh3") - e0("nh3") + p.d_ebb_nh3) / dt
               + p.gamma_no3 * T) if e("nox") > 0 else 0.0
        SOA = (p.tau_voc * (e("voc") + p.d_ebb_voc - e0("voc")) / dt
               + p.tau_bvoc * 0 + p.gamma_soa * T) if e("voc") > 0 else 0.0
        # DUST and SALT are switched off by a leading 0*
        DUST = 0 * (p.tau_dust * 0 + p.gamma_dust * T)
        SALT = 0 * (p.tau_salt * 0 + p.gamma_salt * T)
    AERh = (p.solub_so4 * SO4 + p.solub_poa * POA + p.solub_bc * BC + p.solub_no3 * NO3
            + p.solub_soa * SOA + p.solub_dust * DUST + p.solub_salt * SALT)

    return ChemStep(D_CH4=D_CH4, D_N2O=D_N2O, D_HALO=D_HALO,
                    D_CH4_lag=CH4_lag, D_N2O_lag=N2O_lag, D_HALO_lag=HALO_lag,
                    D_EESC=EESC, D_O3t=O3t, D_O3s=O3s,
                    D_SO4=SO4, D_POA=POA, D_BC=BC, D_NO3=NO3, D_SOA=SOA,
                    D_DUST=DUST, D_SALT=SALT, D_AERh=AERh)


def chemistry_finish(p, s, drivers):
    """Second part of chemistry step n, in place on s (from chemistry_step): kOH, hv,
    sinks, emeth; returns s.

    drivers : ChemFinishDrivers (tatm = scm_D_tatm(n), from DOECLIM at the same step).
    """
    ems_row, gi, tatm = drivers.ems_row, drivers.gi, drivers.tatm
    ENOX, ECO, EVOC = _ozone_precursors(p, lambda g: ems_row[gi(g)])
    s.D_kOH = f_kOH(p, s.D_CH4, s.D_O3s, tatm, ENOX, ECO, EVOC)
    s.D_hv = f_hv(p, s.D_N2O_lag, s.D_EESC, tatm)
    k, h = s.D_kOH, s.D_hv
    s.D_OHSNK_CH4 = -p.alpha_ch4 / p.tau_ch4_oh * (p.ch4_0 * k + s.D_CH4 + k * s.D_CH4)
    s.D_HVSNK_CH4 = -p.alpha_ch4 / p.tau_ch4_hv * (p.ch4_0 * h + s.D_CH4_lag + h * s.D_CH4_lag)
    s.D_XSNK_CH4 = -p.alpha_ch4 * (1 / p.tau_ch4_soil + 1 / p.tau_ch4_ocean) * s.D_CH4
    s.D_HVSNK_N2O = -p.alpha_n2o / p.tau_n2o_hv * (p.n2o_0 * h + s.D_N2O_lag + h * s.D_N2O_lag)
    a, H0 = p.alpha_halo, p.halo_0
    s.D_OHSNK_HALO = -a / p.tau_halo_oh * (H0 * k + s.D_HALO + k * s.D_HALO)
    s.D_HVSNK_HALO = -a / p.tau_halo_hv * (H0 * h + s.D_HALO_lag + h * s.D_HALO_lag)
    s.D_XSNK_HALO = -a / p.tau_halo_other * s.D_HALO
    s.emeth = CH4_FOSSFUELFRACTION * CH4_OXIDATION_COEFF * s.D_CH4 / p.tau_ch4_oh * p.dt
    return s


def gwp_sums(p, D_HALO):
    """chm_CHFC ... chm_CHALO: GWP100-weighted halogen burdens x 1e-6 (reporting only)."""
    c = (p.halo_0 + D_HALO) * p.gwp100
    return {name: float(c[m].sum() * 1e-6) for name, m in GWP_SUMS.items()}


def run_chemistry(p, ems, ems_1850, gi, halo_idx, tatm, conc=None):
    """Run chemistry offline over a prescribed scm_D_tatm series.

    conc : inputs.PrescribedConc for --load_conc, or None.
    Returns {variable name without the chm_ prefix: array}; HALO variables are (N, 48).
    """
    steps, prev = [], None
    for i in range(len(tatm)):
        presc = None if conc is None else (conc.D_CH4[i], conc.D_N2O[i], conc.D_HALO[i],
                                           conc.halo_mask)
        s = chemistry_step(p, prev, ChemDrivers(ems[i], ems_1850, gi, halo_idx,
                                                tatm[i - 1] if i else 0.0, presc))
        prev = chemistry_finish(p, s, ChemFinishDrivers(ems[i], gi, tatm[i]))
        steps.append(prev)
    out = {f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(ChemStep)}
    sums = [gwp_sums(p, s.D_HALO) for s in steps]
    out.update({name: np.array([d[name] for d in sums]) for name in GWP_SUMS})
    return out
