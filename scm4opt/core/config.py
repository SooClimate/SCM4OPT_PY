# SPDX-License-Identifier: GPL-3.0-or-later
"""Run configuration: the model's --name=value options.

Defaults are the standard configuration for --scen=contrib.
"""

import dataclasses
import os
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

# Data locations
#   PACKAGE_DATA_DIR  <package>/data/: in the exported repository the one data folder
#                     (parameter tables shipped with the package + scenario inputs);
#                     absent in the development tree
#   DAT_DIR           where inputs are read: the SCM4OPT_DATA environment variable, else
#                     dat/ next to the package if it exists (development tree),
#                     else PACKAGE_DATA_DIR (exported repository)
PACKAGE_ROOT = Path(str(files(__package__.split(".")[0])))
PACKAGE_DATA_DIR = PACKAGE_ROOT / "data"
REPO_DIR = PACKAGE_ROOT.parent
DAT_DIR = Path(os.environ.get("SCM4OPT_DATA")
               or (REPO_DIR / "dat" if (REPO_DIR / "dat").is_dir() else PACKAGE_DATA_DIR))
TRUNK_DIR = REPO_DIR                      # former name, kept for old scripts


def n_steps(cfg):
    """Number of model steps (1/6 yr) of a configuration, first_year..last_year inclusive
    (after the scenario's own horizon is applied, see resolve)."""
    cfg = resolve(cfg)
    return (cfg.last_year - cfg.first_year) * 6 + 1


def data_path(dat_dir, *parts):
    """dat_dir/<parts> if it exists, else the copy shipped in PACKAGE_DATA_DIR, else
    dat_dir/<parts> (so a missing file is reported under the user's data directory)."""
    p = Path(dat_dir).joinpath(*parts)
    if p.exists():
        return p
    q = PACKAGE_DATA_DIR.joinpath(*parts)
    return q if q.exists() else p


@dataclass(frozen=True)
class ChemOptions:
    """Chemistry model choices (the --<process>_mdl options).

    Field names are the option names (e.g. --OHSNKfct_mdl=lin); defaults are the
    standard configuration.
    """
    OHSNK_mdl: str = "CMAM"                 # CH4 OH lifetime (tau_CH4_OH0)
    OHSNKfct_mdl: str = "log"               # OH dependence on NOx/CO/VOC: log | lin
    OHSNKtrans_mdl: str = "Oslo-CTM3"       # OH sensitivities
    HVSNK_mdl: str = "G2d"                  # N2O photolysis lifetime
    HVSNKtrans_mdl: str = "G2d"             # photolysis sensitivities
    HVSNKcirc_mdl: str = "CAM-35"           # age-of-air sensitivity
    O3Temis_mdl: str = "mean-OxComp"        # tropospheric O3 vs emissions
    O3Tclim_mdl: str = "CESM-CAM-superfast" # tropospheric O3 vs climate
    O3Strans_mdl: str = "LMDZrepro"         # stratospheric O3
    O3Snitrous_mdl: str = "Daniel2010-lin"  # stratospheric O3 vs N2O
    O3Sfracrel_mdl: str = "Engel2017"       # fractional release for EESC
    POAconv_mdl: str = "default"            # OC -> POM conversion
    POAload_mdl: str = "CSIRO-Mk360"
    SO4load_mdl: str = "MIROC-CHEM"
    BCload_mdl: str = "MIROC-CHEM"
    NO3load_mdl: str = "Hauglustaine2014"
    SOAload_mdl: str = "GFDL-AM3"
    DUSTload_mdl: str = "CSIRO-Mk360"
    SALTload_mdl: str = "GFDL-AM3"
    solub_mdl: str = "Lamarque2011"         # aerosol solubility (cloud forcing)


@dataclass(frozen=True)
class OceanOptions:
    """Ocean model choices (OSCAR ocean; option names).

    Defaults are the standard configuration: BD-model surface box model, CO2SysPower
    carbonate chemistry, MPI-ESM-LR mixed-layer response and pco2 weights 0/0/1.
    v_osn_model, mod_OSNKchem, mod_OSNKtrans and the three pco2_w_* weights can all be
    set from the command line.
    """
    v_osn_model: str = "BD-model"       # surface box model: HILDA | BD-model | 2D-model | 3D-model
    mod_OSNKchem: str = "CO2SysPower"   # carbonate chemistry: CO2SysPower | CO2SysPade | (else) Poly
    mod_OSNKtrans: str = "MPI-ESM-LR"   # mixed-layer response when --cm6mdl is unset
    pco2_form: str | None = None        # --pco2_form=Poly|Pade|Power, overrides mod_OSNKchem
    pco2_w_poly: float | None = 0.0     # weights override both (None = not set); normalised
    pco2_w_pade: float | None = 0.0
    pco2_w_power: float | None = 1.0
    ocn_nsub: int = 4                   # Euler sub-steps per model step
    ocean_model: str = "osc"            # --ocean_model=osc (OSCAR ocean, default) |
                                        #   hector (Hector v1.0 ocean, ocean_hector.py)
    same_alk: bool = False              # --same_alk (hector): one alkalinity for all boxes
    hector_tcre_rates: bool = False     # hector, --scen=tcre|tcre2: recompute the k-rates
                                        #   from the tcre transports (default: keep them)


@dataclass(frozen=True)
class ForcingOptions:
    """Forcing model choices (the --<process>_mdl options; standard defaults)."""
    O3Tradeff_mdl: str = "IPCC-AR5"        # tropospheric O3 radiative efficiency
    O3Sradeff_mdl: str = "IPCC-AR4"        # stratospheric O3 radiative efficiency
    POAradeff_mdl: str = "mean-AeroCom2"
    SO4radeff_mdl: str = "mean-AeroCom2"
    BCradeff_mdl: str = "mean-AeroCom2"
    BCadjust_mdl: str = "CSIRO"            # BC semi-direct adjustment
    NO3radeff_mdl: str = "mean-AeroCom2"
    SOAradeff_mdl: str = "mean-AeroCom2"
    CLOUDerf_mdl: str = "mean-ACCMIP"      # aerosol-cloud interaction
    CLOUDpreind_mdl: str = "median"        # preindustrial hydrophilic aerosol: low | median | high
    ALBBCrf_mdl: str = "mean-ACCMIP"       # BC on snow
    ALBBCwarm_mdl: str = "median"          # BC-on-snow efficacy: low | median | high
    ALBLCwarm_mdl: str = "Jones2013"       # land-cover albedo efficacy
    PRECradfact_mdl: str = "Kvalevag2013"  # atmospheric absorption fractions (rfc_atm)
    ALBLCflux_mdl: str = "CERES"           # land-cover albedo: flux | albedo | cover datasets
    ALBLCalb_mdl: str = "GlobAlbedo"
    ALBLCcover_mdl: str = "ESA-CCI"


@dataclass(frozen=True)
class RunConfig:
    scen: str = "contrib"      # scenario driver: contrib | rcmip3 | contrib_rfc | tcre | tcre2
    psce: str = "SSP1_19"      # scenario name in the input file names
    psrc: str = "IAMC"         # scenario source
    hsrc: str = "COMB"         # historical source of the emission totals
    land: str | None = "LUC_BLUE"  # land-use CO2 source; None = keep co2l from the totals
    lcov: str = "LUH"          # land-cover source for the albedo forcing
    rgn: str = "WORLD"
    csec: str = "EMISS_IND"    # emission sector tag (used in the contrib_rfc file names)
    ems: str = "CO2"           # --ems: emission species tag of the contribution reports; unused otherwise

    cm6mdl: str | None = "NorESM2-MM"  # None = no --cm6mdl: MAGICC6 defaults
    rname: str | None = None   # RCMIP scenario name (--rname); switches a few model branches
    use_ncycle: bool = False   # --use_ncycle: N-cycle limitation of land NPP
    ch4rfc: str = "etminan"
    nat_grp: str | None = None # natural-forcing group; None = from rname
    # marginal-analysis period tmargin: at most one of these
    tproj: bool = False        # --tproj: 2023-2100
    thist: bool = False        # --thist: 1850-2022
    tyear: int | None = None   # --tyear=Y: the six steps of year Y
    byear: int | None = None   # --byear=Y: t.val < Y+1
    dyear: int | None = None   # --dyear=Y: Y+1 <= t.val < Y+11
    rdc_rfc: str | None = None # --rdc_rfc=rfc_volc|rfc_solar|rfc_DUST: zero it over tmargin
    rdc_lcc: str | None = None # --rdc_lcc=WORLD|<region>: hold 1850 land cover over tmargin
    zero_nature: bool = False  # --zero_nature: no volcanic, solar or dust forcing
    clim_sensi: float | None = None  # --clim_sensi: overrides DOECLIM's doe_t2co (ECS)
    use_ssp534: bool = False   # --use_ssp534: DOECLIM parameters from doe_gcm_est_ssp534.csv
    co2_only: bool = False     # --co2_only: DOECLIM driven by rfc_CO2 instead of erf_tot
    load_conc: str | None = None  # --load_conc=all|nco2: prescribed concentrations;
                                  # "all" includes CO2, "nco2" keeps CO2 emission-driven
    load_AGAGE: bool = False   # --load_AGAGE: overwrite prescribed halogens with AGAGE data
    load_NOAA: bool = False    # --load_NOAA: overwrite prescribed CO2/CH4/N2O/SF6 with NOAA data
    fix_conc: bool = False     # --fix_conc (contrib): CO2 concentration prescribed from
                               # dat/contrib/<hsrc>_<psrc>/conc/<psce>_WORLD.csv
    fol: str | None = None     # --fol: forcing-data folder for --scen=contrib_rfc
    case: str | None = None    # --case: ensemble member of the contrib_rfc forcing data
    minus_rfc: str | None = None  # --minus_rfc=rfc_X: scale component X by (1-fraction) over tmargin
    fraction: float | None = None # --fraction: cnb_fraction (default 0.001)
    rout3: str | None = None   # --rout3=<dir>: output folder of the report (report.py)
    ctag: str = "all"          # --ctag: tag in the report file name <rname>_<case>_<ctag>
    climate_model: str | None = None  # --climate_model: climate_model of the RCMIP3 rows
                                      # (default version.RCMIP_CLIMATE_MODEL)
    ref_compat: bool = True    # --ref_compat=0: corrected behaviour instead of the deliberate
                               # quirks kept for the SCM4OPT v4.1 reference results (README: Reference compatibility)
    rcmip_csv: bool = True     # with --rout3, also write the RCMIP3 template CSV
    report_csv: bool = True    # with --rout3, write the per-step report CSV (~27 MB per run)
    first_year: int = 1850
    last_year: int = 2100      # end of the --scen=contrib horizon
    chem: ChemOptions = field(default_factory=ChemOptions)
    ocean: OceanOptions = field(default_factory=OceanOptions)
    forcing: ForcingOptions = field(default_factory=ForcingOptions)
    dat_dir: Path = DAT_DIR


def nat_grp_of(cfg):
    """Natural-forcing group: explicit cfg.nat_grp, else from --rname."""
    if cfg.nat_grp is not None:
        return cfg.nat_grp
    from ..scenarios.rcmip3_data import NAT_GRP_BY_RNAME
    return NAT_GRP_BY_RNAME.get(cfg.rname, "default") if cfg.rname else "default"


# --scen=rcmip3 forces these options, whatever was passed
RCMIP3_FORCED = dict(psce="SSP2_45", lcov="LUH", hsrc="CEDS", psrc="IAMC", land="LUC_GCP")


def resolve(cfg):
    """The effective configuration: the rcmip3 overrides and time horizons."""
    import dataclasses
    if cfg.scen != "rcmip3":
        return cfg
    if not cfg.rname:
        raise ValueError("--scen=rcmip3 needs --rname")
    from ..scenarios.rcmip3_data import DEFAULT_LAST_YEAR, LAST_YEAR_BY_RNAME
    return dataclasses.replace(cfg, **RCMIP3_FORCED, first_year=1850,
                               last_year=LAST_YEAR_BY_RNAME.get(cfg.rname, DEFAULT_LAST_YEAR))


def config_from_flags(flags):
    """RunConfig from --name=value flags (RunConfig or ChemOptions fields)."""
    run, chem, ocean, forcing = {}, {}, {}, {}
    forcing_names = {f.name for f in dataclasses.fields(ForcingOptions)}
    run_names = {f.name for f in dataclasses.fields(RunConfig)}
    chem_names = {f.name for f in dataclasses.fields(ChemOptions)}
    ocean_names = {f.name for f in dataclasses.fields(OceanOptions)}
    for flag in flags:
        if not flag.startswith("--") or "=" not in flag:
            sys.exit(f"bad flag {flag!r}; use --name=value")
        name, value = flag[2:].split("=", 1)
        if name in chem_names:
            chem[name] = value
        elif name in forcing_names:
            forcing[name] = value
        elif name in ocean_names:
            if name == "ocn_nsub":
                value = int(value)
            elif name in ("same_alk", "hector_tcre_rates"):
                value = value not in ("0", "")
            elif name.startswith("pco2_w_"):
                value = None if value.lower() == "none" else float(value)
            ocean[name] = value
        elif name in run_names:
            if name in ("first_year", "last_year", "tyear", "byear", "dyear"):
                value = int(value)
            elif name in ("clim_sensi", "fraction"):
                value = float(value)
            elif name == "dat_dir":
                from pathlib import Path
                value = Path(value)
            elif name in ("use_ncycle", "tproj", "thist", "zero_nature", "use_ssp534", "co2_only",
                          "load_AGAGE", "load_NOAA", "fix_conc", "rcmip_csv", "report_csv",
                          "ref_compat",
                          ):
                value = value not in ("0", "")
            elif name == "cm6mdl" and value.lower() == "none":
                value = None
            run[name] = value
        else:
            sys.exit(f"unknown flag --{name}")
    return RunConfig(**run, chem=ChemOptions(**chem), ocean=OceanOptions(**ocean),
                     forcing=ForcingOptions(**forcing))
