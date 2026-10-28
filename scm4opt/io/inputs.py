# SPDX-License-Identifier: GPL-3.0-or-later
"""Scenario inputs for the forward model (the data layer).

Builds, for the configuration in config.RunConfig:
  - the time set t
  - industrial and land-use emissions (dat_eind / dat_elnd / dat_ems)
  - natural forcing (dat_rf_solar / _volcanic / _con)
  - land cover (land_cover_WORLD)

Emissions are kept PER MODEL STEP (1/6 yr), exactly as stored in the input
tables, even though their unit labels say "/yr". Divide by scm_dt only where the
model equations do.

All inputs are CSV files:
  emissions        RCMIP3/<rname>.csv, contrib/<hsrc>_<psrc>/total/<psce>_WORLD.csv
                   (17 significant digits)
  concentrations   RCMIP3/<rname>_conc.csv
  land cover       contrib/LC_<lcov>_<psrc>/<region>/lc_<psce>.csv
  land-use CO2     contrib/<land>_<psrc>/<rgn>/CO2_EMISS_LUC_<psce>.csv
  observations     NOAA/<stem>.gdx.csv, AGAGE/<gas>.csv
  contrib_rfc      contrib/<hsrc>_<psrc>/<fol>/rfc_<X>/<stem>_rfc_<X>_all.csv
                   (one column per ensemble case)
Table files: read_csv_table / read_input; single series: read_input_series.

--scen=rcmip3: emissions from data/RCMIP3/<rname>.csv, time horizon per --rname,
no land-use CO2 override; with --load_conc the prescribed concentrations
(rcmip_conc) of data/RCMIP3/<rname>_conc.csv, optionally overwritten by AGAGE /
NOAA observations.

--fix_conc (contrib): prescribed CO2 concentration from
data/contrib/<hsrc>_<psrc>/conc/<psce>_WORLD.csv.

--scen=contrib_rfc: every forcing component read from
data/contrib/<hsrc>_<psrc>/<fol>/rfc_<X>/..._all.csv for one ensemble --case; no
emissions are needed.

Also: the marginal-analysis period tmargin, the natural forcing adjustments
(--scen=tcre/tcre2, --rdc_rfc, --zero_nature) and the land-cover overrides
(--rdc_lcc).

Not supported (raise or are ignored): psrc=INDC emissions, the ssp1..ssp5
land-cover files, and the contrib options --use_cm, --scmv2, --minus_ems,
--rdc_ems, --fraction and --fix_conc, none of which are set in the contrib
reference run.
"""

from dataclasses import dataclass, field

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import RunConfig, data_path, nat_grp_of, resolve
from ..core.constants import DT
from ..core.sets import GAS, GAS_NONF, HALO, HALO_RCMIP_AGAGE, LAND_TYPE, REGION_11

_STEP_SUFFIX = ("", ".167", ".333", ".5", ".667", ".833")


@dataclass(frozen=True)
class TimeAxis:
    """The time set t: integer step n = 0..N-1 <-> label "1850", "1850.167", ...

    Use step indices, never float equality on years. `tval` is the value of t,
    i.e. the label read as a number (1850.167, not 1850 + 1/6).
    """
    first_year: int
    last_year: int
    labels: tuple = field(init=False)
    tval: np.ndarray = field(init=False)

    def __post_init__(self):
        n = (self.last_year - self.first_year) * 6 + 1
        labels = tuple(f"{self.first_year + i // 6}{_STEP_SUFFIX[i % 6]}" for i in range(n))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "tval", np.array([float(s) for s in labels]))

    @property
    def n(self):
        return len(self.labels)

    def index(self, label):
        """Step index of a time label, e.g. index("2100") -> 1500."""
        return self.labels.index(str(label))


@dataclass
class ScenarioInputs:
    cfg: RunConfig
    t: TimeAxis
    eind: np.ndarray        # (n, gas)  dat_eind, per step
    elnd: np.ndarray        # (n, gas)  dat_elnd, per step
    ems: np.ndarray         # (n, gas)  dat_ems = eind + elnd
    ems_1850: np.ndarray    # (gas,)    dat_ems("1850", gas)
    rf_solar: np.ndarray    # (n,)      W/m2
    rf_volc: np.ndarray     # (n,)      W/m2
    rf_con: np.ndarray      # (n,)      W/m2, contrails
    rf_dust: np.ndarray     # (n,)      W/m2, rfc_DUST (always 0)
    tmargin: np.ndarray     # (n,) bool marginal-analysis period
    land_cover: np.ndarray  # (n, region_11, land_type)  Mha change
    conc: "PrescribedConc | None" = None   # --load_conc
    fix_co2: "np.ndarray | None" = None    # --fix_conc: prescribed D_CO2 (n,)
    rfc_data: "RfcData | None" = None      # --scen=contrib_rfc
    gas: tuple = GAS
    dt: float = DT

    def gi(self, gas):
        """Column index of a gas in eind/elnd/ems."""
        return self.gas.index(gas)

    @property
    def halo_idx(self):
        """Column indices of the 48 halogenated gases, in HALO order."""
        return np.array([self.gas.index(g) for g in HALO])


def to_dense(rec, axes):
    """Scatter long records (label columns + "value") onto a dense array.

    axes : one sequence of labels per label column, in column order.
    Labels missing from the records are left at 0, the value of parameter entries
    not given. Records whose labels are not in `axes` are dropped
    (e.g. years outside the model horizon).
    """
    out = np.zeros(tuple(len(a) for a in axes))
    if rec is None or len(rec) == 0:
        return out
    idx = []
    keep = np.ones(len(rec), dtype=bool)
    for col, labels in zip(rec.columns[:len(axes)], axes):
        pos = {lab: i for i, lab in enumerate(labels)}
        ii = rec[col].map(pos)
        keep &= ii.notna().to_numpy()
        idx.append(ii)
    idx = tuple(np.asarray(ii[keep], dtype=int) for ii in idx)
    out[idx] = rec["value"].to_numpy()[keep]
    return out


def _skip_comment(path):
    """1 if the first line of a CSV is a "*" comment ("*unit", "*symbol", ...), else 0."""
    with open(path) as f:
        return 1 if f.readline().startswith("*") else 0


def read_csv_table(path):
    """Records of an input table CSV, in long form (columns: t, label, value).

    Layout: an optional "*" comment row ("*unit,...", "*symbol,..."), a header row
    ("year" + column labels), then one row per time step with the time
    label ("1850", "1850.167", ...; quotes allowed). Values are parsed exactly
    (round_trip). Empty cells count as absent (0 after to_dense).
    """
    df = pd.read_csv(path, skiprows=_skip_comment(path), dtype={"year": str},
                     float_precision="round_trip")
    df["year"] = df["year"].str.strip().str.strip('"')
    long = df.melt(id_vars="year", var_name="label", value_name="value").dropna(subset=["value"])
    return long.rename(columns={"year": "t"}).reset_index(drop=True)


def read_input(path):
    """Records of a time x label input table CSV (read_csv_table); FileNotFoundError
    if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv_table(path)


def read_csv_column(path, column, t):
    """One column (e.g. an ensemble case) of a table CSV on t, 0 where absent; reads
    only that column, so wide tables (1000 cases) stay fast."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, skiprows=_skip_comment(path), usecols=["year", column],
                     dtype={"year": str, column: float}, float_precision="round_trip")
    s = df.set_index(df["year"].str.strip().str.strip('"'))[column]
    return s.reindex(list(t.labels)).fillna(0.0).to_numpy(dtype=float, copy=True)


def first_existing(*paths):
    """The first of the candidate paths that exists, else the first one (so that the
    error names it)."""
    return next((Path(p) for p in paths if Path(p).exists()), Path(paths[0]))


def read_csv_series(path):
    """A 1-D time series CSV as a Series indexed by the time label.

    Accepts the headerless two-column layout ('"1958.167",314.8'), with optional "*"
    comment lines, and the table layout of read_csv_table with a single value column.
    """
    with open(path) as f:
        first = next((ln for ln in f if not ln.startswith("*")), "")
    if first.split(",")[0].strip().strip('"').lower() == "year":
        df = read_csv_table(path)
        return df.set_index("t")["value"]
    df = pd.read_csv(path, header=None, names=["t", "value"], dtype={"t": str},
                     comment="*", float_precision="round_trip")
    df["t"] = df["t"].str.strip().str.strip('"')
    return df.set_index("t")["value"]


def read_input_series(path, t):
    """A 1-D time input CSV (NOAA / AGAGE observations, land-use CO2) on t, 0 where
    absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv_series(path).reindex(list(t.labels)).fillna(0.0).to_numpy(dtype=float, copy=True)


def _sce_column(gas, kind):
    """dat_sce_ems column feeding dat_eind (kind='i') or dat_elnd (kind='l')."""
    return f"{gas}{kind}" if gas in GAS_NONF else gas


def load_emissions(cfg, t):
    """dat_eind, dat_elnd, dat_ems and dat_ems_1850."""
    if cfg.psrc == "INDC":
        raise NotImplementedError("psrc=INDC uses a different LUC file layout")
    contrib = cfg.dat_dir / "contrib"

    if cfg.scen == "rcmip3":   # one file per RCMIP experiment
        path = cfg.dat_dir / "RCMIP3" / f"{cfg.rname}.csv"
    else:
        path = contrib / f"{cfg.hsrc}_{cfg.psrc}" / "total" / f"{cfg.psce}_WORLD.csv"
    rec = read_input(path)
    sce_cols = tuple(dict.fromkeys(rec[rec.columns[1]]))
    sce = pd.DataFrame(to_dense(rec, (t.labels, sce_cols)), columns=sce_cols)

    no_luc = ("rcmip3",)                # no land-use CO2 override
    if cfg.land and cfg.scen not in no_luc:  # co2l from the LUC file
        sce["co2l"] = read_input_series(
            contrib / f"{cfg.land}_{cfg.psrc}" / cfg.rgn / f"CO2_EMISS_LUC_{cfg.psce}.csv", t)

    n = t.n
    eind = np.zeros((n, len(GAS)))
    elnd = np.zeros((n, len(GAS)))
    for j, g in enumerate(GAS):
        # gases absent from the input table stay 0 (e.g. so2f2, c7f16, cfc13)
        col = _sce_column(g, "i")
        if col in sce:
            eind[:, j] = sce[col].to_numpy()
        if g in GAS_NONF and f"{g}l" in sce:  # dat_elnd(t,halo) = 0
            elnd[:, j] = sce[f"{g}l"].to_numpy()
    ems = eind + elnd
    ems_1850 = ems[t.index("1850")].copy()
    return eind, elnd, ems, ems_1850


def _read_csv_series(path, t):
    """A delimited parameter file ("*year,value" + quoted labels) on t, 0-filled."""
    df = pd.read_csv(path, comment="*", header=None, names=["t", "value"], dtype={"t": str})
    s = df.set_index("t")["value"]
    return s.reindex(list(t.labels), fill_value=0.0).to_numpy(dtype=float, copy=True)


def tmargin_mask(cfg, t):
    """tmargin(t), the marginal-analysis period."""
    y = t.tval
    if cfg.tproj:                       # 2023-2100
        return (y >= 2023) & (y <= 2100)
    if cfg.thist:                       # 1850-2022
        return (y >= 1850) & (y <= 2022)
    if cfg.tyear is not None:
        return np.floor(y) == cfg.tyear
    if cfg.byear is not None:
        return y < cfg.byear + 1
    if cfg.dyear is not None:
        return (y >= cfg.dyear + 1) & (y < cfg.dyear + 11)
    return y <= 2100                    # 1850-2100


def load_natural_forcing(cfg, t, tmargin):
    """rfc_solar, rfc_volc, rfc_con and rfc_DUST, with the natural forcing
    adjustments."""
    grp = nat_grp_of(cfg)
    tag = "" if grp == "default" else f"_rcmip3_{grp}"
    solar = _read_csv_series(data_path(cfg.dat_dir, "nat", f"Solar_foricng{tag}_bimon.csv"), t)
    volc = _read_csv_series(data_path(cfg.dat_dir, "nat", f"Volcanic_foricng{tag}_bimon.csv"), t)
    con = _read_csv_series(data_path(cfg.dat_dir, "AR6", "AR6_foricng_contrails.csv"), t)
    dust = np.zeros(t.n)
    if grp == "zero":
        con[:] = 0.0
    if cfg.scen in ("tcre", "tcre2"):
        volc[:] = 0.0
        solar[:] = 0.0
    if cfg.rdc_rfc == "rfc_volc":
        volc[tmargin] = 0.0
    elif cfg.rdc_rfc == "rfc_solar":
        solar[tmargin] = 0.0
    elif cfg.rdc_rfc == "rfc_DUST":
        dust[tmargin] = 0.0
    if cfg.zero_nature:
        volc[:] = 0.0
        solar[:] = 0.0
        dust[:] = 0.0
    return solar, volc, con, dust


def load_land_cover(cfg, t, tmargin):
    """land_cover_WORLD(t, region_11, land_type) in Mha."""
    if cfg.scen in ("ssp1", "ssp2", "ssp3", "ssp4", "ssp5"):
        raise NotImplementedError("--scen=ssp1..5 land cover is not supported")
    # INDC runs take the IAMC SSP1_26 land cover
    src, sce = ("IAMC", "SSP1_26") if cfg.psrc == "INDC" else (cfg.psrc, cfg.psce)
    lc = np.zeros((t.n, len(REGION_11), len(LAND_TYPE)))
    for r, reg in enumerate(REGION_11):
        path = cfg.dat_dir / "contrib" / f"LC_{cfg.lcov}_{src}" / reg / f"lc_{sce}.csv"
        lc[:, r, :] = to_dense(read_input(path), (t.labels, LAND_TYPE))
    # hold the 2100 land cover beyond 2100
    if cfg.last_year > 2100:
        i2100 = t.index("2100")
        lc[i2100 + 1:] = lc[i2100]
    i1850 = t.index("1850")
    if cfg.rdc_lcc == "WORLD":
        lc[tmargin] = lc[i1850]
    elif cfg.rdc_lcc is not None:
        r = REGION_11.index(cfg.rdc_lcc)
        lc[tmargin, r] = lc[i1850, r]
    return lc


@dataclass
class PrescribedConc:
    """Prescribed concentration anomalies (rcmip_conc), on t."""
    D_CO2: np.ndarray        # (n,) ppm, used with --load_conc=all
    D_CH4: np.ndarray        # (n,) ppb
    D_N2O: np.ndarray        # (n,) ppb
    D_HALO: np.ndarray       # (n, 48) ppt, chm_HALO_conc (0 for the non-AGAGE gases)
    halo_mask: np.ndarray    # (48,) True for halo_rcmip_agage (prescribed); others emission-driven


def load_conc(cfg, t):
    """rcmip_conc and chm_HALO_conc, with the AGAGE / NOAA overrides."""
    if cfg.load_conc not in ("all", "nco2"):
        raise ValueError(f"--load_conc={cfg.load_conc!r}: use all or nco2")
    from ..physics.chemistry import load_chem_params
    from ..scenarios.rcmip3_data import AGAGE_GASES, NOAA_SERIES
    rec = read_input(cfg.dat_dir / "RCMIP3" / f"{cfg.rname}_conc.csv")
    rec[rec.columns[1]] = rec[rec.columns[1]].str.lower()   # labels are case-insensitive
    types = ("chm_d_co2", "chm_d_ch4", "chm_d_n2o") + tuple(f"chm_d_{g}" for g in HALO)
    c = dict(zip(types, to_dense(rec, (t.labels, types)).T))
    pc = load_chem_params(cfg)
    if cfg.load_AGAGE:
        for g, minus_h0 in AGAGE_GASES.items():
            obs = read_input_series(cfg.dat_dir / "AGAGE" / f"{g}.csv", t)
            val = obs - pc.halo_0[HALO.index(g)] if minus_h0 else obs
            c[f"chm_d_{g}"] = np.where(obs != 0, val, c[f"chm_d_{g}"])
    if cfg.load_NOAA:                   # applied after AGAGE
        for stem, sym, ctype, base in NOAA_SERIES:
            obs = read_input_series(first_existing(cfg.dat_dir / "NOAA" / f"{stem}.gdx.csv",
                                                   cfg.dat_dir / "NOAA" / f"{stem}.csv"), t)
            val = obs - getattr(pc, base) if base else obs
            c[ctype.lower()] = np.where(obs != 0, val, c[ctype.lower()])
    mask = np.array([g in HALO_RCMIP_AGAGE for g in HALO])
    halo = np.stack([c[f"chm_d_{g}"] if m else np.zeros(t.n) for g, m in zip(HALO, mask)], axis=1)
    return PrescribedConc(D_CO2=c["chm_d_co2"], D_CH4=c["chm_d_ch4"], D_N2O=c["chm_d_n2o"],
                          D_HALO=halo, halo_mask=mask)


def load_fixconc(cfg, t):
    """sce_conc(t, "chm_D_CO2") (delimited table, conc_type columns)."""
    path = cfg.dat_dir / "contrib" / f"{cfg.hsrc}_{cfg.psrc}" / "conc" / f"{cfg.psce}_WORLD.csv"
    df = pd.read_csv(path, dtype=str)       # keep labels as text ("1851", not 1851.0)
    df.columns = [str(c).strip().strip('"').lower() for c in df.columns]
    df = df.set_index(df.columns[0])
    df.index = df.index.str.strip().str.strip('"')
    return df["chm_d_co2"].astype(float).reindex(list(t.labels), fill_value=0.0).to_numpy(dtype=float, copy=True)


# contrib_rfc components, in load order
RFC_COMPONENTS = ("CO2", "CH4", "H2Os", "N2O", "O3t", "O3s", "SO4", "POA", "BC", "BCsnow",
                  "NO3", "SOA", "DUST", "volc", "solar", "cloud", "LCC")


@dataclass
class RfcData:
    """Forcing components of one contrib_rfc ensemble case [W/m2] (contrib_rfc_*)."""
    comp: dict          # name (RFC_COMPONENTS) -> (n,)
    halo: np.ndarray    # (n, 48) contrib_HALO
    con: np.ndarray     # (n,) contrib_rfc_con = 0


def load_rfc_data(cfg, t, tmargin):
    """Forcing components for --scen=contrib_rfc."""
    if not cfg.fol or not cfg.case:
        raise ValueError("--scen=contrib_rfc needs --fol and --case")
    base = cfg.dat_dir / "contrib" / f"{cfg.hsrc}_{cfg.psrc}" / cfg.fol
    stem = f"{cfg.rgn}_{cfg.csec}_{cfg.hsrc}_{cfg.psrc}_{cfg.psce}"

    def read(name):
        path = base / f"rfc_{name}" / f"{stem}_rfc_{name}_all.csv"
        if name == "DUST" and not path.exists():          # optional, else 0
            return np.zeros(t.n)
        return read_csv_column(path, cfg.case, t)          # contrib_all_rfc_<name>(t, case)

    comp = {name: read(name) for name in RFC_COMPONENTS}
    halo = np.stack([read(f"HALO_{g}") for g in HALO], axis=1)
    if cfg.minus_rfc:                   # scale one component over tmargin
        f = 1 - (cfg.fraction if cfg.fraction is not None else 0.001)
        key = cfg.minus_rfc.removeprefix("rfc_")
        if key.startswith("HALO_"):
            halo[tmargin, HALO.index(key[5:])] *= f
        else:
            comp[key][tmargin] *= f
    return RfcData(comp=comp, halo=halo, con=np.zeros(t.n))


def load_inputs(cfg=None):
    """Load every exogenous input of the forward model for one run configuration."""
    cfg = resolve(cfg or RunConfig())
    if cfg.scen == "rcmip3" and (cfg.rdc_rfc or cfg.rdc_lcc):
        raise ValueError("--rdc_rfc / --rdc_lcc need tmargin, which --scen=rcmip3 does not define")
    t = TimeAxis(cfg.first_year, cfg.last_year)
    tmargin = tmargin_mask(cfg, t)
    if cfg.scen == "contrib_rfc":        # no emissions are read in this mode
        eind = elnd = ems = np.zeros((t.n, len(GAS)))
        ems_1850 = np.zeros(len(GAS))
    else:
        eind, elnd, ems, ems_1850 = load_emissions(cfg, t)
    solar, volc, con, dust = load_natural_forcing(cfg, t, tmargin)
    return ScenarioInputs(cfg=cfg, t=t, eind=eind, elnd=elnd, ems=ems, ems_1850=ems_1850,
                          rf_solar=solar, rf_volc=volc, rf_con=con, rf_dust=dust,
                          tmargin=tmargin, land_cover=load_land_cover(cfg, t, tmargin),
                          conc=load_conc(cfg, t) if cfg.load_conc else None,
                          fix_co2=(load_fixconc(cfg, t)
                                   if cfg.fix_conc and cfg.scen == "contrib" else None),
                          rfc_data=(load_rfc_data(cfg, t, tmargin)
                                    if cfg.scen == "contrib_rfc" else None))
