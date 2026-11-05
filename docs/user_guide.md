# SCM4OPT_PY user guide

SCM4OPT (Simple Climate Model for OPTimization) is a reduced-complexity climate model:
atmospheric chemistry of the greenhouse gases, ozone and aerosols; radiative and effective
radiative forcing; the DOECLIM energy-balance model; and a carbon cycle with land, ocean
(OSCAR-type mixed layer or the Hector four-box ocean) and atmosphere. It runs from 1850 at a
time step of 1/6 year. This guide covers installation, the data, single runs, ensembles,
the outputs and the Python interface.

Reference material, generated from the code:

* [options.md](options.md): every `--flag` with its default
* [variables.md](variables.md): every output and parameter, with unit and time basis
* [api/](api/README.md): modules, classes and functions

---

## 1. Installation

Python 3.10 or newer with numpy, pandas and scipy. Ensembles also need GNU parallel.

```bash
pip install scm4opt         # from PyPI
scm4opt --version
```

or, to work with the code and the tests, from a clone of the repository:

```bash
git clone https://github.com/SooClimate/SCM4OPT_PY SCM4OPT_PY
cd SCM4OPT_PY
pip install -e .            # or: pip install -e ".[test]" to run the tests
scm4opt --version
```

The parameter tables come with the package, and so do the scenario inputs of the RCMIP3
experiments `historical`, `esm-hist`, `esm-allGHG-hist`; the other scenario inputs are a separate download, found
through `$SCM4OPT_DATA` or in `scm4opt/data/` of an editable install (section 2). Without installing, run from the
repository folder as `python3 -m scm4opt ...`.

GNU parallel: `brew install parallel` (macOS), `apt install parallel` / `dnf install
parallel` (Linux). On Windows use WSL.

---

## 2. Data

The model reads two kinds of data:

| folder | contents | where |
|---|---|---|
| `CMIP6/`, `RCMIP/`, `AR6/`, `hector/`, `nat/`, `tables/`, `MAGICC6/`, `RCMIP3_src/`, `ensemble/`, `registry/` | parameter tables and calibrations (CMIP6 model fits, halogen tables, natural forcing, RCMIP3 output mapping, the ensemble member list, the variable registry) | `scm4opt/data/`, installed with the package |
| `RCMIP3/` | RCMIP phase 3 emissions and concentrations, one file per experiment | scenario inputs |
| `contrib/` | emission totals, land cover, land-use CO2 and the contribution-run forcing data | scenario inputs |
| `AGAGE/`, `NOAA/` | observed concentrations | scenario inputs |

The scenario inputs of the RCMIP3 experiments `historical`, `esm-hist`, `esm-allGHG-hist` (from 1850 and from 1750) are
included in `scm4opt/data/` of this repository. The others are distributed separately
as SCM4OPT_PY_DAT, published on Zenodo (from 1850 only so far):
[doi:10.5281/zenodo.23718272](https://doi.org/10.5281/zenodo.23718272). Use them from their own folder with `export SCM4OPT_DATA=/path/to/SCM4OPT_PY_DAT`,
or copy the four folders into `scm4opt/data/`. They are listed with their size and SHA-256
checksum in `inputs_manifest.csv` of the data folder:

```bash
scm4opt data verify                       # check the data folder against the manifest
scm4opt data fetch --source <URL or folder> --data <folder>   # download and check archives
```

Another data folder is used with `export SCM4OPT_DATA=/path/to/data` or `--dat_dir=...`.
Parameter tables that are missing there are still read from `scm4opt/data/`. The input sets
depend on each other: RCMIP3 runs also read the land cover in `contrib/`, and
`--load_AGAGE` / `--load_NOAA` the observation sets, so fetch all sets.

The scenario inputs are derived from external datasets (RCMIP, CEDS, LUH2, AGAGE, NOAA and
others) and remain subject to their terms and citation requirements.

---

## 3. A single run

```bash
scm4opt --scen=rcmip3 --rname=ssp585 --load_conc=all --rout3=out/
```

Options are `--name=value` flags; booleans take `=1` or `=0`. The run prints a summary
(CO2, temperature and forcing at a few years). With `--rout3=<folder>` it writes two files
there:

* `<rname>_<case>_<ctag>.csv`: the model report, one row per time step and one column per
  report variable (`--report_csv=0` skips it; about 27 MB for 1850-2500);
* `SCM4OPTv40_rcmip_phase3_<scenario>_<case>.csv`: the RCMIP3 submission rows of the run
  (yearly values, RCMIP variable names and units; `--rcmip_csv=0` skips it).

### Scenario drivers (`--scen`)

| `--scen` | what it runs | main options |
|---|---|---|
| `contrib` (default) | emission-driven run 1850-2100 from `contrib/<hsrc>_<psrc>/total/<psce>_WORLD.csv` | `--psce`, `--psrc`, `--hsrc`, `--land`, `--lcov`; marginal analysis `--tyear`, `--byear`, `--dyear`, `--tproj`, `--thist` with `--minus_ems=1 --rdc_ems=<column> [--rdc_mix=1]`, `--rdc_lcc`, `--rdc_rfc`; `--use_cm=1` (Carbon Monitor CO2 2019-2025); `--fix_conc=1`; `--cout5=<dir>` writes the contribution report |
| `rcmip3` | an RCMIP phase 3 experiment; `--rname` selects the experiment and its time range | `--rname`, `--load_conc=all` (CO2, CH4, N2O and halogens prescribed) or `--load_conc=nco2` (as all, but CO2 emission-driven), `--load_AGAGE=1`, `--load_NOAA=1`, `--first_year=1750` (see below) |
| `contrib_rfc` | forcing components read from files, then effective forcing and DOECLIM only (no carbon cycle) | `--fol`, `--case`, `--minus_rfc`, `--fraction` |
| `tcre`, `tcre2` | a `contrib` run with the settings of the TCRE analyses (transient climate response to cumulative emissions): natural forcing off, a fixed DOECLIM parameter set, the alternative land and Hector initialisation | as `contrib` |

`scm4opt ensemble --list-rcmip3` prints the 135 RCMIP3 experiments.

### Frequently used options

| option | meaning |
|---|---|
| `--cm6mdl=<ESM>` | CMIP6 model whose calibration is used for DOECLIM, land and ocean (default `NorESM2-MM`); `--cm6mdl=None` uses the MAGICC6 land defaults (`data/MAGICC6/`, CC BY-NC-SA 3.0) |
| `--ch4rfc=etminan\|myhre` | CO2/CH4/N2O forcing formulas (Etminan et al. 2016 or Myhre et al. 1998) |
| `--ocean_model=osc\|hector` | ocean carbon model: OSCAR-type mixed layer (default) or Hector v1.0 four-box ocean (`--same_alk=1`: one alkalinity) |
| `--clim_sensi=<K>` | equilibrium climate sensitivity overriding the calibration |
| `--co2_conc_initial=<ppm>` | CO2 of the atmospheric carbon pool at the first step (default 284, or 278.00745 with `--first_year=1750`; former name `--co2_conc_1850`, still accepted). When CO2 is prescribed (`--load_conc=all`, `--fix_conc=1`), the run outputs the compatible CO2 emissions `scm_eind_co2_compat` (fossil and industry) and `scm_ems_co2_compat` (plus land use). Running emission-driven with these emissions and `--co2_conc_initial` set to the prescribed first-year value reproduces the prescribed concentrations |
| `--rcmip_compat_ems=1` | with `--rout3`: where CO2 is prescribed, the RCMIP3 rows `Emissions\|CO2` and `Emissions\|CO2\|Fossil and Industry` hold the compatible emissions instead of the input emissions (the other rows are unchanged; emission-driven runs are not affected). The 1850 value sums five of the six steps, because the first step has no compatible emission |
| `--<process>_mdl=<choice>` | the chemistry, forcing and ocean process choices (e.g. `--OHSNK_mdl=CMAM`); see options.md |
| `--ref_compat=0` | corrected behaviour instead of the reference quirks (section 6) |
| `--climate_model=<name>` | the `climate_model` written in the RCMIP3 rows (default `SCM4OPT(PY)v4.0`) |
| `--dat_dir=<folder>` | data folder |

The complete list with defaults is [options.md](options.md).

### Runs from 1750 (`--first_year=1750`)

By default an RCMIP3 run starts in 1850 from an equilibrium state (the RCMIP3 inputs). With
`--scen=rcmip3 --first_year=1750` it starts in 1750 instead, for example for assessments that
use 1750 as the pre-industrial reference:

- inputs: the `_1750` files, `RCMIP3/<rname>_1750.csv` and `RCMIP3/<rname>_conc_1750.csv`
  (concentrations as anomalies from 1750) and `nat/*_rcmip3_<group>_bimon_1750.csv`;
- reference concentrations: the 1750 values of the RCMIP3 inputs (CO2 278.00745 ppm, CH4
  720.8548 ppb, N2O 266.57 ppb), which are also the initial concentrations of the Etminan et
  al. (2016) forcing, so the CO2, CH4 and N2O forcing is zero in 1750;
- land: the land carbon cycle is calibrated for an equilibrium start in 1850, so by default
  `--lnd_fb_1850=fert` starts the CO2 fertilisation in 1850 (relative to the simulated 1850
  CO2) and `--norgrwth_pre1850=1` keeps the 1750-1849 land-use emissions from regrowing.
  `--lnd_fb_1850=both` also starts the land temperature feedbacks in 1850;
  `--lnd_fb_1850=none --norgrwth_pre1850=none` applies the 1850 calibration from 1750 unchanged;
- outputs: the RCMIP3 rows add the 1750 reference concentrations (`RCMIP3_src/conc_mapping_1750.csv`).
  `scm4opt ensemble ... -- --first_year=1750` writes to `rcmip_<rname>_1750/`, and
  `scm4opt rcmip_export --run_suffix=_1750` combines those members.

---

## 4. Ensembles

The ensemble has 500 members (`scm4opt/data/ensemble/rcmip_unc.txt`): each line is a full
set of flags (`--case`, `--cm6mdl`, `--ch4rfc`, process choices). Members run in parallel
with GNU parallel:

```bash
scm4opt ensemble --rname=ssp585 -j 8                 # all 500 members, 8 at a time
scm4opt ensemble --rname=ssp585 --cases=0-49         # a subset (0-based)
scm4opt ensemble --rname=esm-ssp585 -- --cm6mdl=MIROC6   # extra flags for every member
scm4opt ensemble --rname=ssp585 --dry-run            # print the command only
```

For RCMIP3 experiments the experiment's standard options are added (`--load_conc=all` for
concentration-driven, `--load_conc=nco2` for `esm-*` experiments). Results go to
`<base>/rcmip_<rname>/` and member logs to `<base>/exe/<rname>/`, where `<base>` is `$mdl_dir`
or the repository folder (`--out` sets the results folder). Exit codes: 3 for missing
inputs, 4 if the Python cannot import the model's dependencies.

All 135 experiments, one after the other:

```bash
bash scm4opt/run_rcmip3_all_py.sh -j 8 --out-dir /path/to/output/
bash scm4opt/run_rcmip3_all_py.sh --only "ssp126 ssp585" --skip-done
bash scm4opt/run_rcmip3_all_py.sh --help
```

By default the runner writes only the RCMIP3 CSV of each member (about 2 MB);
`--full-report` adds the per-step report.

### Contribution runs

`scm4opt/run_period_py.sh` runs the contribution analysis of SSP1_19 (with Carbon Monitor
CO2) for the contribution ensemble (`scm4opt/data/ensemble/contrib_unc_h.txt`, first 1000
members): a reference run and 64 runs that each, in the chosen year, reduce one emission
species by 0.1 %, hold the 1850 land cover, or remove a natural forcing. The script sets
`--reduce_after_cm=1`: the reduction is applied after the Carbon Monitor step, so CO2
reductions in 2019-2025 count and a reduction in 2018-2022 does not leak into 2023-2025
(`--ref_compat=1` alone applies it before, as the reference results do).

```bash
bash scm4opt/run_period_py.sh              # 2020
bash scm4opt/run_period_py.sh 2010         # another year
bash scm4opt/run_period_py.sh tAll         # the whole period 1850-2100
bash scm4opt/run_period_py.sh 2010 periodb # all years up to 2010 (1850-2010)
nmem=50 njobs=8 bash scm4opt/run_period_py.sh   # 50 members, 8 at a time
```

With `periodb` (or `ptype=periodb`) the reduction covers every step up to the end of the
year (`--byear`) instead of that year alone (`--tyear`).

Each run writes the contribution report
`csv_<ptype>_<year>_py/WORLD/COMB_IAMC_SSP1_19_<ems>_EMISS_IND_<case>_<ctag>.csv`
(about 6.4 MB; 65 runs x 1000 members are about 415 GB). A contribution is the difference
between the reference run (`co2i_..._all`) and a reduced run. A rerun continues where an
interrupted one stopped.

Results go under `$mdl_dir` (default: the folder the script is started from);
`qsub -v mdl_dir=...` or `mdl_dir=... bash ...` sets it.

### Combining members into submission files

```bash
scm4opt export --csv_dir=/path/to/output --out_dir=/path/to/ens        # all experiments found
scm4opt export --csv_dir=/path/to/output --out_dir=/path/to/ens --sce=ssp585 -j 8
```

This writes `SCM4OPTv40_rcmip_phase3_<scenario>_ens.csv` per experiment, one row per
variable and member, with the columns of the RCMIP3 submission template. A 500-member
experiment takes about a minute and a few hundred MB of memory. `--run_suffix=_hector`
reads Hector-ocean ensembles; `--climate_model` sets the model name written in the rows.

---

## 5. Outputs

Output names follow `<prefix>_<name>`, with the prefix of the producing module:

| prefix | module |
|---|---|
| `scm_` | variables passed between modules (atmospheric CO2, temperature, fluxes) |
| `chm_` | atmospheric chemistry (concentrations, sinks, ozone, aerosols) |
| `rfc_` / `erf_` | radiative forcing / effective radiative forcing |
| `doe_` | DOECLIM (temperatures, ocean heat) |
| `lnd_` | land carbon cycle |
| `ocn_` / `hct_` | OSCAR ocean / Hector ocean |
| `dat_` | inputs as used by the model |

`D_` marks an anomaly from the pre-industrial state (e.g. `chm_D_CH4`). Every output is in
[variables.md](variables.md) with its dimensions, unit and time basis: `state` (a level at a
time point), `per_step` (an amount per 1/6-year step, e.g. emissions and fluxes in
GtC/step) or `per_year`. Per-step amounts are summed into yearly values by the RCMIP3
export.

---

## 6. Reference compatibility (`--ref_compat`)

By default the model reproduces the SCM4OPT v4.1 reference results bit for bit, including a
few known quirks. `--ref_compat=0` uses the corrected behaviour:

* OSCAR ocean: the air-sea flux is mass-consistent with the box integration;
* Hector ocean with `--scen=tcre/tcre2`: transport rates recomputed;
* ocean heat diagnostics: mixed-layer heat uptake in W/m2, heat content = mixed + interior;
* RCMIP3 export: ocean heat content and uptake converted with the factors implied by their
  units; yearly means over all steps; all rows kept;
* the temperature derivatives of the OH and photolysis attribution helpers.

Temperatures, concentrations and forcing differ only through the ocean carbon flux (about
0.6 ppm CO2 by 2100 in ssp245). Published RCMIP3 results use the default.

---

## 7. Using the model from Python

```python
from scm4opt.core.config import config_from_flags
from scm4opt.io.inputs import load_inputs
from scm4opt.scenarios.forward import run, write_report

cfg = config_from_flags(["--scen=rcmip3", "--rname=ssp245", "--load_conc=nco2"])
inp = load_inputs(cfg)            # time axis, emissions, forcing data, resolved options
out = run(inp.cfg, inp)           # {name: numpy array over the time steps}

t = inp.t                          # 1/6-year steps; t.labels ("1850", "1850.167", ...), t.tval
i = t.index("2100")
print(out["scm_atmoCO2"][i], out["scm_tatm"][i])   # ppm, K

# report and RCMIP3 rows as files (same as --rout3)
from dataclasses import replace
write_report(out, replace(inp, cfg=replace(inp.cfg, rout3="out")))
```

Arrays of gases, boxes or regions have extra trailing dimensions (e.g. `chm_D_HALO` is
(steps, 48)). `RunConfig` is a frozen dataclass: change options with
`dataclasses.replace(cfg, ...)` or build it from flags. The physics modules can also be run
step by step (`load_<module>_params(cfg)`, `<module>_step(params, prev, drivers)`); see the
developer guide and the API docs.

---

## 8. Problems

| message | cause |
|---|---|
| `FileNotFoundError: .../RCMIP3/<rname>.csv` (or contrib files) | the scenario inputs are missing: `scm4opt data verify`, set `$SCM4OPT_DATA`, or fetch the data |
| `ModelBoundsError: <variable> ...` | a variable left its physical bounds (e.g. a negative pool or an unphysical temperature); the run stops with the variable, its value and the bound |
| `unknown flag --<name>` | not an option; see options.md |
| `--scen=<x> is not supported` | the scenario driver is not part of this package |
| ensemble exit code 4 | the Python used by GNU parallel cannot import numpy/pandas/scipy (`--python` in the runner) |

---

## 9. Citation and license

Cite SCM4OPT as given in `CITATION.cff`, and the component models in `NOTICE`. The code is
licensed under the GNU GPL v3 or later (`LICENSE`); `scm4opt/data/MAGICC6/` is under CC
BY-NC-SA 3.0.
