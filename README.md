# SCM4OPT_PY

The Python implementation of SCM4OPT (RCMIP3 model name `SCM4OPT(PY)v4.0`).
SCM4OPT — **S**imple **C**limate **M**odel for **OPT**imization — is a reduced-complexity
coupled climate model: carbon cycle (land, ocean, atmosphere), atmospheric chemistry,
radiative forcing and the DOECLIM energy-balance model, at a 1/6-year time step.
It runs scenario experiments (contribution analysis, RCMIP3) and parameter ensembles.

Version 4.1.0.dev0.

## Documentation

`docs/`: the [user guide](docs/user_guide.md), the [developer guide](docs/developer_guide.md),
and references generated from the code: [options](docs/options.md),
[variables](docs/variables.md) and the [API](docs/api/README.md).

## Install

```bash
pip install -e .            # numpy, pandas, scipy
pip install -e ".[test]"    # pytest
```

Python >= 3.10. Ensembles also need GNU parallel. All inputs and outputs are CSV.

## Data

The parameter tables (CMIP6 calibrations, natural forcing, RCMIP3 output mapping,
ensemble members, ...) are in `scm4opt/data/`, tracked in git and installed with the
package. The scenario inputs (`RCMIP3/`, `contrib/`, `AGAGE/`, `NOAA/`: emissions,
concentrations, land cover, observations, forcing data; 7933 files) are distributed
separately as SCM4OPT_PY_DAT, published on Zenodo:
[doi:10.5281/zenodo.23718272](https://doi.org/10.5281/zenodo.23718272). Download it into
a folder and point the model to that folder:

```bash
export SCM4OPT_DATA=/path/to/SCM4OPT_PY_DAT      # or --dat_dir=... per run
scm4opt data verify                              # files against inputs_manifest.csv
```

Parameter tables missing in that folder are taken from `scm4opt/data/`. Copying the input
folders into `scm4opt/data/` works as well (they are git-ignored there). `scm4opt data pack
--out DIR` writes one archive per set (`scm4opt-inputs-<set>.tar.gz`) with
`archives_manifest.csv`; `scm4opt data fetch --source URL_OR_DIR [--data DIR]` downloads
them, checks each archive's checksum, unpacks it and verifies the files (all sets by
default; RCMIP3 runs also need the land cover in `contrib/`).

## Run

```bash
scm4opt --scen=contrib --psrc=IAMC --psce=SSP1_19
scm4opt --scen=rcmip3 --rname=ssp585 --load_conc=all --rout3=out/
scm4opt --scen=rcmip3 --rname=ssp126 --ocean_model=hector --cm6mdl=MIROC6
scm4opt ensemble --rname=ssp585 -j 8 --cases=0-49
scm4opt export --sce=ssp585 --out_dir=ens/      # members -> RCMIP3 ensemble file
bash scm4opt/run_rcmip3_all_py.sh -j 8          # every RCMIP3 experiment (--help)
python3 -m scm4opt --help
```

Options are given as `--name=value` flags; see `scm4opt/scenarios/forward.py` and
`scm4opt/core/config.py` for all of them.

## Names

* Output variables: `<prefix>_<name>`, with the prefix of the producing module:
  `scm_` (coupling between modules), `chm_` (chemistry), `rfc_` / `erf_` (radiative
  forcing / effective radiative forcing), `doe_` (DOECLIM), `lnd_` (land), `ocn_`
  (OSCAR ocean), `hct_` (Hector ocean), `dat_` (inputs).
  `D_` marks an anomaly against the pre-industrial state (e.g. `chm_D_CH4`).
* Parameters: snake_case, `<quantity>_<species/box>_<qualifier>` (e.g. `tau_ch4_oh`,
  `alpha_n2o`, `d_ch4_1850` for an anomaly); values per model step end in `_per_step`.
* `scm4opt/data/registry/registry.csv` lists every output and parameter with its
  module, dimensions, unit, time basis (state / per_step / per_year / none),
  description and position in the model report. The report symbol list and the
  RCMIP3 export's time aggregation are generated from it; `python3 -m
  scm4opt.core.registry` checks the units (grammar in `scm4opt/core/units.py`) and
  every RCMIP3 export conversion against them.

## Package layout

| Subpackage | Contents |
|---|---|
| `core` | run configuration (`config`), constants, index sets |
| `physics` | atmosphere, chemistry, forcing, DOECLIM, land, ocean (OSCAR, Hector) |
| `scenarios` | coupled forward run (`forward`), RCMIP3 experiment tables (`rcmip3_data`) |
| `io` | inputs (CSV), report, RCMIP3 rows and ensemble export (`rcmip_export`) |
| `ensemble`, `cli` | ensemble driver, command line |

## Reference compatibility (`--ref_compat`)

By default (`--ref_compat=1`) the model reproduces the SCM4OPT v4.1 reference results
bit for bit, including a few known quirks. `--ref_compat=0` uses the corrected
behaviour instead:

* ocean carbon (OSCAR): the air-sea flux is computed from the outflows that drive the
  box integration, so ocean and atmosphere carbon are mass-consistent (the reference
  form loses about 2 GtC by 2100);
* Hector ocean with `--scen=tcre/tcre2`: the transport rates are recomputed;
* ocean heat diagnostics: the mixed-layer heat uptake is a flux in W/m2 and the heat
  content is the sum of the mixed-layer and interior heat (temperatures are unchanged);
* RCMIP3 export: ocean heat content and uptake converted with the factors implied by
  their units (1e22 J = 10 ZJ; the uptake as an annual mean); yearly means over all
  steps and every row kept;
* the temperature derivatives of the OH and photolysis attribution helpers.

Submitted RCMIP3 results use the default. `tests/test_compat.py` checks the corrected
branches.

## Tests

```bash
pytest tests/
```

The regression tests compare against reference outputs written by the export; the
registry tests check units and the RCMIP3 export conversions. GitHub Actions runs
them on Python 3.10 and 3.13 (`.github/workflows/tests.yml`).

## License and citation

The code is licensed under the GNU General Public License, version 3 or later
(`LICENSE`). It contains parts derived from Hector (GPL-3.0), OSCAR (CeCILL 2.1),
DOECLIM and MAGICC6; see `NOTICE` for the credits and references. To cite SCM4OPT, see
`CITATION.cff`.

`scm4opt/data/MAGICC6/land_c4mip_bern.csv` (the land parameters of the MAGICC6-defaults
mode, used when no `--cm6mdl` calibration is selected) comes from the configuration
files distributed with the MAGICC6 executable and keeps their license, Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 (see `scm4opt/data/MAGICC6/NOTICE.txt`; cite
Meinshausen et al., 2011, doi:10.5194/acp-11-1417-2011). It is not covered by the code
license.
