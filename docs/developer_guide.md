# SCM4OPT_PY developer guide

How the model is organised, the conventions every module follows, and how to add an option,
a variable, a parameter table or an ocean model without breaking the reference results.
The user guide ([user_guide.md](user_guide.md)) covers running the model; the API docs
([api/](api/README.md)) list every module, class and function.

---

## 1. Package layout

```
scm4opt/
  cli.py, __main__.py       command line: scm4opt [run] | ensemble | export | data | --version
  ensemble.py               ensemble driver (GNU parallel); run_rcmip3_all_py.sh runs all experiments
  core/
    config.py               RunConfig and the option classes; --flag parsing; data folder lookup
    constants.py, sets.py   physical constants, time step, index sets (gases, regions, land types)
    tables.py               lookup helpers for option tables (pick, halo_mask)
    errors.py               ModelBoundsError
    units.py                unit grammar and conversion factors of the registry
    registry.py             the variable registry: report lists, per-step variables, checks
    version.py              __version__ and the RCMIP3 model name
  physics/
    atmosphere.py           atmospheric carbon pool
    chemistry.py            CH4/N2O/halogen stocks and sinks, ozone, EESC, aerosols
    forcing.py              radiative forcing and effective radiative forcing
    halo_data.py, forcing_data.py   loaders of the parameter tables in data/tables/
    doeclim.py              DOECLIM energy balance (land/ocean temperatures, ocean heat)
    land.py                 land carbon cycle (plant, detritus, soil)
    ocean_oscar.py          OSCAR-type mixed-layer ocean carbon
    ocean_hector.py         Hector v1.0 four-box ocean carbon
    ocean_models.py         the ocean plug-in interface
  scenarios/
    forward.py              the coupled time loop (run) and the forcing-only run (run_rfc)
    rcmip3_data.py          per-experiment tables, maintained by hand
  io/
    inputs.py               scenario inputs (CSV) on the model time axis
    report.py               model report and RCMIP3 rows of one run
    rcmip_export.py         ensemble members -> RCMIP3 ensemble files
    datasets.py             scenario input manifest, verification, archives, download
  data/                     parameter tables (packaged) and scenario inputs (see the user guide)
tests/                      regression, registry, compatibility, data and export tests
docs/                       this guide, the user guide, generated references
```

---

## 2. Time stepping

The model runs at a fixed step of 1/6 year (`core.constants.DT`) from 1850. Every equation of
step n depends only on earlier equations of the same step or on earlier steps, so the model
is a sequential time loop (`scenarios.forward.run`). Within step n:

1. atmosphere: pool A(n) from the fluxes of n-1 -> CO2(n), D_CO2(n)
2. chemistry, part 1: stocks (sinks of n-1), ozone, EESC, aerosols (temperature of n-1)
3. forcing: RF and ERF of step n -> `scm_ftot(n)`
4. DOECLIM: land and ocean temperatures, `tas(n)`
5. chemistry, part 2: OH and photolysis sinks with `tas(n)` -> sinks(n), emeth(n)
6. land: pools(n) from n-1, rates(n) with CO2(n) and the temperature anomaly -> land flux(n)
7. ocean: sub-steps with the sea-surface temperature of n -> sea flux(n)

Fluxes, sinks and the CO2 from methane oxidation of step n enter the stocks at n+1. The split
of the chemistry into two parts follows from this order. `--scen=contrib_rfc` runs only
forcing -> ERF -> DOECLIM from prescribed forcing components (`run_rfc`).

Amounts are stored per step (emissions, fluxes, NPP in GtC/step or Mt/step), levels as states;
the registry records the time basis of every variable (section 5).

---

## 3. Module interface

Every physics module follows the same pattern:

| | rule |
|---|---|
| parameters | `load_<module>_params(cfg) -> <Module>Params`, a frozen dataclass built from a `RunConfig` |
| stateful step | `<module>_step(params, prev, drivers) -> state`; `prev` is the state after step n-1, or `None` at the first step |
| stateless step | `forcing_step(params, drivers)` (forcing keeps no state) |
| drivers | one frozen dataclass per module (`LandDrivers`, `OceanDrivers`, ...) with the inputs of step n |
| run-wide options | parameters, not step arguments (e.g. `OceanParams.flux_bounds`, `DoeclimParams.t_lo`, `ref_compat`) |
| outputs | the state dataclasses' fields, collected with the module prefix (`lnd_`, `ocn_`, ...) |
| bounds | `ModelBoundsError(variable, value, bound)`, raised when a variable leaves its physical range |

Chemistry has two steps (`chemistry_step` before forcing, `chemistry_finish` after DOECLIM),
see section 2. Offline runners (`run_land`, `run_doeclim`, `run_ocean`, `run_hector`, ...) drive a
single module from given series, which is useful for calibration and tests.

### Ocean models

`physics.ocean_models.get_ocean_model(name)` returns an `OceanModel(name, prefix, load_params,
step, collect)`; both oceans take the same `OceanDrivers(co2, d_co2, tatm, d_sst)` and their
states expose `atmoSeaFlux`, `surface_cpool` and `total_cpool`. To add an ocean model:

1. a module with `load_<x>_params(cfg)`, `<x>_step(params, prev, OceanDrivers)` whose state has
   the three coupling attributes, and `collect_<x>(states)` returning prefixed outputs;
2. a branch in `get_ocean_model`, its name in `OCEAN_MODELS` and in the `--ocean_model` help;
3. registry rows for its outputs (section 5) and a regression case (section 9).

---

## 4. Configuration

`core.config.RunConfig` holds the options of one run; the process choices are grouped in
`ChemOptions`, `OceanOptions` and `ForcingOptions`. `config_from_flags(["--name=value", ...])`
builds a `RunConfig`: a flag is matched against the fields of the four classes, and is
converted by type (integers, floats, booleans `=0/=1`, `None` for `--cm6mdl=None`). Unknown
flags stop the run. `config.resolve(cfg)` applies the scenario's own settings (the RCMIP3
forced options and time horizons) and is used by `io.inputs.load_inputs`.

To add an option: a field with its default and a comment (the comment becomes its entry in
[options.md](options.md)); for a boolean, integer or float, its name in the conversion lists
of `config_from_flags`; then read it in the module's `load_<module>_params` (not in the step
function).

Data files are found with `config.data_path(dat_dir, ...)`: the data folder (`$SCM4OPT_DATA`,
`--dat_dir`), else the package's `data/` folder.

---

## 5. Names, units and the registry

* Outputs: `<prefix>_<name>` with the module prefix (user guide, section 5); `D_` for an
  anomaly from the pre-industrial state.
* Parameters: snake_case `<quantity>_<species/box>_<qualifier>` (`tau_ch4_oh`, `alpha_n2o`);
  values per step end in `_per_step`.

`data/registry/registry.csv` has one row per output, report symbol and parameter: `name`,
`kind` (output | report | parameter), `module`, `dims`, `unit`, `time_basis` (state | per_step |
per_year | none), `description`, `reference`, `report_order`, `report_group`. It is the single
source of:

* the report symbols and their order (`report_order`), and which of them are parameters
  (`report_group`), used by `io.report`;
* the per-step variables, which the RCMIP3 export sums into yearly values;
* the unit checks (`python3 -m scm4opt.core.registry`, `tests/test_registry.py`): every unit
  parses in the grammar of `core.units`, the time basis agrees with the unit (`/step` <->
  per_step, `/yr` <-> per_year), names are unique, and every RCMIP3 export conversion
  (mapping scale x steps per year) equals the factor implied by the registry unit and the
  RCMIP unit; `check_outputs(out)` checks that every array of a run is registered with its
  number of dimensions.

Units are written as `ppm`, `W/m2`, `GtC/step`, `Mt N/step`, `kt HFC23/step`, `1e22 J`, `1`,
... (grammar in `core/units.py`); species conversions go through a shared element
(GtC -> Mt CO2, Mt N -> Mt N2O).

**Adding an output:** return it from the module (with the prefix), add a registry row with
dims, unit, time basis and description, and run the registry tests. A report symbol also
gets a `report_order`.

---

## 6. Data and parameter tables

* Parameter tables are CSV files in `data/` (e.g. `data/tables/halo_parameters.csv`,
  `data/CMIP6/*_gcm_est.csv`, `data/nat/*.csv`). Edit the CSV files; the modules only load
  them. Run the regression tests after any change.
* `scenarios/rcmip3_data.py` is maintained by hand: experiment time ranges, the list of
  RCMIP3 experiments with their standard options, and the per-experiment forcing options.
  A new experiment needs an entry there and its input files in `data/RCMIP3/`.
* Scenario inputs: after adding or changing files in `RCMIP3/`, `contrib/`, `AGAGE/` or
  `NOAA/`, rewrite the manifest with `scm4opt data manifest` (and `scm4opt data verify`).
* `data/MAGICC6/` keeps the license of the MAGICC6 distribution (CC BY-NC-SA 3.0); do not
  move other data into it, and do not copy its values into code.

---

## 7. Report and RCMIP3 export

`io.report.build_report(out, inp, doeclim_params)` builds the report symbols (registry order)
from a run; `write_report_csv` writes them per step, `rcmip_rows` / `write_rcmip_csv` the RCMIP3
rows of one member. The RCMIP3 rows use `data/RCMIP3_src/output_mapping.csv` (RCMIP variable
-> report symbol, offset, scale) and the template header
`rcmip3-data-submission-template-v1-1-0_columns.csv`; per-step variables are summed per year
(x6), states take the value at the start of the year. `io.rcmip_export` stacks the members of
an ensemble into one file per experiment.

---

## 8. Reference compatibility

The default behaviour reproduces the SCM4OPT v4.1 reference results exactly. Where the
reference contains a known quirk, the corrected behaviour is implemented next to it and
selected with `RunConfig.ref_compat` (`--ref_compat=0`); the parameter classes carry the flag
(e.g. `OceanParams.ref_compat`). The switched items are listed in the user guide
(section 6). Rules:

* never change the default results silently: a change of the default needs new regression
  references and a changelog entry;
* a new quirk fix goes behind `ref_compat`, with a test of the corrected branch in
  `tests/test_compat.py`.

---

## 9. Tests

```bash
pytest                        # all tests (pythonpath and test paths are set in pyproject.toml)
python3 -m scm4opt.core.registry   # registry and unit checks
scm4opt data verify           # scenario inputs against their checksums
```

| file | checks |
|---|---|
| `tests/test_regression.py` | four reference cases (contrib, rcmip3 with prescribed concentrations, rcmip3 with observations, Hector ocean) against `tests/reference/*.npz` (annual values of every output; tolerance 1e-10 relative) |
| `tests/test_registry.py` | registry, units, RCMIP3 conversions, outputs of a run registered |
| `tests/test_compat.py` | the corrected branches of `--ref_compat=0` (mass balance, heat, derivatives, conversions) |
| `tests/test_data.py` | inputs against the manifest; pack/fetch round trip |
| `tests/test_export.py` | ensemble export reproduces the member values |

Tests whose inputs are missing are skipped. GitHub Actions (`.github/workflows/tests.yml`)
runs the registry check, the data verification and pytest on Python 3.10 and 3.13.

---

## 10. Versions, licenses, citation

* `core/version.py` holds `__version__` (package, `pyproject.toml`, logs, `scm4opt --version`)
  and the RCMIP3 model name `SCM4OPT(PY)v4.0`; the file prefix `SCM4OPTv40_` is kept.
* Every source file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`. Parts derived
  from other models are credited in `NOTICE`; keep those credits when moving code.
* `CITATION.cff` lists how to cite the model.

---

## 11. How this repository is produced

This repository is generated from the authors' development tree, which also holds the
comparison with the original implementation of SCM4OPT. Every release is checked to give the
same outputs, bit for bit, as the development version on the regression cases. Please send
changes as issues or pull requests; the maintainers apply them in the development tree, so
they are kept in the next release.
