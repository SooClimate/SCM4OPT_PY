# SPDX-License-Identifier: GPL-3.0-or-later
"""SCM4OPT coupled forward model.

The modules are solved as a sequential time loop: every equation of step n depends
only on earlier equations of the same step or on earlier steps. Within step n:

  1. atmosphere   A(n) from the fluxes of n-1           -> CO2(n), D_CO2(n)
  2. chemistry    stocks (sinks of n-1), O3, EESC, aerosols (tatm of n-1)
  3. forcing      RF / ERF(n)                           -> scm_ftot(n)
  4. DOECLIM      TL, TS, tas(n)
  5. chemistry    OH / photolysis with tas(n)           -> sinks(n), emeth(n)
  6. land         pools(n) from n-1, rates(n) with CO2(n) and deltem(n) -> LndFlux(n)
  7. ocean        sub-steps with D_sst(n)               -> SeaFlux(n)
Fluxes, sinks and emeth of step n enter the stocks at n+1.

Usage (from the directory that contains scm4opt/):
    python3 -m scm4opt [--flag=value ...] [--rout3=<dir>]
with --flag=value options: the fields of config.RunConfig and its option classes
(e.g. --psce=SSP1_19 --cm6mdl=MIROC6; full list in docs/options.md of the repository).
--rout3 writes the model report (report.py) as <dir>/<rname>_<case>_<ctag>.csv
and the RCMIP3 submission rows as SCM4OPTv40_rcmip_phase3_<scenario>_<case>.csv

Scenario drivers:
  --scen=contrib      emission-driven contribution runs (default); optionally --fix_conc
  --scen=contrib_rfc  prescribed forcing components -> ERF -> DOECLIM (run_rfc)
  --scen=rcmip3       RCMIP3 experiments, --rname=<exp> sets the time range (rcmip3_data.py);
                      optional --load_conc=all|nco2:
                        all   CO2, CH4, N2O and the AGAGE-list halogens prescribed
                        nco2  as all, but CO2 stays emission-driven
                      --load_AGAGE / --load_NOAA overwrite them with observations

Examples:
  # contrib
  python3 -m scm4opt --scen=contrib --psrc=IAMC --psce=SSP1_19
  python3 -m scm4opt --scen=contrib --psce=SSP1_19 --cm6mdl=MIROC6 --rout3=out/
  # --fix_conc=1 needs data/contrib/<hsrc>_<psrc>/conc/<psce>_WORLD.csv
  python3 -m scm4opt --scen=contrib --psce=SSP1_19 --fix_conc=1

  # contrib_rfc (forcing data in <fol>/rfc_<X>/..._all.csv, one ensemble member)
  python3 -m scm4opt --scen=contrib_rfc --fol=<folder> --case=000000
  python3 -m scm4opt --scen=contrib_rfc --fol=<folder> --case=000007 \\
      --minus_rfc=rfc_HALO_cfc12 --tyear=2000 --fraction=0.5

  # rcmip3
  python3 -m scm4opt --scen=rcmip3 --rname=esm-ssp585 --rout3=out/
  python3 -m scm4opt --scen=rcmip3 --rname=ssp585 --load_conc=all --rout3=out/
  python3 -m scm4opt --scen=rcmip3 --rname=historical --load_conc=nco2 \\
      --load_AGAGE=1 --load_NOAA=1

Ocean carbon cycle (--ocean_model):
  osc     OSCAR-type mixed-layer ocean (default)
  hector  ocean_hector.py (Hector v1.0 four-box ocean), with
          --cm6mdl calibration from data/CMIP6/ocean_gcm_est.csv and optional --same_alk=1
  python3 -m scm4opt --scen=rcmip3 --rname=ssp585 --ocean_model=hector --rout3=out/

Reference compatibility: --ref_compat=1 (default) reproduces the SCM4OPT v4.1 reference
results including their known quirks; --ref_compat=0 uses the corrected behaviour
(ocean carbon mass balance, Hector tcre rates, ocean heat diagnostics, RCMIP3 heat
export and yearly means; README: Reference compatibility).

Not supported: --use_ncycle (the N-cycle module is not implemented), --scen=ssp1..5.
"""

import sys
from dataclasses import fields

import numpy as np

from ..physics.atmosphere import AtmosphereDrivers, atmosphere_step, load_atmosphere_params
from ..physics.chemistry import (GWP_SUMS, ChemDrivers, ChemFinishDrivers, ChemStep, chemistry_finish,
                        chemistry_step, gwp_sums, load_chem_params)
from ..core.config import RunConfig
from ..core.constants import CO2_CONC_1850, PPM2GTC
from ..physics.doeclim import (DoeclimDrivers, DoeclimStep, delsst, doeclim_step, glue, load_doeclim_params,
                      scm_ftot)
from ..physics.forcing import ForcingDrivers, ForcingStep, forcing_from_data, forcing_step, load_forcing_params
from ..io.inputs import load_inputs
from ..physics.land import LandDrivers, LandRates, deltem, land_step, load_land_params
from ..physics.ocean_models import OceanDrivers, get_ocean_model
from ..core.errors import ModelBoundsError
from ..core.version import __version__

_UNSUPPORTED_SCEN = ("inv_rcmip", "ssp1", "ssp2", "ssp3", "ssp4", "ssp5")


def load_params(cfg, nsteps):
    """All module parameters for one configuration."""
    return dict(
        atmosphere=load_atmosphere_params(cfg),
        chem=load_chem_params(cfg),
        forcing=load_forcing_params(cfg),
        doeclim=load_doeclim_params(cfg, nsteps),
        land=load_land_params(cfg),
        ocean=get_ocean_model(cfg.ocean.ocean_model).load_params(cfg),
    )


def run(cfg=None, inp=None):
    """Run the coupled model. Returns {variable name: array over t}.

    Per-gas, per-box and per-region variables have extra trailing dimensions
    (e.g. chm_D_HALO is (N, 48), ocn_D_CSURF_sub is (N, 8, nsub)).
    """
    inp = inp or load_inputs(cfg or RunConfig())
    cfg = inp.cfg                       # resolved (rcmip3 overrides, time horizon)
    if cfg.use_ncycle:
        raise NotImplementedError("--use_ncycle: the N-cycle module is not implemented yet")
    if cfg.scen in _UNSUPPORTED_SCEN:
        raise NotImplementedError(f"--scen={cfg.scen} is not supported by the coupled run")
    if cfg.scen == "contrib_rfc":
        return run_rfc(cfg, inp)
    if cfg.fix_conc and (cfg.scen != "contrib" or cfg.load_conc):
        raise ValueError("--fix_conc is defined for --scen=contrib without --load_conc")
    N = inp.t.n
    pc_ = inp.conc                      # prescribed concentrations, or None
    P = load_params(cfg, N)
    pa, pc, pf, pl, po = P["atmosphere"], P["chem"], P["forcing"], P["land"], P["ocean"]
    doe = None                          # DoeclimState
    gi, halo_idx, ico2, ibc = inp.gi, inp.halo_idx, inp.gi("co2"), inp.gi("bc")
    om = get_ocean_model(cfg.ocean.ocean_model)       # ocean plug-in (ocean_models.py)

    rec = {k: [] for k in ("chem", "forcing", "doeclim", "land", "ocean")}
    scm = {k: np.zeros(N) for k in ("atmo_pool", "atmoCO2", "D_CO2", "tatm", "sst", "deltem",
                                    "delsst", "cca", "ftot")}
    chem = land = ocean = None
    for n in range(N):
        # 1. atmosphere (scm_ATMOEQ, scm_ATMOCO2EQ, scm_D_CO2_EQ)
        if cfg.load_conc == "all":      # CO2 prescribed
            co2 = pc_.D_CO2[n] + pc.co2_0
            A = co2 * PPM2GTC
        else:
            A = atmosphere_step(pa, None if n == 0 else scm["atmo_pool"][n - 1],
                                None if n == 0 else AtmosphereDrivers(
                                    inp.eind[n, ico2], chem.emeth, ocean.atmoSeaFlux,
                                    land.atmoLndFlux))
            # --fix_conc: CO2 prescribed, the pool stays emission-driven (diagnostic only)
            co2 = inp.fix_co2[n] + pc.co2_0 if cfg.fix_conc else A / PPM2GTC
        if co2 < 1 or (not cfg.load_conc and not 1 <= A <= 1e5):
            raise ModelBoundsError("scm_atmo_pool / scm_atmoCO2", A if co2 >= 1 else co2,
                                  "pool [1, 1e5] GtC, CO2 >= 1 ppm")
        d_co2 = co2 - pc.co2_0
        scm["atmo_pool"][n], scm["atmoCO2"][n], scm["D_CO2"][n] = A, co2, d_co2
        scm["cca"][n] = 0.0 if n == 0 else scm["cca"][n - 1] + inp.ems[n - 1, ico2]

        # 2. chemistry, temperature-independent part
        tatm_prev = scm["tatm"][n - 1] if n else 0.0
        presc = None if pc_ is None else (pc_.D_CH4[n], pc_.D_N2O[n], pc_.D_HALO[n],
                                          pc_.halo_mask)
        chem = chemistry_step(pc, chem, ChemDrivers(inp.ems[n], inp.ems_1850, gi, halo_idx,
                                                    tatm_prev, presc))

        # 3. forcing
        f = forcing_step(pf, ForcingDrivers(d_co2, chem, inp.ems[n, ibc], inp.land_cover[n],
                                            inp.rf_con[n], inp.rf_volc[n], inp.rf_solar[n],
                                            inp.rf_dust[n], pc.d_ebb_bc))
        scm["ftot"][n] = scm_ftot(f, cfg.co2_only)

        # 4. DOECLIM
        doe = doeclim_step(P["doeclim"], doe, DoeclimDrivers(scm["ftot"][n]))
        d = doe.out
        g = glue(P["doeclim"], d)
        scm["tatm"][n], scm["sst"][n] = g["scm_tatm"], g["scm_sst"]
        scm["delsst"][n] = delsst(scm["sst"][n - 1] if n else 0.0,
                                  scm["sst"][n - 2] if n > 1 else 0.0, n)

        # 5. chemistry sinks and CH4 oxidation (scm_D_tatm = doe_tas)
        chem = chemistry_finish(pc, chem, ChemFinishDrivers(inp.ems[n], gi, g["scm_D_tatm"]))

        # 6. land (scm_DELTEMEQ: from scm_tatm at n-1 and n-2)
        scm["deltem"][n] = deltem(scm["tatm"][n - 1] if n else 0.0,
                                  scm["tatm"][n - 2] if n > 1 else 0.0, n)
        land = land_step(pl, land, LandDrivers(co2, scm["deltem"][n], inp.elnd[n, ico2]))

        # 7. ocean (the selected plug-in uses the drivers it needs)
        ocean = om.step(po, ocean, OceanDrivers(co2=co2, d_co2=d_co2, tatm=g["scm_tatm"],
                                                d_sst=g["scm_D_sst"]))

        for k, v in (("chem", chem), ("forcing", f), ("doeclim", d), ("land", land),
                     ("ocean", ocean)):
            rec[k].append(v)

    return _collect(rec, scm, pc, om)


def run_rfc(cfg, inp):
    """--scen=contrib_rfc: prescribed forcing components -> ERF -> DOECLIM only;
    no carbon cycle or chemistry."""
    N = inp.t.n
    pf = load_forcing_params(cfg)
    pd_ = load_doeclim_params(cfg, N)
    doe = None
    d_ = inp.rfc_data
    fsteps, dsteps = [], []
    scm = {k: np.zeros(N) for k in ("ftot", "tatm", "sst")}
    for n in range(N):
        f = forcing_from_data(pf, {k: v[n] for k, v in d_.comp.items()}, d_.halo[n], d_.con[n])
        scm["ftot"][n] = f.erf_tot              # scm_ftot_EQ, contrib_rfc branch
        doe = doeclim_step(pd_, doe, DoeclimDrivers(scm["ftot"][n]))
        d = doe.out
        g = glue(pd_, d)
        scm["tatm"][n], scm["sst"][n] = g["scm_tatm"], g["scm_sst"]
        fsteps.append(f)
        dsteps.append(d)
    out = _stack(fsteps, ForcingStep, "") | _stack(dsteps, DoeclimStep, "doe_")
    for k in ("rfc_CO2_Cbar", "rfc_CH4_Mbar", "rfc_N2O_Nbar", "rfc_LCC_rgn", "rfc_LCC_lnd"):
        out.pop(k, None)                        # not defined in the contrib_rfc branch
    out |= {f"scm_{k}": v for k, v in scm.items()}
    return out


def _stack(steps, cls, prefix):
    return {prefix + f.name: np.array([getattr(s, f.name) for s in steps]) for f in fields(cls)}


def _collect(rec, scm, pc, om):
    """Arrays named as the model variables."""
    out = {}
    out |= _stack(rec["chem"], ChemStep, "chm_")
    sums = [gwp_sums(pc, s.D_HALO) for s in rec["chem"]]
    out |= {f"chm_{k}": np.array([d[k] for d in sums]) for k in GWP_SUMS}
    out |= _stack(rec["forcing"], ForcingStep, "")
    out |= _stack(rec["doeclim"], DoeclimStep, "doe_")
    out |= _stack(rec["land"], LandRates, "lnd_")
    out |= om.collect(rec["ocean"])
    # glue variables of the coupled model
    out |= {f"scm_{k}": v for k, v in scm.items()}
    out["scm_D_tatm"] = out["doe_tas"]
    out["scm_D_sst"] = out["doe_temp_sst"]
    out["scm_emeth"] = out["chm_emeth"]
    oc = om.prefix                                  # ocn (OSCAR) or hct (Hector)
    out["scm_atmoSeaFlux"] = out[f"{oc}_atmoSeaFlux"]
    out["scm_atmoLndFlux"] = out["lnd_atmoLndFlux"]
    out["lnd_landSink"] = out["lnd_atmoLndFlux"]
    out["scm_terrest_pool"] = out["lnd_terrest_pool"]
    out["scm_ocn_surface_cpool"] = out[f"{oc}_surface_cpool"]
    out["scm_ocn_total_cpool"] = out[f"{oc}_total_cpool"]
    for k in ("D_CH4", "D_N2O", "D_CH4_lag", "D_HALO", "D_O3t", "D_O3s", "D_EESC", "D_SO4",
              "D_POA", "D_BC", "D_NO3", "D_SOA", "D_DUST", "D_SALT", "D_AERh"):
        out[f"scm_{k}"] = out[f"chm_{k}"]
    return out


def write_report(out, inp):
    """--rout3: the model report as CSV (+ the RCMIP3 template CSV)."""
    from ..io.report import build_report, write_rcmip_csv, write_report_csv
    cfg = inp.cfg
    rep = build_report(out, inp, load_doeclim_params(cfg, inp.t.n))
    paths = [write_report_csv(rep, inp, cfg.rout3)] if cfg.report_csv else []
    if cfg.rcmip_csv:
        paths.append(write_rcmip_csv(rep, inp, cfg.rout3))
    return paths




def main(argv):
    from ..core.config import config_from_flags
    from ..core.version import version_line
    if argv[:1] in (["--version"], ["-V"]):
        print(version_line())
        return
    inp = load_inputs(config_from_flags(argv))
    cfg = inp.cfg
    out = run(cfg, inp)
    t = inp.t
    if cfg.rout3:
        if cfg.scen == "contrib_rfc":
            raise ValueError("--rout3 needs the full model; contrib_rfc has no carbon cycle")
        for path in write_report(out, inp):
            print(f"  wrote {path}")
    label = f"rname={cfg.rname} load_conc={cfg.load_conc}" if cfg.scen == "rcmip3" else f"psce={cfg.psce}"
    print(f"SCM4OPT Python {__version__}: scen={cfg.scen} {label} cm6mdl={cfg.cm6mdl} "
          f"steps={t.n} ({t.labels[0]}-{t.labels[-1]})")
    for y in dict.fromkeys(("1850", "1900", "2000", "2020", "2050", "2100", "2300", "2500", t.labels[-1])):
        if y in t.labels:
            i = t.index(y)
            co2 = (f"CO2 {out['scm_atmoCO2'][i]:8.3f} ppm  " if "scm_atmoCO2" in out else "")
            print(f"  {y}: {co2}tatm {out['scm_tatm'][i]:7.4f} K"
                  f"  ftot {out['scm_ftot'][i]:7.4f} W/m2")


if __name__ == "__main__":
    main(sys.argv[1:])
