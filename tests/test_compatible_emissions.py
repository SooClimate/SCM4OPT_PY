"""Compatible CO2 emissions of a concentration-driven run (scm_eind_co2_compat): driving
the model with them reproduces the prescribed CO2 concentrations, and
--rcmip_compat_ems=1 exports them in the RCMIP3 rows."""

import numpy as np
import pytest

from scm4opt.core.config import config_from_flags
from scm4opt.io.inputs import load_inputs
from scm4opt.io.report import build_report, load_output_mapping, rcmip_rows
from scm4opt.physics.doeclim import load_doeclim_params
from scm4opt.scenarios.forward import run


def _inputs(flags):
    try:
        return load_inputs(config_from_flags(flags))
    except FileNotFoundError as e:
        pytest.skip(f"inputs not available ({e})")


def test_round_trip():
    a = _inputs(["--scen=rcmip3", "--rname=ssp585", "--load_conc=all"])
    oa = run(a.cfg, a)
    b = _inputs(["--scen=rcmip3", "--rname=ssp585", "--load_conc=nco2",
                 f"--co2_conc_initial={float(oa['scm_atmoCO2'][0])!r}"])
    co2 = b.gi("co2")
    b.eind[:, co2] = oa["scm_eind_co2_compat"]
    b.ems[:, co2] = b.eind[:, co2] + b.elnd[:, co2]
    ob = run(b.cfg, b)
    assert np.abs(ob["scm_atmoCO2"] - oa["scm_atmoCO2"]).max() < 1e-9      # ppm
    assert np.abs(ob["scm_tatm"] - oa["scm_tatm"]).max() < 1e-12           # K
    assert np.allclose(oa["scm_ems_co2_compat"] - oa["scm_eind_co2_compat"], a.elnd[:, co2])
    assert "scm_eind_co2_compat" not in ob


def test_rcmip_rows():
    """--rcmip_compat_ems=1 changes only the two CO2 emission rows, to the yearly sums of
    the compatible emissions."""
    flags = ["--scen=rcmip3", "--rname=ssp585", "--load_conc=all"]
    a, b = _inputs(flags), _inputs(flags + ["--rcmip_compat_ems=1"])
    out = run(a.cfg, a)
    rep = build_report(out, a, load_doeclim_params(a.cfg, a.t.n))
    ra = rcmip_rows(rep, a).set_index("variable")
    rb = rcmip_rows(rep, b).set_index("variable")
    years = [c for c in ra.columns if c.isdigit()]
    changed = {v for v in ra.index if not ra.loc[v, years].equals(rb.loc[v, years])}
    assert changed == {"Emissions|CO2", "Emissions|CO2|Fossil and Industry"}
    yr = np.floor(a.t.tval + 1e-6).astype(int)
    scale = load_output_mapping().set_index("Variable")["rpt_var_scale"]     # GtC -> MtCO2
    for v, name in (("Emissions|CO2", "scm_ems_co2_compat"),
                    ("Emissions|CO2|Fossil and Industry", "scm_eind_co2_compat")):
        for y in (1900, 2020, 2100):
            want = out[name][yr == y].sum() * scale[v]
            assert np.isclose(float(rb.loc[v, str(y)]), want, rtol=1e-12)
