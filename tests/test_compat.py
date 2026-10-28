"""--ref_compat=0 (corrected behaviour instead of the reference quirks; README "Reference
compatibility"): the corrected branches are consistent; the default is tested by
test_regression.py."""

import numpy as np
import pytest

from scm4opt.core.config import config_from_flags
from scm4opt.core.registry import check_rcmip_mapping
from scm4opt.io.inputs import load_inputs
from scm4opt.io.report import load_output_mapping
from scm4opt.physics import chemistry as C
from scm4opt.physics.doeclim import load_doeclim_params
from scm4opt.physics.ocean_oscar import load_ocean_params
from scm4opt.scenarios.forward import run

FLAGS = ["--scen=contrib", "--psrc=IAMC", "--psce=SSP1_19"]


def _run(flags):
    try:
        inp = load_inputs(config_from_flags(flags))
    except FileNotFoundError as e:
        pytest.skip(f"inputs not available ({e})")
    return inp, run(inp.cfg, inp)


@pytest.fixture(scope="module")
def corrected():
    return _run(FLAGS + ["--ref_compat=0"])


def test_ocean_mass_balance(corrected):
    inp, o = corrected
    p = load_ocean_params(inp.cfg)
    dc = np.diff(o["ocn_D_CSURF_all"])
    export = (o["ocn_D_FCIRC"][:-1].sum(1) + o["ocn_D_FCIRC_sub"][1:, :, :-1].sum((1, 2))) * p.ocn_dt_sub
    assert np.abs(o["ocn_atmoSeaFlux"][1:] - dc - export).max() < 1e-10


def test_heat_diagnostics(corrected):
    inp, o = corrected
    p = load_doeclim_params(inp.cfg, inp.t.n)
    assert np.allclose(o["doe_heatcontent"], o["doe_heat_mixed"] + o["doe_heat_interior"], rtol=0, atol=1e-9)
    assert np.allclose(np.diff(o["doe_heatcontent"]), o["doe_heatflux"][1:] * p.dt, rtol=0, atol=1e-9)
    assert np.allclose(o["doe_heatflux_mixed"][1:], p.heatcap_mixed * np.diff(o["doe_temp_sst"]) / p.dt)


def test_derivatives(corrected):
    inp, _ = corrected
    p = C.load_chem_params(inp.cfg)
    x = dict(D_CH4=900.0, D_O3s=5.0, T=1.2, ENOX=20.0, ECO=300.0, EVOC=50.0)
    h = 1e-6
    fd = (C.f_kOH(p, **{**x, "T": 1.2 + h}) - C.f_kOH(p, **{**x, "T": 1.2 - h})) / (2 * h)
    assert np.isclose(C.df_kOH(p, **x, ref_compat=False)["gst"], fd, rtol=1e-6)
    fd = (C.f_hv(p, 300.0, 1500.0, 1.2 + h) - C.f_hv(p, 300.0, 1500.0, 1.2 - h)) / (2 * h)
    assert np.isclose(C.df_hv(p, 300.0, 1500.0, 1.2, ref_compat=False)["gst"], fd, rtol=1e-6)


def test_rcmip_conversions_corrected():
    assert check_rcmip_mapping(load_output_mapping(ref_compat=False), known=()) == []


def test_default_differs_only_where_documented(corrected):
    _, ref = _run(FLAGS)
    _, cor = corrected
    assert not np.array_equal(ref["ocn_atmoSeaFlux"], cor["ocn_atmoSeaFlux"])
    assert abs(ref["scm_atmoCO2"][-1] - cor["scm_atmoCO2"][-1]) < 2.0   # ppm
