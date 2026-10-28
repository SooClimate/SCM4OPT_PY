"""Regression tests: the model reproduces the reference outputs stored by the export
(tests/reference/<case>.npz, written from the development package).

Needs the scenario inputs in scm4opt/data/ (or $SCM4OPT_DATA); cases whose inputs are missing
are skipped.
"""

from pathlib import Path

import numpy as np
import pytest

from scm4opt.core.config import DAT_DIR, config_from_flags
from scm4opt.io.inputs import load_inputs
from scm4opt.scenarios.forward import run

REF = Path(__file__).parent / "reference"
STRIDE = 6                 # references hold every STRIDE-th step (annual)
CASES = {'contrib_ssp1_19': ['--scen=contrib', '--psrc=IAMC', '--psce=SSP1_19'], 'rcmip3_ssp585_conc': ['--scen=rcmip3', '--rname=ssp585', '--load_conc=all'], 'rcmip3_historical_obs': ['--scen=rcmip3', '--rname=historical', '--load_conc=nco2', '--load_AGAGE=1', '--load_NOAA=1'], 'rcmip3_ssp126_hector': ['--scen=rcmip3', '--rname=ssp126', '--ocean_model=hector', '--cm6mdl=MIROC6', '--ch4rfc=myhre']}


@pytest.mark.parametrize("case", sorted(CASES))
def test_regression(case):
    try:
        inp = load_inputs(config_from_flags(CASES[case]))
        out = run(inp.cfg, inp)
    except FileNotFoundError as e:
        pytest.skip(f"inputs for {case} not available ({e}); data dir {DAT_DIR}")
    ref = np.load(REF / f"{case}.npz")
    missing = sorted(set(ref.files) - set(out))
    assert not missing, f"outputs missing: {missing[:10]}"
    for name in ref.files:
        np.testing.assert_allclose(np.asarray(out[name], float)[::STRIDE], ref[name],
                                   rtol=1e-10, atol=1e-12,
                                   err_msg=name)
