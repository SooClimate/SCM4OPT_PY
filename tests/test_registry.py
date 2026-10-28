"""Registry checks (scm4opt/core/registry.py): units, time bases, report order, the
RCMIP3 export conversions, and the outputs of a run against the registry."""

import pytest

from scm4opt.core.config import config_from_flags
from scm4opt.core.registry import (KNOWN_RCMIP_ISSUES, check_outputs, check_registry,
                                   default_mapping, rcmip_conversions)
from scm4opt.io.inputs import load_inputs
from scm4opt.io.report import REPORT_SYMBOLS
from scm4opt.scenarios.forward import run


def test_registry():
    assert check_registry(mapping=default_mapping()) == []


def test_known_rcmip_issues_still_present():
    """Update KNOWN_RCMIP_ISSUES when one of them is fixed."""
    c = rcmip_conversions(default_mapping())
    assert set(c.loc[c["problem"] != "", "rpt_var"]) - {"erf_O3"} == set(KNOWN_RCMIP_ISSUES)


def test_report_symbols():
    assert len(REPORT_SYMBOLS) == len(set(REPORT_SYMBOLS)) == 402


def test_outputs_registered():
    try:
        inp = load_inputs(config_from_flags(["--scen=contrib", "--psrc=IAMC", "--psce=SSP1_19"]))
    except FileNotFoundError as e:
        pytest.skip(f"inputs not available ({e})")
    assert check_outputs(run(inp.cfg, inp)) == []
