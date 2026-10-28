\
"""RCMIP3 ensemble export (scm4opt/io/rcmip_export.py): members written by the model
are stacked into one ensemble file without changing a value."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from scm4opt.core.config import config_from_flags
from scm4opt.io.inputs import load_inputs
from scm4opt.io.rcmip_export import main as export_main
from scm4opt.scenarios.forward import run, write_report


def test_export_two_members(tmp_path):
    try:
        inp = load_inputs(config_from_flags(["--scen=rcmip3", "--rname=ssp585", "--load_conc=all",
                                             "--report_csv=0"]))
    except FileNotFoundError as e:
        pytest.skip(f"inputs not available ({e})")
    out = run(inp.cfg, inp)
    runs = tmp_path / "rcmip_ssp585"
    for case in ("000000", "000001"):
        cfg = dataclasses.replace(inp.cfg, case=case, rout3=str(runs))
        write_report(out, dataclasses.replace(inp, cfg=cfg))
    assert export_main([f"--csv_dir={tmp_path}", f"--out_dir={tmp_path / 'ens'}", "--sce=ssp585",
                        "-j", "1"]) == 0
    rd = lambda p: pd.read_csv(p, dtype={"ensemble_member": str}, float_precision="round_trip")
    ens = rd(tmp_path / "ens" / "SCM4OPTv40_rcmip_phase3_ssp585_ens.csv")
    mem = pd.concat([rd(p) for p in sorted(runs.glob("SCM4OPTv40_*_00000[01].csv"))])
    assert list(ens.columns) == list(mem.columns)
    assert ens["ensemble_member"].tolist() == mem["ensemble_member"].tolist()
    years = [c for c in mem.columns if c.isdigit()]
    assert np.array_equal(ens[years].to_numpy(float), mem[years].fillna(0).to_numpy(float))
