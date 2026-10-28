# SPDX-License-Identifier: GPL-3.0-or-later
"""Combine the ensemble members of RCMIP3 experiments into RCMIP3 submission files.

An ensemble run (ensemble.py) writes per member, in <csv_dir>/rcmip_<sce><suffix>/:
  SCM4OPTv40_rcmip_phase3_<export_sce>_<case>.csv  RCMIP3 rows of one member (report.py)
  <sce>_<case>_all.csv                             model report, one row per step
This combines one experiment's members into
  <out_dir>/SCM4OPTv40_rcmip_phase3_<export_sce>_ens.csv
with the columns of the RCMIP3 submission template (one row per variable and ensemble
member, members in case order). <export_sce> is <sce>, or <sce>C for the
concentration-driven ScenarioMIP experiments (report.SCENARIOMIP_CONC).

  --src member  (default) stack the per-member RCMIP3 CSVs as written by the runs.
  --src report  rebuild the rows from the per-step report CSVs with output_mapping.csv:
                yearly means over all steps of a year, all-zero series kept (the member
                CSVs follow the reference submission files instead, see report.rcmip_rows).

Members are read in parallel and written one after the other, so memory use stays at a
few members whatever the ensemble size. Values are written at full precision.

Usage:
  python3 -m scm4opt.io.rcmip_export --sce=esm-allGHG-ssp119
  python3 -m scm4opt.io.rcmip_export --csv_dir=/path/to/runs --out_dir=/path/to/ens
  python3 -m scm4opt.io.rcmip_export --sce=ssp585 --sce=ssp126 -j 8
  python3 -m scm4opt.io.rcmip_export --sce=ssp585 --src=report
  python3 -m scm4opt.io.rcmip_export --sce=ssp585 --run_suffix=_hector   # Hector ensemble
  python3 -m scm4opt.io.rcmip_export                 # every RCMIP3 experiment found
Options (also with "-" instead of "_"):
  --sce NAME        experiment (repeat or comma-separate; default: all 135, those found)
  --csv_dir DIR     folder holding rcmip_<sce>/ (default: $mdl_dir, else as ensemble.py)
  --out_dir DIR     output folder (default: <csv_dir>/rcmip3_ens<run_suffix>)
  --out_csv FILE    output file, with a single --sce only
  --run_suffix S    folder suffix of the runs, e.g. _hector (rcmip_<sce>_hector/)
  --src member|report
  --climate_model NAME  climate_model of the rows (default SCM4OPT(PY)v4.0, version.py;
                    "keep" keeps the members' value). Members written before the rename
                    (SCM4OPTv4.0) are relabelled, so they need no re-run for the name.
  -j, --jobs N      parallel readers (default: CPUs - 1, at most 16)
  --outmap_csv F / --template_csv F   mapping / template header (default: package data)
Exit code 1 if no experiment was exported.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import DAT_DIR, data_path
from ..scenarios.rcmip3_data import RCMIP3_ALL_SCENARIOS
from .report import SCENARIOMIP_CONC, load_output_mapping, template_columns
from ..core.version import RCMIP_CLIMATE_MODEL, RCMIP_FILE_PREFIX

CLIMATE_MODEL = RCMIP_CLIMATE_MODEL            # default climate_model of the written rows
FILE_STEM = RCMIP_FILE_PREFIX


def export_name(sce):
    return f"{sce}C" if sce in SCENARIOMIP_CONC else sce


def member_files(rsl_dir, sce, src):
    """[(case, path)] of one experiment's members, sorted by case."""
    if src == "member":
        pat = re.compile(rf"^{re.escape(FILE_STEM + export_name(sce))}_([0-9]{{6}})\.csv$")
    else:
        pat = re.compile(rf"^{re.escape(sce)}_([0-9]{{6}})_all\.csv$")
    out = []
    for p in Path(rsl_dir).iterdir():
        m = pat.match(p.name)
        if m:
            out.append((m.group(1), p))
    return sorted(out)


def _year_cols(cols):
    return [c for c in cols if c.isdigit()]


# ---------------------------------------------------------------- one member
def rows_from_member(path):
    """The member's RCMIP3 rows as written by the run."""
    return pd.read_csv(path, dtype={"ensemble_member": str}, float_precision="round_trip")


def rows_from_report(path, export_sce, member, mapping, year_cols):
    """The member's RCMIP3 rows rebuilt from its per-step report CSV."""
    hdr = pd.read_csv(path, nrows=0).columns
    use = [v for v in dict.fromkeys(mapping["rpt_var"]) if v in hdr]
    rpt = pd.read_csv(path, usecols=["t"] + use, float_precision="round_trip")
    t = rpt["t"].to_numpy(dtype=float)
    years = np.floor(t + 1e-6).astype(int)
    whole = np.abs(t - np.round(t)) < 1e-6
    rows = []
    for _, mp in mapping.iterrows():
        v = mp["rpt_var"]
        if v not in rpt:
            continue
        x = rpt[v].to_numpy(dtype=float)
        if mp["aggregate_to_year"]:
            s = pd.Series(x).groupby(years).mean()
        else:
            s = pd.Series(x[whole], index=np.round(t[whole]).astype(int))
        s = s.dropna()
        if s.empty:
            continue
        s = (s + mp["rpt_var0"]) * mp["rpt_var_scale"] * mp["time_scale"]
        row = dict(climate_model=CLIMATE_MODEL, model="unspecified", scenario=export_sce,
                   region="World", variable=mp["Variable"],
                   unit=str(mp["Unit"]).replace("MAGICC ", ""), ensemble_member=member)
        row.update({str(y): val for y, val in s.items() if str(y) in year_cols})
        rows.append(row)
    return pd.DataFrame(rows)


def _read(job):
    """Worker: one member's rows (job = (src, path, export_sce, member, mapping, years))."""
    src, path, export_sce, member, mapping, year_cols = job
    if src == "member":
        return rows_from_member(path)
    return rows_from_report(path, export_sce, member, mapping, year_cols)


# ---------------------------------------------------------------- one experiment
def export_experiment(sce, csv_dir, out_path, src="member", run_suffix="", jobs=1,
                      mapping=None, template=None, log=print, climate_model=CLIMATE_MODEL):
    """Write one experiment's ensemble file. Returns (path, rows, members, first, last),
    or None (with the reason in the log) if there is nothing to export.

    climate_model : the climate_model written in every row (default: the Python model's
                    RCMIP3 name, version.RCMIP_CLIMATE_MODEL); None keeps the members' value.
    """
    rsl_dir = Path(csv_dir) / f"rcmip_{sce}{run_suffix}"
    if not rsl_dir.is_dir():
        return None
    files = member_files(rsl_dir, sce, src)
    if not files:
        log(f"{sce}: no {'member RCMIP3 CSVs' if src == 'member' else 'report CSVs'} in {rsl_dir}")
        return None
    log(f"{sce}: {len(files)} members in {rsl_dir} (src={src})")
    cols = template if template is not None else template_columns()
    ens = cols.index("ensemble_member")
    meta_cols, tmpl_years = cols[:ens + 1], cols[ens + 1:]
    mapping = mapping if mapping is not None else load_output_mapping()
    export_sce = export_name(sce)
    jobs_ = [(src, p, export_sce, str(int(case)), mapping, tmpl_years) for case, p in files]

    # the year range: from the member headers (member CSVs) or from the first member
    if src == "member":
        avail = set()
        for _, p in files:
            avail.update(int(c) for c in _year_cols(pd.read_csv(p, nrows=0).columns))
    else:
        avail = {int(c) for c in _year_cols(_read(jobs_[0]).columns)}
    if not avail:
        raise ValueError(f"{sce}: no year columns in the member files")
    years = [str(y) for y in range(min(avail), max(avail) + 1) if str(y) in tmpl_years]
    out_cols = meta_cols + years

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, members, names_in = 0, set(), set()
    tmp = out_path.with_name(out_path.name + ".part")
    with open(tmp, "w", newline="") as fh:
        fh.write(",".join(out_cols) + "\n")
        pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
        try:
            parts = pool.map(_read, jobs_, chunksize=4) if pool else map(_read, jobs_)
            for df in parts:
                if df.empty:
                    continue
                df = df.reindex(columns=out_cols)
                df[years] = df[years].fillna(0.0)
                df = df.fillna(0)
                names_in.update(df["climate_model"].astype(str))
                if climate_model is not None:
                    df["climate_model"] = climate_model
                df.to_csv(fh, header=False, index=False)
                n_rows += len(df)
                members.update(df["ensemble_member"].astype(str))
        finally:
            if pool:
                pool.shutdown()
    if n_rows == 0:
        tmp.unlink()
        return None
    os.replace(tmp, out_path)
    log(f"{out_path} saved: {n_rows} rows, {len(members)} members, {years[0]}-{years[-1]}")
    if climate_model is not None and names_in - {climate_model}:
        log(f"  climate_model {', '.join(sorted(names_in))} -> {climate_model}")
    return out_path, n_rows, len(members), years[0], years[-1]


# ---------------------------------------------------------------- command line
def default_csv_dir():
    """$mdl_dir, else the folder ensemble.py writes to by default."""
    if os.environ.get("mdl_dir"):
        return Path(os.environ["mdl_dir"])
    from ..core.config import REPO_DIR
    return REPO_DIR.parent if (REPO_DIR / "mpi" / "rcmip_unc.sh").exists() else Path.cwd()


def _parser():
    ap = argparse.ArgumentParser(prog="rcmip_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sce", action="append", default=None)
    ap.add_argument("--csv_dir", "--csv-dir", type=Path, default=None)
    ap.add_argument("--out_dir", "--out-dir", type=Path, default=None)
    ap.add_argument("--out_csv", "--out-csv", type=Path, default=None)
    ap.add_argument("--run_suffix", "--run-suffix", default="")
    ap.add_argument("--src", choices=("member", "report"), default="member")
    ap.add_argument("--ref_compat", "--ref-compat", type=int, choices=(0, 1), default=1,
                    help="--src report: 0 uses the corrected heat-content/uptake conversions")
    ap.add_argument("--climate_model", "--climate-model", default=CLIMATE_MODEL,
                    help=f"climate_model of the rows (default {CLIMATE_MODEL}; keep: as in the members)")
    ap.add_argument("-j", "--jobs", "--nthread", type=int,
                    default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--outmap_csv", "--outmap-csv", type=Path, default=None)
    ap.add_argument("--template_csv", "--template-csv", "--template_file", type=Path, default=None)
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    scenarios = (list(RCMIP3_ALL_SCENARIOS) if not args.sce else
                 [s for a in args.sce for s in a.split(",") if s])
    csv_dir = (args.csv_dir or default_csv_dir()).expanduser()
    if not csv_dir.is_dir():
        sys.exit(f"--csv_dir not found: {csv_dir}")
    out_dir = (args.out_dir or csv_dir / f"rcmip3_ens{args.run_suffix}").expanduser()
    if args.out_csv and len(scenarios) != 1:
        sys.exit("--out_csv needs a single --sce")
    mapping = load_output_mapping(args.outmap_csv or data_path(DAT_DIR, "RCMIP3_src", "output_mapping.csv"),
                                  ref_compat=bool(args.ref_compat))
    template = template_columns(args.template_csv)
    saved, no_folder, no_files = 0, [], []
    for sce in scenarios:
        rsl_dir = csv_dir / f"rcmip_{sce}{args.run_suffix}"
        if not rsl_dir.is_dir():
            no_folder.append(sce)
            continue
        out = args.out_csv or out_dir / f"{FILE_STEM}{export_name(sce)}_ens.csv"
        r = export_experiment(sce, csv_dir, out, args.src, args.run_suffix, args.jobs,
                              mapping, template, log=lambda m: print(m, flush=True),
                              climate_model=None if args.climate_model == "keep" else args.climate_model)
        if r is None:
            no_files.append(sce)
        else:
            saved += 1
    print(f"exported {saved} of {len(scenarios)} experiments from {csv_dir} to "
          f"{args.out_csv or out_dir}")
    if no_files:
        print(f"  folder without member files: {' '.join(no_files)}")
    if no_folder:
        print(f"  no folder {csv_dir}/rcmip_<experiment>{args.run_suffix} for {len(no_folder)} "
              f"experiments" + (f": {' '.join(no_folder)}" if len(no_folder) <= 10 else ""))
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
