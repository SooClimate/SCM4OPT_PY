# SPDX-License-Identifier: GPL-3.0-or-later
"""Ensemble runs of SCM4OPT through GNU parallel.

One model run per line of the member list (500 members, each a full --case=NNNNNN
--cm6mdl=... --ch4rfc=... --<option>_mdl=... set), fanned out by GNU parallel, so it
runs on a local desktop:

  parallel ... eval 'python3 -m scm4opt --scen=rcmip3 --rout3=${rsl_dir} \\
     --rname=<rname> --ctag=all --ems=CO2 --csec=EMISS_IND --rgn=WORLD {1}'

Each member writes, in rsl_dir (report.py):
  <rname>_<case>_all.csv                          model report, one row per step
  SCM4OPTv40_rcmip_phase3_<scenario>_<case>.csv   RCMIP3 submission rows

Usage (from anywhere):
  python3 -m scm4opt.ensemble --rname=esm-allGHG-ssp119
  python3 -m scm4opt.ensemble --rname=ssp585 -j 8 --cases=0-49
  python3 -m scm4opt.ensemble --rname=ssp585 --dry-run           # print the command
  python3 -m scm4opt.ensemble --rname=esm-ssp585 -- --load_conc=nco2   # extra model flags
  python3 -m scm4opt.ensemble --rname=ssp585 --ocean_model=hector     # Hector ocean
  python3 -m scm4opt.ensemble --list-rcmip3                          # all experiments
For --scen=rcmip3 the experiment's standard options are added (--load_conc=all for
concentration-driven, --load_conc=nco2 for esm-* experiments; rcmip3_data.py), unless
--no-standard-options or given explicitly after --. Missing inputs: exit code 3.

Directories (environment values are honoured when already set):
  mdl_dir  repository root (default: two levels above this file)
  rsl_dir  results, default ${mdl_dir}rcmip_<rname>/       (--out overrides)
  tmp_dir  member list + parallel job log, default ${mdl_dir}exe/<rname>/
With --ocean_model=hector (ocean_hector.py instead of the OSCAR ocean) the defaults get a
_hector suffix (rcmip_<rname>_hector/, exe/<rname>_hector/), so the two ensembles do not
overwrite each other (same file names inside).
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_FLAGS = "--ctag=all --ems=CO2 --csec=EMISS_IND --rgn=WORLD"   # common to all members


def _dir(name, default):
    """A directory: the environment value if set, else the default."""
    p = Path(os.environ.get(name) or default)
    return p


def parse_cases(spec, n):
    """--cases: 'all', or comma-separated line indices / ranges (0-based), e.g. 0-9,42."""
    if spec in (None, "all"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), (int(hi) if hi else int(lo)) + 1)
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        sys.exit(f"--cases out of range 0-{n - 1}: {bad[:5]}")
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SCM4OPT ensemble via GNU parallel",
        epilog="Arguments after -- are passed to every scm4opt.py run.")
    ap.add_argument("--rname", default="esm-allGHG-ssp119",
                    help="RCMIP experiment (sets the time range)")
    ap.add_argument("--scen", default="rcmip3", help="scenario driver (default rcmip3)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel jobs (default: CPUs - 1)")
    ap.add_argument("--members", type=Path, default=None,
                    help="member list, one flag set per line (default: the development tree's "
                         "member list, else data/ensemble/rcmip_unc.txt of the package)")
    ap.add_argument("--cases", default="all",
                    help="subset of member lines, 0-based, e.g. 0-49 or 0,5,7 (default all)")
    ap.add_argument("--out", type=Path, default=None, help="results directory (rsl_dir)")
    ap.add_argument("--ocean_model", choices=("osc", "hector"), default=None,
                    help="ocean carbon module (scm4opt.py --ocean_model; default osc); "
                         "hector adds a _hector suffix to the default directories")
    ap.add_argument("--delay", default="0",
                    help="parallel --delay between job starts [s] (default 0; a delay caps the "
                         "start rate at 1/delay members per second)")
    ap.add_argument("--dry-run", action="store_true", help="print the parallel command only")
    ap.add_argument("--quiet", action="store_true",
                    help="discard member output (to /dev/null); default: per-member logs")
    ap.add_argument("--no-standard-options", action="store_true",
                    help="do not add the experiment's standard options (--scen=rcmip3: "
                         "--load_conc=all|nco2 from rcmip3_data.RCMIP3_STANDARD_OPTIONS)")
    ap.add_argument("--list-rcmip3", action="store_true",
                    help="print all RCMIP3 experiments (rcmip3_data.RCMIP3_ALL_SCENARIOS), "
                         "one per line, and exit")
    args, extra = ap.parse_known_args(argv)
    from .scenarios.rcmip3_data import RCMIP3_ALL_SCENARIOS, RCMIP3_STANDARD_OPTIONS
    if args.list_rcmip3:
        print("\n".join(RCMIP3_ALL_SCENARIOS))
        return 0
    extra = [a for a in extra if a != "--"]
    for a in list(extra):                      # --ocean_model given as a model flag
        if a.startswith("--ocean_model="):
            args.ocean_model = args.ocean_model or a.split("=", 1)[1]
            extra.remove(a)
    suffix = "_hector" if args.ocean_model == "hector" else ""
    # the experiment's standard options, unless given explicitly or switched off
    std = [] if args.no_standard_options or args.scen != "rcmip3" else \
        RCMIP3_STANDARD_OPTIONS.get(args.rname, "").split()
    std = [o for o in std if o.split("=")[0] not in {a.split("=")[0] for a in extra}]

    from .core.config import DAT_DIR, PACKAGE_DATA_DIR, PACKAGE_ROOT, REPO_DIR, data_path
    root = PACKAGE_ROOT.name                     # scm4opt here, scm4opt when exported
    dev_members = REPO_DIR / "mpi" / "rcmip_unc.sh"        # development tree
    trunk = REPO_DIR
    mdl_dir = _dir("mdl_dir", trunk.parent if dev_members.exists() else Path.cwd())
    rsl_dir = (args.out or _dir("rsl_dir", mdl_dir / f"rcmip_{args.rname}{suffix}")).resolve()
    tmp_dir = (_dir("tmp_dir", mdl_dir / "exe") / f"{args.rname}{suffix}").resolve()
    members = (args.members or (dev_members if dev_members.exists()
                                else PACKAGE_DATA_DIR / "ensemble" / "rcmip_unc.txt")).resolve()

    # inputs of the experiment: stop early (exit code 3) if they are missing
    if args.scen == "rcmip3":
        dat = Path(next((a.split("=", 1)[1] for a in extra if a.startswith("--dat_dir=")), DAT_DIR))
        need = [f"{args.rname}.csv"] + ([f"{args.rname}_conc.csv"]
                                        if any(o.startswith("--load_conc") for o in std + extra) else [])
        missing = [n for n in need if not data_path(dat, "RCMIP3", n).exists()]
        if missing:
            print(f"SCM4OPT Python ensemble: rname={args.rname}: missing inputs "
                  f"{', '.join('RCMIP3/' + m for m in missing)} in {dat}; skipped")
            return 3

    # the members' Python must have the model's dependencies: stop early (exit code 4)
    chk = subprocess.run([sys.executable, "-c", "import numpy, pandas, scipy"],
                         capture_output=True, text=True, cwd=REPO_DIR)
    if chk.returncode != 0:
        print(f"SCM4OPT Python ensemble: {sys.executable} cannot import the model's "
              f"dependencies (numpy, pandas, scipy):\n{chk.stderr.strip().splitlines()[-1]}")
        return 4

    parallel = shutil.which("parallel")
    if parallel is None:
        sys.exit("GNU parallel not found (brew install parallel / apt install parallel)")
    lines = [ln.strip() for ln in members.read_text().splitlines() if ln.strip()]
    sel = parse_cases(args.cases, len(lines))

    for d in (rsl_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
    # the member lines to run (parallel -a reads one {1} per line)
    member_file = tmp_dir / "members.txt"
    member_file.write_text("".join(lines[i] + "\n" for i in sel))
    log_dir = tmp_dir / "log"
    if not args.quiet:
        log_dir.mkdir(exist_ok=True)

    model = (f"{shlex.quote(sys.executable)} -m {root} --scen={args.scen} "
             f"--rout3={shlex.quote(str(rsl_dir) + '/')} --rname={args.rname} {MODEL_FLAGS}")
    if std:
        model += " " + " ".join(std)
    if args.ocean_model:
        model += f" --ocean_model={args.ocean_model}"
    if extra:
        model += " " + " ".join(shlex.quote(a) for a in extra)
    sink = "/dev/null" if args.quiet else shlex.quote(str(log_dir)) + "/member_{#}.log"
    job = f"{model} {{1}} > {sink} 2>&1"
    cmd = [parallel, "--no-run-if-empty", "--delay", args.delay, "-j", str(args.jobs),
           "--joblog", str(tmp_dir / "joblog.txt")]
    if sys.stdout.isatty():                          # progress bar only in a terminal
        cmd.append("--bar")
    cmd += ["-a", str(member_file), "eval", job]

    from .core.version import __version__
    print(f"SCM4OPT Python {__version__} ensemble: scen={args.scen} rname={args.rname} "
          f"ocean={args.ocean_model or 'osc'} options={' '.join(std + extra) or '-'} "
          f"members={len(sel)}/{len(lines)} jobs={args.jobs}")
    print(f"  members  {members}")
    print(f"  rsl_dir  {rsl_dir}/")
    print(f"  tmp_dir  {tmp_dir}/  (members.txt, joblog.txt{', log/' if not args.quiet else ''})")
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")   # one core per member
    rc = subprocess.call(cmd, cwd=REPO_DIR, env=env)   # REPO_DIR holds the package

    # summary from the job log (column 7 = exit value)
    failed = []
    with open(tmp_dir / "joblog.txt") as f:
        next(f, None)
        for row in f:
            col = row.rstrip("\n").split("\t")
            if len(col) > 8 and col[6] != "0":
                failed.append(int(col[0]))
    print(f"done: {len(sel) - len(failed)} ok, {len(failed)} failed (parallel rc={rc})")
    if failed:
        where = f"{log_dir}/member_<seq>.log" if not args.quiet else "rerun without --quiet"
        print(f"  failed job seq: {failed[:20]}{' ...' if len(failed) > 20 else ''}  ({where})")
        first = log_dir / f"member_{failed[0]}.log"
        if not args.quiet and first.exists():
            tail = first.read_text(errors="replace").strip().splitlines()[-8:]
            print(f"  end of member_{failed[0]}.log:")
            print("\n".join("    | " + ln for ln in tail))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
