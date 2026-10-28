# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line.

  scm4opt [run] [--flag=value ...]     coupled forward run
                                       (--name=value flags, e.g. --scen=rcmip3 --rname=ssp585)
  scm4opt ensemble [options] [-- flags] ensemble via GNU parallel (ensemble.py)
  scm4opt export [options]             combine ensemble members into RCMIP3 files
                                       (io/rcmip_export.py; scm4opt export --help)
  scm4opt data verify|pack|fetch|manifest  scenario input data: checksums, archives for
                                       hosting, download (io/datasets.py; scm4opt data --help)
  scm4opt --version                    package version and RCMIP3 model name
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["-V"], ["--version"], ["version"]):
        from .core.version import version_line
        print(version_line())
        return 0
    if argv[:1] in (["-h"], ["--help"], ["help"]):
        print(__doc__)
        return 0
    cmd = argv[0] if argv and not argv[0].startswith("-") else "run"
    if argv and argv[0] == cmd:
        argv = argv[1:]
    if cmd == "run":
        from .scenarios.forward import main as run_main
        run_main(argv)
        return 0
    if cmd == "ensemble":
        from .ensemble import main as ensemble_main
        return ensemble_main(argv)
    if cmd == "export":
        from .io.rcmip_export import main as export_main
        return export_main(argv)
    if cmd == "data":
        from .io.datasets import main as data_main
        return data_main(argv)
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    sys.exit(f"unknown command {cmd!r}; use run, ensemble, export or data (scm4opt help)")
