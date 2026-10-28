# SPDX-License-Identifier: GPL-3.0-or-later
"""Version of the Python SCM4OPT and the names it writes into its outputs.

__version__          package version (the exported package's __version__ and pyproject.toml)
RCMIP_CLIMATE_MODEL  climate_model of the RCMIP3 submission rows of this implementation;
                     the original SCM4OPT submits as "SCM4OPTv4.0". The submission's model
                     metadata and comments files must carry the same name (rcmip validate:
                     "climate_model not consistent across the bundle").
RCMIP_FILE_PREFIX    file-name prefix of the RCMIP3 files (member and ensemble files), kept
                     as in the original SCM4OPT so that the export, the runners'
                     --skip-done and the analysis scripts find the files of both
"""

__version__ = "4.1.0.dev0"
RCMIP_CLIMATE_MODEL = "SCM4OPT(PY)v4.0"
RCMIP_FILE_PREFIX = "SCM4OPTv40_rcmip_phase3_"


def version_line():
    """One line for logs: package version and the RCMIP3 model name."""
    return f"SCM4OPT Python {__version__} (RCMIP3 climate_model {RCMIP_CLIMATE_MODEL})"
