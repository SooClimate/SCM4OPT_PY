# SPDX-License-Identifier: GPL-3.0-or-later
"""Forcing tables: halogen radiative efficiencies and land-cover albedo parameters.

Loaded from tables/*.csv in the data folder (config.data_path); edit the CSV
files, not this module, and re-run the regression tests after any edit:

  halo_radeff.csv          gas, radeff_mW_m2_ppt   radeff_HALO(halo) before *1E-3 [mW/m2/ppt]
  lcc_albedo_global.csv    flux, albedo, cover, <land types>
                           alpha_LCC_param(ALBLCflux, ALBLCalb, ALBLCcover, land_type) [W/m2/Mha]
  lcc_albedo_regional.csv  flux, albedo, cover, region, <land types>
                           alpha_LCC_reg_param(..., region_11, land_type) [W/m2/Mha]
Land types in sets.LAND_TYPE order, regions in sets.REGION_11 order. Values are raw table
entries (forcing.py applies the transformations).

The per-experiment forcing options (natural-forcing group, log-form CO2 forcing) are in
rcmip3_data.py (NAT_GRP_BY_RNAME, LOG_CO2_RNAMES), maintained by hand.
"""

import csv

from ..core.config import DAT_DIR, data_path
from ..scenarios.rcmip3_data import LOG_CO2_RNAMES, NAT_GRP_BY_RNAME  # noqa: F401  (former location)
from ..core.sets import LAND_TYPE, REGION_11


def _read(name):
    with open(data_path(DAT_DIR, "tables", name), newline="") as fh:
        return list(csv.DictReader(fh))


RADEFF_HALO = {r["gas"]: float(r["radeff_mW_m2_ppt"]) for r in _read("halo_radeff.csv")}

ALPHA_LCC_PARAM = {(r["flux"], r["albedo"], r["cover"]): tuple(float(r[c]) for c in LAND_TYPE)
                   for r in _read("lcc_albedo_global.csv")}


def _regional():
    rows = {}
    for r in _read("lcc_albedo_regional.csv"):
        rows.setdefault((r["flux"], r["albedo"], r["cover"]), {})[r["region"]] = \
            tuple(float(r[c]) for c in LAND_TYPE)
    for key, by_region in rows.items():
        if set(by_region) != set(REGION_11):
            raise ValueError(f"lcc_albedo_regional.csv {key}: regions {sorted(by_region)}")
    return {key: tuple(by_region[g] for g in REGION_11) for key, by_region in rows.items()}


ALPHA_LCC_REG_PARAM = _regional()
