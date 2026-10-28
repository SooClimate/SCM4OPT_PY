# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-gas parameters of the 48 halogenated gases (lifetimes, molar masses, GWP100).

Loaded from tables/*.csv in the data folder (config.data_path); edit the CSV
files, not this module, and re-run the regression tests after any edit. Values are raw
table entries (chemistry.py applies the transformations: molar mass x 0.1765, tau_hv x
1.06, inf/0 defaults).

halo_parameters.csv -> HALO_TABLE[gas] = tuple in COLUMNS order:
  molar_mass  alpha_HALO before *0.1765
  tau_total   tau_HALO_total_lifetime [yr] (not used by the equations)
  tau_OH      tau_HALO_OH [yr]
  tau_hv_raw  tau_HALO_hv before *1.06 [yr]
  tau_othr    tau_HALO_othr_nzero [yr]; empty (None) -> inf
  n_Cl, n_Br  Cl / Br atoms (ODS only); empty (None) -> 0
  gwp100      def_gwp100_oa2022_ems; empty (None) -> 0
  ("inf" in the file is an infinite lifetime)
halo_fracrel.csv -> FRACREL[option][gas]: f_fracrel(halo_ods) for each --O3Sfracrel_mdl
  option (one column per option); gases absent -> 0
"""

import csv

from ..core.config import DAT_DIR, data_path

INF = float("inf")

COLUMNS = ("molar_mass", "tau_total", "tau_OH", "tau_hv_raw", "tau_othr", "n_cl", "n_br", "gwp100")


def _read(name):
    with open(data_path(DAT_DIR, "tables", name), newline="") as fh:
        return list(csv.DictReader(fh))


def _num(s):
    return None if s == "" else float(s)


HALO_TABLE = {r["gas"]: tuple(_num(r[c]) for c in COLUMNS) for r in _read("halo_parameters.csv")}

_FR = _read("halo_fracrel.csv")
FRACREL = {opt: {r["gas"]: float(r[opt]) for r in _FR if r[opt] != ""}
           for opt in _FR[0] if opt != "gas"}
