# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical constants shared by the SCM4OPT Python forward model.

Values are those of the SCM4OPT v4.1 parameter definitions; keep them in sync.
"""

# Atmospheric CO2 conversion and pre-industrial level
PPM2GTC = 2.123          # GtC per ppm CO2
CO2_CONC_1850 = 284.0    # ppm

# Internal time step of the model (1/6 year)
DT = 1.0 / 6.0           # years per step

# Sigmoid CO2 fertilisation; NOT the CSV's CO2_GIFFORD_CONC_FOR_ZERONPP, which the
# land carbon cycle reads but never uses
CO2_NPP_ZERO_CONC = 31.0  # ppm

# Reference CO2 for anomalies in both CH4-forcing branches,
# scm_D_CO2 = scm_atmoCO2 - CO2_0
CO2_0 = 284.0  # ppm
