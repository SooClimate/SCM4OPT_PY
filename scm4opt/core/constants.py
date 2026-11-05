# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical constants shared by the SCM4OPT Python forward model.

Values are those of the SCM4OPT v4.1 parameter definitions; keep them in sync.
"""

# Atmospheric CO2 conversion and pre-industrial level
PPM2GTC = 2.123          # GtC per ppm CO2
CO2_CONC_INITIAL = 284.0  # ppm, CO2 at the first step (1850 start)
# --first_year=1750: the 1750 values of the RCMIP3 v3.0.1 inputs
# (RCMIP3_src/conc_mapping_1750.csv), reference and initial concentrations
CO2_CONC_1750 = 278.00745  # ppm
CH4_CONC_1750 = 720.8548   # ppb
N2O_CONC_1750 = 266.57     # ppb

# Internal time step of the model (1/6 year)
DT = 1.0 / 6.0           # years per step

# Sigmoid CO2 fertilisation; NOT the CSV's CO2_GIFFORD_CONC_FOR_ZERONPP, which the
# land carbon cycle reads but never uses
CO2_NPP_ZERO_CONC = 31.0  # ppm

# Reference CO2 for anomalies in both CH4-forcing branches,
# scm_D_CO2 = scm_atmoCO2 - CO2_0
CO2_0 = 284.0  # ppm
