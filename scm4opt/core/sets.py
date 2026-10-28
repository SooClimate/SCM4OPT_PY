# SPDX-License-Identifier: GPL-3.0-or-later
"""Sets used by the forward model, in their declaration order.

Order matters: arrays indexed by these tuples line up with the model sets
(gas, halo, region_11, land_type).
"""

# Non-halogenated gases (gas_nonf)
GAS_NONF = ("co2", "ch4", "n2o", "so2", "co", "voc", "nox", "bc", "oc", "nh3")

# Fluorinated halogens (fhalo), 26 gases
FHALO = (
    "hfc23", "hfc32", "hfc125", "hfc134a", "hfc143a", "hfc152a", "hfc227ea",
    "hfc236fa", "hfc245fa", "hfc365mfc", "hfc4310mee", "hfc134", "hfc143", "hfc41",
    "sf6", "nf3", "so2f2", "cf4", "c2f6", "c3f8", "cc4f8", "c4f10", "c5f12",
    "c6f14", "c7f16", "c8f18",
)

# Montreal-protocol halogens (mhalo), 22 gases
MHALO = (
    "cfc11", "cfc12", "cfc113", "cfc114", "cfc115", "cfc13", "ccl4", "ch3ccl3",
    "hcfc22", "hcfc124", "hcfc132b", "hcfc133a", "hcfc141b", "hcfc142b",
    "halon1211", "halon1202", "halon1301", "halon2402", "ch3br", "ch3cl",
    "ch2cl2", "chcl3",
)

HALO = FHALO + MHALO     # halo, 48 gases
GAS = GAS_NONF + HALO    # gas, 58 gases

# Land-cover regions and types
REGION_11 = ("LAM", "OAS", "AFR", "ROW", "MEA", "EUR", "CHN", "IND", "JPN", "USA", "RUS")
LAND_TYPE = ("land_des", "land_for", "land_gra", "land_cro", "land_pas")

# Halogen subsets
HALO_HFC = ("hfc23", "hfc32", "hfc125", "hfc134a", "hfc143a", "hfc152a", "hfc227ea",
            "hfc236fa", "hfc245fa", "hfc365mfc", "hfc4310mee", "hfc134", "hfc143", "hfc41")
HALO_PFC = ("cf4", "c2f6", "c3f8", "cc4f8", "c4f10", "c5f12", "c6f14", "c7f16", "c8f18")
HALO_ODS = ("cfc11", "cfc12", "cfc113", "cfc114", "cfc115", "cfc13", "ccl4", "ch3ccl3",
            "hcfc22", "hcfc124", "hcfc132b", "hcfc133a", "hcfc141b", "hcfc142b",
            "halon1211", "halon1202", "halon1301", "halon2402", "ch3br", "ch3cl")
HALO_CFC = ("cfc11", "cfc12", "cfc113", "cfc114", "cfc115", "cfc13")

# Halogens prescribed from RCMIP concentrations with --load_conc;
# the rest (halo_non_rcmip_agage) stay emission-driven
HALO_NON_RCMIP_AGAGE = ("hfc134", "hfc143", "hfc41", "halon1202")
HALO_RCMIP_AGAGE = tuple(g for g in HALO if g not in HALO_NON_RCMIP_AGAGE)
