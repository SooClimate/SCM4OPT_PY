# SPDX-License-Identifier: GPL-3.0-or-later
"""Ocean carbon models as plug-ins, selected with --ocean_model.

Every ocean model provides
  load_params(cfg)              -> parameters
  step(params, prev, drivers)   -> state of step n; prev = state of n-1 or None
  collect(states)               -> {output name: array over t}
and its states expose atmoSeaFlux (GtC per step into the ocean), surface_cpool and
total_cpool (GtC), which couple it to the rest of the model. All take the same
OceanDrivers; each uses the fields it needs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OceanDrivers:
    """Inputs of one ocean step n (fields an ocean model does not use may be None)."""
    co2: float = None           # scm_atmoCO2(n) [ppm]                (Hector)
    d_co2: float = None         # scm_D_CO2(n) [ppm]                  (OSCAR)
    tatm: float = None          # scm_tatm(n) [K anomaly]             (Hector)
    d_sst: float = None         # scm_D_sst(n) [K], DOECLIM at n      (OSCAR)


@dataclass(frozen=True)
class OceanModel:
    name: str                   # --ocean_model value
    prefix: str                 # prefix of its output names
    load_params: object
    step: object
    collect: object


OCEAN_MODELS = ("osc", "hector")


def get_ocean_model(name):
    """The OceanModel registered under --ocean_model=name."""
    if name == "osc":
        from .ocean_oscar import collect_ocean, load_ocean_params, ocean_step
        return OceanModel("osc", "ocn", load_ocean_params, ocean_step, collect_ocean)
    if name == "hector":
        from .ocean_hector import collect_hector, hector_step, load_hector_params
        return OceanModel("hector", "hct", load_hector_params, hector_step, collect_hector)
    raise ValueError(f"--ocean_model={name}: use one of {OCEAN_MODELS}")
