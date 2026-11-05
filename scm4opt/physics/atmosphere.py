# SPDX-License-Identifier: GPL-3.0-or-later
"""Atmospheric carbon pool of SCM4OPT (equation scm_ATMOEQ).

Default form of scm_ATMOEQ:

    scm_atmo_pool(t) =E= ( scm_atmo_pool(t-1)
                         + dat_eind(t,"co2")
                         + scm_emeth(t-1)
                         - scm_atmoSeaFlux(t-1)
                         - scm_atmoLndFlux(t-1) )$(not tfirst(t))
                       + (co2_conc_initial * PPM2GTC)$tfirst(t);

All pools are in GtC and all fluxes in GtC per model time step (the model
runs at 1/6-year steps; the series are used as-is, no rescaling).
Land-use CO2 emissions (dat_elnd) are not added here: they enter through the land
module (deforestation fluxes in lnd_atmoLndFlux, land.py).
"""

from dataclasses import dataclass

import numpy as np

from ..core.config import RunConfig
from ..core.constants import CO2_CONC_1750, CO2_CONC_INITIAL, PPM2GTC


@dataclass(frozen=True)
class AtmosphereParams:
    """Atmosphere parameters."""
    pool_initial: float     # scm_atmo_pool at the first step [GtC]


def load_atmosphere_params(cfg=None):
    """Atmosphere parameters of a run configuration: the pool starts at co2_conc_initial,
    284 ppm (1850 start) or 278.00745 ppm (--first_year=1750); --co2_conc_initial overrides it."""
    cfg = cfg or RunConfig()
    conc = cfg.co2_conc_initial
    if conc is None:
        conc = CO2_CONC_1750 if cfg.first_year == 1750 else CO2_CONC_INITIAL
    return AtmosphereParams(pool_initial=conc * PPM2GTC)


@dataclass(frozen=True)
class AtmosphereDrivers:
    """Inputs of one atmosphere step t (all in GtC per step)."""
    eind_co2: float         # dat_eind(t,"co2")       (note: time t, not t-1)
    emeth_prev: float       # scm_emeth(t-1)          CO2 from CH4 oxidation
    sea_flux_prev: float    # scm_atmoSeaFlux(t-1)    atmosphere -> ocean
    lnd_flux_prev: float    # scm_atmoLndFlux(t-1)    atmosphere -> land


def atmosphere_step(p, prev, drivers):
    """scm_ATMOEQ for step t: the atmospheric carbon pool [GtC].

    prev    : scm_atmo_pool(t-1), or None at the first step (-> p.pool_initial)
    drivers : AtmosphereDrivers (ignored at the first step)
    """
    if prev is None:
        return p.pool_initial
    d = drivers
    return prev + d.eind_co2 + d.emeth_prev - d.sea_flux_prev - d.lnd_flux_prev


def atmo_pool_series(eind_co2, emeth, sea_flux, lnd_flux,
                     co2_conc_initial=CO2_CONC_INITIAL, ppm2gtc=PPM2GTC):
    """scm_ATMOEQ over the whole time axis t = 0 (tfirst) .. n-1.

    Inputs are 1-D arrays of equal length indexed by t. Returns scm_atmo_pool(t).
    In the coupled model the ocean/land fluxes depend on the pool itself, so a
    full model run has to call atmosphere_step inside the main time loop; this
    helper is for the case where the flux series are already known (e.g. from
    saved model output).
    """
    eind_co2, emeth, sea_flux, lnd_flux = (np.asarray(a, dtype=float)
                                           for a in (eind_co2, emeth, sea_flux, lnd_flux))
    n = eind_co2.size
    pool = np.empty(n)
    p = AtmosphereParams(pool_initial=co2_conc_initial * ppm2gtc)
    pool[0] = atmosphere_step(p, None, None)
    for t in range(1, n):
        pool[t] = atmosphere_step(p, pool[t - 1], AtmosphereDrivers(
            eind_co2[t], emeth[t - 1], sea_flux[t - 1], lnd_flux[t - 1]))
    return pool


def compatible_emissions(atmo_pool, emeth, sea_flux, lnd_flux):
    """Compatible fossil and industrial CO2 emissions: scm_ATMOEQ solved for
    dat_eind(t,"co2") given the pool (atmo_pool_series run backwards).

        eind(t) = pool(t) - pool(t-1) - emeth(t-1) + sea_flux(t-1) + lnd_flux(t-1)

    With CO2 prescribed (--load_conc=all) the land and ocean fluxes are driven by the
    prescribed concentration, so this is exact: an emission-driven run with these
    emissions (and --co2_conc_initial set to the prescribed first-year value) reproduces the
    concentrations. Land-use emissions are not included (they enter through lnd_flux).
    The first step has no balance and is 0. All in GtC per step.
    """
    atmo_pool, emeth, sea_flux, lnd_flux = (np.asarray(a, dtype=float)
                                            for a in (atmo_pool, emeth, sea_flux, lnd_flux))
    eind = np.zeros_like(atmo_pool)
    eind[1:] = np.diff(atmo_pool) - emeth[:-1] + sea_flux[:-1] + lnd_flux[:-1]
    return eind


if __name__ == "__main__":
    n = 5
    pool = atmo_pool_series(eind_co2=np.full(n, 0.5), emeth=np.full(n, 0.01),
                            sea_flux=np.full(n, 0.2), lnd_flux=np.full(n, 0.1))
    print(pool)  # first value 284*2.123 = 602.932, then +0.21 per step
