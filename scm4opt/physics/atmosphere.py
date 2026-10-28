# SPDX-License-Identifier: GPL-3.0-or-later
"""Atmospheric carbon pool of SCM4OPT (equation scm_ATMOEQ).

Default form of scm_ATMOEQ:

    scm_atmo_pool(t) =E= ( scm_atmo_pool(t-1)
                         + dat_eind(t,"co2")
                         + scm_emeth(t-1)
                         - scm_atmoSeaFlux(t-1)
                         - scm_atmoLndFlux(t-1) )$(not tfirst(t))
                       + (co2_conc_1850 * PPM2GTC)$tfirst(t);

All pools are in GtC and all fluxes in GtC per model time step (the model
runs at 1/6-year steps; the series are used as-is, no rescaling).
Land-use CO2 emissions (dat_elnd) are not added here: they enter through the land
module (deforestation fluxes in lnd_atmoLndFlux, land.py).
"""

from dataclasses import dataclass

import numpy as np

from ..core.config import RunConfig
from ..core.constants import CO2_CONC_1850, PPM2GTC


@dataclass(frozen=True)
class AtmosphereParams:
    """Atmosphere parameters."""
    pool_1850: float        # scm_atmo_pool at the first step [GtC]


def load_atmosphere_params(cfg=None):
    """Atmosphere parameters of a run configuration."""
    return AtmosphereParams(pool_1850=CO2_CONC_1850 * PPM2GTC)


@dataclass(frozen=True)
class AtmosphereDrivers:
    """Inputs of one atmosphere step t (all in GtC per step)."""
    eind_co2: float         # dat_eind(t,"co2")       (note: time t, not t-1)
    emeth_prev: float       # scm_emeth(t-1)          CO2 from CH4 oxidation
    sea_flux_prev: float    # scm_atmoSeaFlux(t-1)    atmosphere -> ocean
    lnd_flux_prev: float    # scm_atmoLndFlux(t-1)    atmosphere -> land


def atmosphere_step(p, prev, drivers):
    """scm_ATMOEQ for step t: the atmospheric carbon pool [GtC].

    prev    : scm_atmo_pool(t-1), or None at the first step (-> p.pool_1850)
    drivers : AtmosphereDrivers (ignored at the first step)
    """
    if prev is None:
        return p.pool_1850
    d = drivers
    return prev + d.eind_co2 + d.emeth_prev - d.sea_flux_prev - d.lnd_flux_prev


def atmo_pool_series(eind_co2, emeth, sea_flux, lnd_flux,
                     co2_conc_1850=CO2_CONC_1850, ppm2gtc=PPM2GTC):
    """scm_ATMOEQ over the whole time axis t = 0 (tfirst = 1850) .. n-1.

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
    p = AtmosphereParams(pool_1850=co2_conc_1850 * ppm2gtc)
    pool[0] = atmosphere_step(p, None, None)
    for t in range(1, n):
        pool[t] = atmosphere_step(p, pool[t - 1], AtmosphereDrivers(
            eind_co2[t], emeth[t - 1], sea_flux[t - 1], lnd_flux[t - 1]))
    return pool


if __name__ == "__main__":
    n = 5
    pool = atmo_pool_series(eind_co2=np.full(n, 0.5), emeth=np.full(n, 0.01),
                            sea_flux=np.full(n, 0.2), lnd_flux=np.full(n, 0.1))
    print(pool)  # first value 284*2.123 = 602.932, then +0.21 per step
