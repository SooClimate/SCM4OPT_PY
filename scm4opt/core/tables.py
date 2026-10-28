# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for the option tables of the physics modules."""

import numpy as np

from .sets import HALO


def pick(table, key, flag):
    """table[key] for the option value `key` of the command-line flag `flag`; a
    ValueError listing the valid options if the key is unknown."""
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown {flag}={key!r}; options: {list(table)}") from None


def halo_mask(subset):
    """Boolean mask over the 48 halogenated gases (sets.HALO) of the gases in `subset`."""
    return np.array([g in subset for g in HALO])
