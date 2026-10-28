# SPDX-License-Identifier: GPL-3.0-or-later
"""Model errors."""


class ModelBoundsError(FloatingPointError):
    """A model variable left its admissible range.

    variable : name of the variable (e.g. "scm_atmo_pool", "hct_dic(ob_LL)")
    value    : its value, if known
    bound    : the violated bound or range (e.g. "[1, 1e5]", ">= 1e-3")
    Subclass of FloatingPointError, so `except FloatingPointError` still catches it.
    """

    def __init__(self, variable, value=None, bound=None, message=None):
        self.variable, self.value, self.bound = variable, value, bound
        if message is None:
            message = f"{variable}" + (f" = {value:.6g}" if isinstance(value, (int, float)) else "") \
                + (f" outside {bound}" if bound is not None else " outside its admissible range")
        super().__init__(message)
