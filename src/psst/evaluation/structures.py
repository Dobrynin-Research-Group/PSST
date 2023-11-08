from __future__ import annotations
from typing import NamedTuple

import numpy as np

__all__ = ["RepeatUnit", "PeResult", "InferenceResult"]


class RepeatUnit(NamedTuple):
    """Details of the repeat unit. Mass in units of g/mol, projection length (along
    fully extended axis) in nm (:math:`10^{-9}` m).
    """

    length: float
    mass: float


class PeResult(NamedTuple):
    """The optimized value of :math:`P_e` and the variance of that value from the
    fitting function.
    """

    opt: float
    var: float


class InferenceResult(NamedTuple):
    bg: float
    bth: float
    pe_combo: PeResult
    pe_bg_only: PeResult
    pe_bth_only: PeResult
    reduced_conc: np.ndarray
    degree_polym: np.ndarray
    specific_visc: np.ndarray
