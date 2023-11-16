from __future__ import annotations
from pathlib import Path
from typing import NamedTuple

import numpy as np

from psst.configuration import *

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

    def arrays_to_csv(self, filepath: str | Path, overwrite: bool = False):
        filepath = validate_filepath(filepath, exists=(None if overwrite else False))
        np.savetxt(
            filepath,
            np.stack(
                [self.reduced_conc, self.degree_polym, self.specific_visc], axis=1
            ),
        )

    def write_to_file(self, filepath: str | Path, overwrite: bool = False):
        d = self._asdict()
        d.pop("reduced_conc")
        d.pop("degree_polym")
        d.pop("specific_visc")
        write_dict_to_file(d, filepath, overwrite)
