from __future__ import annotations

import attrs

from psst.configuration import GenericConfig

__all__ = ["EvaluationConfig"]


@attrs.define(kw_only=True)
class EvaluationConfig(GenericConfig):
    """_summary_

    Args:
        GenericConfig (_type_): _description_
    """

    datafile: str
    results_file: str
    reduced_datafile: str
    length: float
    mass: float
