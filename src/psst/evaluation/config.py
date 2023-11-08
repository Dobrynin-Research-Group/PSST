from __future__ import annotations
from pathlib import Path
from typing import NamedTuple

import attrs
import numpy as np
import numpy.typing as npt
import torch
import psst

from psst.configuration import GenericConfig
from psst.models import Inception3, Vgg13
from .structures import RepeatUnit

__all__ = ["EvaluationConfig", "ModelSpec"]


class ModelSpec(NamedTuple):
    model_name: str
    filename: str


def get_data_from_file(filename: str):
    data = np.loadtxt(filename)
    assert data.shape[1] == 3
    return data


@attrs.define(kw_only=True)
class EvaluationConfig(GenericConfig):
    """_summary_

    Args:
        GenericConfig (_type_): _description_
    """

    bg_model: torch.nn.Module
    bth_model: torch.nn.Module
    repeat_unit: RepeatUnit
    concentration: npt.NDArray
    molecular_weight: npt.NDArray
    specific_viscosity: npt.NDArray

    @classmethod
    def from_file(cls, filepath: str | Path):
        filepath = cls._validate_filepath(filepath)
        d = cls._get_dict_from_file(filepath)

        assert Path(d["datafile"]).is_file()
        repeat_unit = RepeatUnit(length=d["length"], mass=d["mass"])

        if d["bg_model"]["model_name"] == "Inception3":
            bg_model = Inception3()
        elif d["bg_model"]["model_name"] == "Vgg13":
            bg_model = Vgg13()
        else:
            raise ValueError("Only Inception3 and Vgg13 models are supported")
        chkpt: psst.Checkpoint = torch.load(d["bg_model"]["filename"])
        bg_model.load_state_dict(chkpt.model_state)

        if d["bth_model"]["model_name"] == "Inception3":
            bth_model = Inception3()
        elif d["bth_model"]["model_name"] == "Vgg13":
            bth_model = Vgg13()
        else:
            raise ValueError("Only Inception3 and Vgg13 models are supported")
        chkpt: psst.Checkpoint = torch.load(d["bth_model"]["filename"])
        bth_model.load_state_dict(chkpt.model_state)

        conc, mw, visc = np.loadtxt(d["datafile"], unpack=True, delimiter=",")

        return cls(
            bg_model=bg_model,
            bth_model=bth_model,
            repeat_unit=repeat_unit,
            concentration=conc,
            molecular_weight=mw,
            specific_viscosity=visc,
        )
