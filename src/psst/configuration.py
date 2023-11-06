"""Configuration classes for the training cycle, Adam optimizer, and SampleGenerator.

Defines three configuration classes: :class:`RunConfig`, :class:`AdamConfig`, and
:class:`GeneratorConfig`, which are repectively used to configure the machine learning
run settings, the Adam optimizer settings, and the settings for the SurfaceGenerator
class. Each configuration class has its own `load_*_config` helper function.
"""
from __future__ import annotations
from functools import singledispatch
import json
import logging
from pathlib import Path
from typing import Any, Optional

import attrs
import attrs.validators as valid
import attrs.converters as conv
from ruamel.yaml import YAML

from psst import Range, convert_to_range

__all__ = [
    "RunConfig",
    "AdamConfig",
    "OptimConfig",
    "GeneratorConfig",
    "TrimConfig",
    "convert_to_trim_config",
]


is_two_positive_floats = [
    valid.max_len(2),
    valid.min_len(2),
    valid.deep_iterable([valid.gt(0.0), valid.instance_of(float)]),
]


def get_dict_from_file(filepath: str | Path) -> dict[str, Any]:
    """Reads a YAML or JSON file and returns the contents as a dictionary.

    Args:
        filepath (str | Path): The YAML or JSON file to interpret.

    Raises:
        ValueError: If the extension in the filename is not one of ".yaml", ".yml", or
          ".json".

    Returns:
        dict[str]: The contents of the file in dictionary form.
    """
    log = logging.getLogger("psst.main")

    if isinstance(filepath, str):
        filepath = Path(filepath)

    ext = filepath.suffix
    if ext == ".json":
        load = json.load
    elif ext in [".yaml", ".yml"]:
        yaml = YAML(typ="safe", pure=True)
        load = yaml.load
    else:
        raise ValueError(
            f"Invalid file extension for config file: {ext}\n"
            "Please use .yaml or .json."
        )

    log.info("Loading configuration from %s", str(filepath))

    with open(filepath, "r") as f:
        config_dict = dict(load(f))
        log.debug("Loaded configuration: %s", str(config_dict))

    return config_dict


class GenericConfig:
    def keys(self):
        return attrs.asdict(self).keys()

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def from_file(cls, filepath: str | Path):
        return cls(**get_dict_from_file(filepath))

    def _validate_filepath(self, filepath: str | Path, overwrite: bool = False) -> Path:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if filepath.is_file() and not overwrite:
            raise FileExistsError(
                "File already exists; please pass `overwrite=True` to overwrite:"
                f" {filepath}"
            )

        if not filepath.parent.is_dir():
            raise FileNotFoundError(
                f"Could not locate parent directory: {filepath.parent}"
            )

        return filepath


@attrs.define(kw_only=True)
class RunConfig(GenericConfig):
    """Configuration settings for the training/testing cycle.

    Args:
        num_epochs (int): Number of epochs to run
        num_samples_train (int): Number of samples to run through model training per epoch
        num_samples_test (int): Number of samples to run through model testing/validation
          per epoch
        checkpoint_frequency (int): Frequency with which to save the model and optimizer.
          Positive values are number of epochs. Negative values indicate to save when
          the value of the loss function hits a new minimum. Defaults to 0, never saving.
        checkpoint_filename (str): Name of checkpoint file into which to save the model
          and optimizer. Defaults to ``"chk.pt"``.
    """

    num_epochs: int = attrs.field(converter=int, validator=valid.ge(0))
    num_samples_train: int = attrs.field(converter=int, validator=valid.ge(0))
    num_samples_test: int = attrs.field(converter=int, validator=valid.ge(0))
    checkpoint_frequency: int = attrs.field(default=0, converter=int)
    checkpoint_filename: str = attrs.field(default="chk.pt", converter=str)

    def to_yaml(self, filepath: str | Path, overwrite: bool = False):
        super()._validate_filepath(filepath, overwrite)

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, **self)


@attrs.define(kw_only=True)
class AdamConfig(GenericConfig):
    """Configuration settings for the Adam optimizer. See PyTorch's
    `Adam optimizer <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_
    documentation for details.
    """

    lr: float = attrs.field(
        default=0.001, converter=float, validator=[valid.gt(0.0), valid.lt(1.0)]
    )
    betas: tuple[float, float] = attrs.field(
        default=(0.9, 0.999),
        validator=is_two_positive_floats,
    )
    eps: float = attrs.field(
        default=1e-8, converter=float, validator=[valid.gt(0), valid.lt(1)]
    )
    weight_decay: float = attrs.field(
        default=0.0, converter=float, validator=[valid.ge(0), valid.lt(1)]
    )
    amsgrad: bool = attrs.field(default=False, converter=conv.to_bool)
    foreach: Optional[bool] = attrs.field(
        default=None, converter=conv.optional(conv.to_bool)
    )
    maximize: bool = attrs.field(default=False, converter=conv.to_bool)
    capturable: bool = attrs.field(default=False, converter=conv.to_bool)
    differentiable: bool = attrs.field(default=False, converter=conv.to_bool)
    fused: Optional[bool] = attrs.field(
        default=None, converter=conv.optional(conv.to_bool)
    )

    def to_yaml(self, filepath: str | Path, overwrite: bool = False):
        super()._validate_filepath(filepath, overwrite)

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, **self)


@attrs.define(kw_only=True)
class OptimConfig(GenericConfig):
    num_trials: int = attrs.field(validator=[valid.instance_of(int), valid.gt(0)])
    num_epochs: int = attrs.field(validator=[valid.instance_of(int), valid.gt(0)])
    num_samples_per_epoch: int = attrs.field(
        validator=[valid.instance_of(int), valid.gt(0)]
    )
    lr: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    beta_1: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    beta_2: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    eps: tuple[float, float] = attrs.field(validator=is_two_positive_floats)

    def to_yaml(self, filepath: str | Path, overwrite: bool = False):
        super()._validate_filepath(filepath, overwrite)

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, **self)


@attrs.define(kw_only=True)
class TrimConfig:
    num_nw_choices: int = 48
    num_nw_to_select: int = 12
    num_phi_to_select: int = 65


@singledispatch
def convert_to_trim_config(d) -> TrimConfig:
    raise NotImplementedError(f"Cannot convert object of type {type(d)} to TrimConfig")


@convert_to_trim_config.register(dict)
def _(d):
    return TrimConfig(**d)


@convert_to_trim_config.register(list)
@convert_to_trim_config.register(tuple)
def _(d):
    return TrimConfig(*d)


@convert_to_trim_config.register
def _(d: TrimConfig):
    return d


@attrs.define(kw_only=True, eq=True)
class GeneratorConfig(GenericConfig):
    """Configuration settings for the :class:`SampleGenerator` class.

    Args:
        parameter (str): Either 'Bg' or 'Bth' to generate
          viscosity samples for the good solvent parameter or the thermal blob
          parameter, respectively.
        batch_size (int): Number of samples generated per batch.
        phi_range (Range): The range of values for the normalized
          concentration :math:`cl^3`.
        nw_range (Range): The range of values for the weight-average
          degree of polymerization of the polymers.
        visc_range (Range): The range of values for the specific
          viscosity. This is only used for normalization, so `num=0` is fine.
        bg_range (Range): The range of values for the good solvent
          parameter. This is only used for normalization, so `num=0` is fine.
        bth_range (Range): The range of values for the thermal blob
          parameter. This is only used for normalization, so `num=0` is fine.
        pe_range (Range): The range of values for the entanglement
          packing number. This is only used for normalization, so `num=0` is fine.
    """

    batch_size: int = attrs.field(converter=int, validator=valid.gt(0))

    phi_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )
    nw_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )
    visc_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )

    bg_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )
    bth_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )
    pe_range: Range = attrs.field(
        converter=convert_to_range, validator=valid.instance_of(Range)
    )

    noise_factor: float = attrs.field(default=0.05, converter=float)
    trim: TrimConfig = attrs.field(factory=TrimConfig, converter=convert_to_trim_config)

    def to_yaml(self, filepath: str | Path, overwrite: bool = False):
        super()._validate_filepath(filepath, overwrite)

        d = dict(self)
        d["phi_range"] = attrs.asdict(d["phi_range"])
        d["nw_range"] = attrs.asdict(d["nw_range"])
        d["visc_range"] = attrs.asdict(d["visc_range"])
        d["bg_range"] = attrs.asdict(d["bg_range"])
        d["bth_range"] = attrs.asdict(d["bth_range"])
        d["pe_range"] = attrs.asdict(d["pe_range"])
        d["trim"] = attrs.asdict(d["trim"])

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, d)
