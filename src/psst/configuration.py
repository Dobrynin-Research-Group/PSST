"""Configuration classes for the training cycle, Adam optimizer, and SampleGenerator.

Defines three configuration classes: :class:`RunConfig`, :class:`AdamConfig`, and
:class:`GeneratorConfig`, which are repectively used to configure the machine learning
run settings, the Adam optimizer settings, and the settings for the SurfaceGenerator
class. Each configuration class has its own `load_*_config` helper function.
"""
from __future__ import annotations
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
    "validate_filepath",
    "get_dict_from_file",
    "write_dict_to_file",
    "ModelConfig",
    "RangeConfig",
    "AdamConfig",
    "GeneratorConfig",
]

is_two_positive_floats = [
    valid.max_len(2),
    valid.min_len(2),
    valid.deep_iterable([valid.gt(0.0), valid.instance_of(float)]),
]


def validate_filepath(filepath: str | Path, *, exists: bool | None = False) -> Path:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.parent.is_dir():
        FileNotFoundError(f"Could not locate parent directory: {filepath.parent}")

    if exists is not None:
        if exists and not filepath.exists():
            raise FileNotFoundError(f"Could not locate file: {filepath}")
        elif not exists and filepath.exists():
            raise FileExistsError(f"File already exists: {filepath}")

    return filepath


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

    filepath = validate_filepath(filepath, exists=True)

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


def write_dict_to_file(
    d: dict[str, Any], filepath: str | Path, overwrite: bool = False
):
    filepath = validate_filepath(filepath, exists=(None if overwrite else False))

    ext = filepath.suffix
    if ext == ".json":
        dump = json.dump
    elif ext in [".yaml", ".yml"]:
        yaml = YAML(typ="safe", pure=True)
        dump = yaml.dump
    else:
        raise ValueError(
            f"Invalid file extension for config file: {ext}\n"
            "Please use .yaml or .json."
        )

    with open(filepath, "w") as out:
        dump(d, out)


class GenericConfig:
    def keys(self):
        return attrs.asdict(self).keys()

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(**d)

    @classmethod
    def from_file(cls, filepath: str | Path):
        return cls.from_dict(get_dict_from_file(filepath))

    def to_yaml(self, filepath: str | Path, overwrite: bool = False):
        exists = None if overwrite else False
        validate_filepath(filepath, exists=exists)

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, **self)


@attrs.define(kw_only=True)
class ModelConfig(GenericConfig):
    bg_name: Optional[str] = None
    bth_name: Optional[str] = None
    bg_path: Optional[Path] = None
    bth_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        if not isinstance(d, dict):
            raise TypeError("Must supply dict to from_dict class method")

        if "bg" not in d:
            bg_name = None
            bg_path = None
        elif isinstance(d["bg"], str):
            bg_name = d["bg"]
            bg_path = None
        elif isinstance(d["bg"], dict):
            bg_name = d["bg"]["name"]
            bg_path = d["bg"].get("path", None)
        else:
            raise TypeError("Key 'bg' must have value of type dict or string")

        if "bth" not in d:
            bth_name = None
            bth_path = None
        elif isinstance(d["bth"], str):
            bth_name = d["bth"]
            bth_path = None
        elif isinstance(d["bth"], dict):
            bth_name = d["bth"]["name"]
            bth_path = d["bth"].get("path", None)
        else:
            raise TypeError("Key 'bth' must have value of type dict or string")

        return cls(
            bg_name=bg_name, bg_path=bg_path, bth_name=bth_name, bth_path=bth_path
        )


@attrs.define(kw_only=True)
class RangeConfig(GenericConfig):
    """_summary_

    Args:
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

    phi_range: Range = attrs.field(converter=convert_to_range)
    nw_range: Range = attrs.field(converter=convert_to_range)
    visc_range: Range = attrs.field(converter=convert_to_range)
    bg_range: Range = attrs.field(converter=convert_to_range)
    bth_range: Range = attrs.field(converter=convert_to_range)
    pe_range: Range = attrs.field(converter=convert_to_range)


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
        exists = None if overwrite else False
        validate_filepath(filepath, exists=exists)

        yaml = YAML(typ="safe", pure=True)
        yaml.dump(filepath, **self)


@attrs.define(kw_only=True)
class GeneratorConfig(GenericConfig):
    """Configuration settings for the :class:`SampleGenerator` class.

    Args:
        batch_size (int): Number of samples generated per batch.
        noise_factor (float): Adjusts amplitude of Gaussian noise applied to generated
          samples according to the equation
          ``samples *= 1 + noise_factor * torch.normal(mean=0.0, std=1.0)`` (default is
          0.05). If set to 0.0, the noise will not be added and the normal distribution
          will not be generated at all.
        num_nw_choices (int): The number of :math:`N_w` rows to choose from during the
          trimming stage of the SampleGenerator (default is 48). This many row indices
          with the most number of non-zero values in each sample are available for
          random selection to be kept.
        num_nw_to_select (int): The number of :math:`N_w` rows to choose from the
          selection described by ``num_nw_choices`` (default is 12). This is a maximum
          value, as the row indices are chosen with replacement.
        num_phi_to_select (int): The maximum number of :math:`cl^3` rows to keep.
          (default is 65). This is a maximum value, as the row indices are chosen with
          replacement.

    Returns:
        _type_: _description_
    """

    batch_size: int = attrs.field(converter=int, validator=valid.gt(0))
    noise_factor: float = attrs.field(default=0.05, converter=float)

    num_nw_choices: int = attrs.field(default=48, converter=int, validator=valid.gt(0))
    num_nw_to_select: int = attrs.field(
        default=12, converter=int, validator=valid.gt(0)
    )
    num_phi_to_select: int = attrs.field(
        default=65, converter=int, validator=valid.gt(0)
    )
