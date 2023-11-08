from __future__ import annotations
from pathlib import Path

import attrs
import attrs.validators as valid
from ruamel.yaml import YAML

from psst.configuration import GenericConfig, is_two_positive_floats


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
