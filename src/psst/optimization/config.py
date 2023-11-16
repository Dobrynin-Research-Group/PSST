from __future__ import annotations

import attrs
import attrs.validators as valid

import psst
from psst.configuration import GenericConfig, is_two_positive_floats


@attrs.define(kw_only=True)
class OptimConfig(GenericConfig):
    output_file: str = ""
    num_trials: int = attrs.field(validator=[valid.instance_of(int), valid.gt(0)])
    num_epochs: int = attrs.field(validator=[valid.instance_of(int), valid.gt(0)])
    num_samples_train: int = attrs.field(
        validator=[valid.instance_of(int), valid.gt(0)]
    )
    lr: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    beta_1: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    beta_2: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
    eps: tuple[float, float] = attrs.field(validator=is_two_positive_floats)
