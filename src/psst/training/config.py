from pathlib import Path

import attrs
import attrs.validators as valid
from ruamel.yaml import YAML

from psst.configuration import GenericConfig


@attrs.define(kw_only=True)
class TrainingConfig(GenericConfig):
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
