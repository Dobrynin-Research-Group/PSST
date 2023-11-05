from enum import Enum
from pathlib import Path
from typing import NamedTuple

from attrs import asdict
import torch


__all__ = ["ModelType", "Checkpoint", "FinalState"]


class ModelType(str, Enum):
    Inception3 = "Inception3"
    Vgg13 = "Vgg13"


class Checkpoint(NamedTuple):
    """Represents a state during training. Can be easily saved to file `filepath` with

    ```python
    >>> chkpt = psst.Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
    >>> torch.save(chkpt, filepath)
    ```

    Args:
        epoch (int): How many cycles of training have been completed.
        model_state (dict): The state of the neural network model as given by
          ``torch.nn.Module.state_dict()``.
        optimizer_state (dict): The state of the training optimizer as given by
          ``torch.optim.Optimizer.state_dict()``.
    """

    epoch: int
    model_state: dict
    optimizer_state: dict


class FinalState(NamedTuple):
    model_type: ModelType

    bg_model_state: dict
    bth_model_state: dict

    def save(self, filepath: Path | str) -> None:
        torch.save(
            {
                "model_type": self.model_type,
                "bg_model_state": self.bg_model_state,
                "bth_model_state": self.bth_model_state,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: Path | str) -> "FinalState":
        obj = torch.load(filepath)
        return cls(
            obj["model_state"],
            obj["bg_model_state"],
            obj["bth_model_state"],
        )
