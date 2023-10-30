from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from attrs import asdict
from torch import save as tsave
from torch import load as tload

from psst import Range


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

    phi_range: Range
    nw_range: Range
    visc_range: Range
    bg_range: Range
    bth_range: Range

    def save(self, filepath: Path | str) -> None:
        obj: dict[str, Any] = dict()
        for key, val in self._asdict().items():
            if isinstance(val, Range):
                obj[key] = asdict(val)
            else:
                obj[key] = val

        tsave(obj, filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "FinalState":
        obj = tload(filepath)
        return cls(
            obj["model_state"],
            obj["bg_model_state"],
            obj["bth_model_state"],
            Range.from_dict(obj["phi_range"]),
            Range.from_dict(obj["nw_range"]),
            Range.from_dict(obj["visc_range"]),
            Range.from_dict(obj["bg_range"]),
            Range.from_dict(obj["bth_range"]),
        )
