from typing import NamedTuple

__all__ = ["Checkpoint"]


class Checkpoint(NamedTuple):
    """Represents a state during training. Can be easily saved to file `filepath` with

    ```python
    >>> chkpt = psst.Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
    >>> torch.save(chkpt, filepath)
    ```

    Args:
        epoch (int): The number of completed training cycles.
        model_name (ModelName): The name of the neural network used.
        model_state (dict): The state of the neural network model as given by
          ``torch.nn.Module.state_dict()``.
        optimizer_state (dict): The state of the training optimizer as given by
          ``torch.optim.Optimizer.state_dict()``.
    """

    epoch: int
    model_state: dict
    optimizer_state: dict
