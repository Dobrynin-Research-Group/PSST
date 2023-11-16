from __future__ import annotations
from functools import partial
import logging
from pathlib import Path
from warnings import warn

import torch

from psst import Parameter, SampleGenerator, Checkpoint, models
import psst
from .config import TrainingConfig


__all__ = ["train", "validate", "train_model", "run"]


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: SampleGenerator,
    num_samples: int,
) -> float:
    """The neural network model is trained based on the configuration parameters,
    evaluated by the loss function, and incrementally adjusted by the optimizer.

    Args:
        model (torch.nn.Module): Machine learning model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the model.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (SampleGenerator): Procedurally generates data for model training.
        num_samples (int): Number of samples to generate and train.

    Returns:
        float: Average loss over training cycle.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.",
            num_samples,
            batch_size,
            num_batches * batch_size,
        )
        num_samples = num_batches * batch_size

    choice = 1 if generator.parameter is Parameter.bth else 0
    avg_loss: float = 0.0

    model.train()
    count = 0
    log.info("Starting training run of %d batches", num_batches)
    for samples, *batch_values in generator(num_batches):
        optimizer.zero_grad()
        log.debug("Training batch %d", count / batch_size)
        pred: torch.Tensor = model(samples)

        log.debug("Computing loss")
        loss: torch.Tensor = loss_fn(pred, batch_values[choice])
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()

    return avg_loss / num_batches


def validate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    generator: SampleGenerator,
    num_samples: int,
) -> float:
    """Tests the neural network model based on the configuration parameters using
    the loss function.

    Args:
        model (torch.nn.Module): Machine learning model to be validated.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (SampleGenerator): Procedurally generates data for model validation.
        num_samples (int): Number of samples to generate and validate.

    Returns:
        float: Average loss over validation cycle.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.",
            num_samples,
            batch_size,
            num_batches * batch_size,
        )
        num_samples = num_batches * batch_size

    choice = 1 if generator.parameter is Parameter.bth else 0
    avg_loss: float = 0.0

    model.eval()
    log.info("Starting validation run of %d batches", num_batches)
    with torch.no_grad():
        for i, (samples, *batch_values) in enumerate(generator(num_batches)):
            log.debug("Testing batch %d", i)
            pred = model(samples)

            log.debug("Computing loss")
            loss = loss_fn(pred, batch_values[choice])
            avg_loss += loss

    return avg_loss / num_batches


def update_chkpt_on_new_loss(
    chkpt: Checkpoint,
    loss: float,
    min_loss: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Checkpoint:
    if loss < min_loss:
        min_loss = loss
        chkpt = Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
    return chkpt


def update_chkpt_every(
    chkpt: Checkpoint,
    loss: float,
    min_loss: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    every: int,
) -> Checkpoint:
    if epoch % every == 0:
        min_loss = loss
        chkpt = Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
    return chkpt


def no_update(
    chkpt: Checkpoint,
    loss: float,
    min_loss: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Checkpoint:
    return chkpt


def train_model(
    train_config: TrainingConfig,
    *,
    generator: psst.SampleGenerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    start_epoch: int = 0,
):
    """Run the model through `num_epochs` train/test cycles.

    Args:
        model (torch.nn.Module): Machine learning model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the model.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (:class:`SampleGenerator`): Procedurally generates data for model
          training.
        train_config (:class:`TrainingConfig`): Configuration object loaded from YAML
          file for training cycles.
        start_epoch (int): The index of the first epoch to run (useful for continuing
          training from a checkpoint).
    """

    loc_chkpt = Checkpoint(start_epoch, model.state_dict(), optimizer.state_dict())
    min_loss = 1e6
    loss = min_loss

    if train_config.checkpoint_frequency < 0:

        def update_chkpt(epoch: int) -> tuple[float, Checkpoint]:
            if loss < min_loss:
                ml = loss
                chkpt = Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
            else:
                ml = min_loss
                chkpt = loc_chkpt
            return ml, chkpt

    elif train_config.checkpoint_frequency > 0:

        def update_chkpt(epoch: int) -> tuple[float, Checkpoint]:
            if epoch % train_config.checkpoint_frequency == 0:
                ml = min(loss, min_loss)
                chkpt = Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
            else:
                ml = min_loss
                chkpt = loc_chkpt
            return ml, chkpt

    else:

        def update_chkpt(epoch: int) -> tuple[float, Checkpoint]:
            return min_loss, loc_chkpt

    for epoch in range(start_epoch, train_config.num_epochs):
        min_loss, loc_chkpt = update_chkpt(epoch)
        train(model, optimizer, loss_fn, generator, train_config.num_samples_train)
        loss = validate(model, loss_fn, generator, train_config.num_samples_test)

    if train_config.checkpoint_frequency != 0:
        torch.save(loc_chkpt, train_config.checkpoint_filename)


def run(
    train_config: TrainingConfig,
    *,
    model_config: psst.ModelConfig,
    range_config: psst.RangeConfig,
    adam_config: psst.AdamConfig,
    generator_config: psst.GeneratorConfig,
    device: str = "cpu",
    overwrite: bool = False,
    load_file: str | Path | None = "",
    **kwargs,
):
    if len(kwargs):
        warn(f"WARNING: Ignoring unsupported options to evaluate:\n\t{kwargs.keys()}")

    exists = None if overwrite else False
    train_config.checkpoint_filename = psst.configuration.validate_filepath(
        train_config.checkpoint_filename, exists=exists
    )
    if load_file:
        load_file = psst.configuration.validate_filepath(load_file, exists=exists)

    dev = torch.device(device)

    if model_config.bg_name is not None and model_config.bth_name is not None:
        raise ValueError(
            "Cannot optimize both Bg and Bth models simultaneously."
            " Please only specify one of 'bg' or 'bth'."
        )
    if model_config.bg_name is not None:
        model_cls: type[torch.nn.Module] = getattr(models, model_config.bg_name)
        parameter = Parameter.bg
    elif model_config.bth_name is not None:
        model_cls: type[torch.nn.Module] = getattr(models, model_config.bth_name)
        parameter = Parameter.bth
    else:
        raise ValueError(
            "One of either bg or bth must be specified. Please specify one."
        )
    model = model_cls().to(dev)

    start_epoch = 0
    if load_file:
        chkpt: Checkpoint = torch.load(load_file)
        model.load_state_dict(chkpt.model_state)
        start_epoch = chkpt.epoch

    adam = torch.optim.Adam(model.parameters(), **adam_config)
    generator = psst.SampleGenerator(
        parameter, range_config, generator_config, device=dev
    )

    loss_fn = torch.nn.MSELoss()

    train_model(
        train_config,
        generator=generator,
        model=model,
        optimizer=adam,
        loss_fn=loss_fn,
        start_epoch=start_epoch,
    )

    return 0
