from __future__ import annotations
from functools import partial
from typing import Any
from warnings import warn

import optuna
import torch

import psst
from psst import Parameter, models
from .config import OptimConfig


# Maybe update training.train.train with this, call that from here?
def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: psst.SampleGenerator,
    num_samples: int,
    b_range: psst.Range,
) -> tuple[float, float]:
    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        num_samples = num_batches * batch_size

    avg_loss: float = 0.0
    avg_error: float = 0.0

    if generator.parameter is Parameter.bg:
        idx = 0
    else:
        idx = 1

    model.train()
    for samples, *params, _ in generator(num_batches):
        true_value = params[idx]

        optimizer.zero_grad()
        pred: torch.Tensor = model(samples)

        loss: torch.Tensor = loss_fn(pred, true_value)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        b_range.unnormalize(true_value)
        b_range.unnormalize(pred)

        avg_error += (torch.abs(true_value - pred) / true_value).mean().item()

    return avg_loss / num_batches, avg_error / num_batches


def objective(
    trial: optuna.Trial,
    *,
    generator: psst.SampleGenerator,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optim_config: OptimConfig,
    b_range: psst.Range,
) -> float:
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 0.9, log=False)
    beta_2 = trial.suggest_float("beta_2", 0.75, 0.999, log=False)
    eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps
    )

    avg_error: float = 0.0
    for epoch in range(20):
        _, avg_error = train(
            model,
            optimizer,
            loss_fn,
            generator,
            optim_config.num_samples_train,
            b_range,
        )

        trial.report(avg_error, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned

    return avg_error


def optimize(
    config: OptimConfig,
    *,
    parameter: Parameter,
    range_config: psst.RangeConfig,
    generator_config: psst.GeneratorConfig,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if parameter is Parameter.bg:
        study_name = "Bg_Hyperparam_Tune"
    else:
        study_name = "Bth_Hyperparam_Tune"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=9, interval_steps=2, n_min_trials=5
        ),
    )

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    generator = psst.SampleGenerator(
        parameter,
        range_config,
        generator_config,
        device=torch.device(device),
    )

    if parameter is Parameter.bg:
        b_range = range_config.bg_range
    else:
        b_range = range_config.bth_range

    func = partial(
        objective,
        generator=generator,
        model=model,
        loss_fn=loss_fn,
        optim_config=config,
        b_range=b_range,
    )
    study.optimize(func, n_trials=config.num_trials)

    states = [t.state for t in study.trials]
    num_pruned = states.count(optuna.trial.TrialState.PRUNED)
    num_completed = states.count(optuna.trial.TrialState.COMPLETE)

    print("Study statistics: ")
    print("\tNumber of finished trials: ", len(study.trials))
    print("\tNumber of pruned trials: ", num_pruned)
    print("\tNumber of complete trials: ", num_completed)

    print("Best trial:")
    trial = study.best_trial

    print("\tValue: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    return trial.params


def run(
    opt_config: OptimConfig,
    model_config: psst.ModelConfig,
    range_config: psst.RangeConfig,
    generator_config: psst.GeneratorConfig,
    device: str = "cpu",
    overwrite: bool = False,
    load_file: str = "",
    **kwargs,
):
    if len(kwargs):
        warn(f"WARNING: Ignoring unsupported options to evaluate:\n\t{kwargs.keys()}")

    outpath = psst.configuration.validate_filepath(
        opt_config.output_file, exists=(None if overwrite else False)
    )

    if model_config.bg_name is not None and model_config.bth_name is not None:
        raise ValueError(
            "Cannot optimize both Bg and Bth models simultaneously."
            " Please only specify one of 'bg' or 'bth'."
        )
    if model_config.bg_name is not None:
        model = getattr(models, model_config.bg_name)
        parameter = Parameter.bg
    elif model_config.bth_name is not None:
        model = getattr(models, model_config.bth_name)
        parameter = Parameter.bth
    else:
        raise ValueError(
            "One of either bg or bth must be specified. Please specify one."
        )

    params = optimize(
        opt_config,
        parameter=parameter,
        range_config=range_config,
        generator_config=generator_config,
        model=model,
        device=device,
    )

    psst.configuration.write_dict_to_file(params, outpath, overwrite)

    return 0
