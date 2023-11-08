from __future__ import annotations
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import torch

import psst
from psst import Parameter
from .config import OptimConfig


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: psst.SampleGenerator,
    num_samples: int,
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
        pred: psst.NormedTensor = model(samples)

        loss: torch.Tensor = loss_fn(pred, true_value)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        true_value.unnormalize()
        pred.unnormalize()

        avg_error += (torch.abs(true_value - pred) / true_value).mean().item()

    return avg_loss / num_batches, avg_error / num_batches


def objective(
    trial: optuna.Trial,
    *,
    generator: psst.SampleGenerator,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optim_config: OptimConfig,
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
            model, optimizer, loss_fn, generator, optim_config.num_samples_per_epoch
        )

        trial.report(avg_error, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned

    return avg_error


def run(
    optim_config: OptimConfig,
    generator: psst.SampleGenerator,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
) -> dict[str, Any]:
    if generator.parameter is Parameter.bg:
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

    func = partial(
        objective,
        generator=generator,
        model=model,
        loss_fn=loss_fn,
        optim_config=optim_config,
    )
    study.optimize(func, n_trials=optim_config.num_trials)

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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "output_file",
        help="Selects file location to store optimized results in (YAML file)",
    )
    parser.add_argument(
        "-c",
        required=True,
        metavar="optimizer_config_file",
        help="Selects the configuration file for the optimizer",
    )
    parser.add_argument(
        "-g",
        required=True,
        metavar="generator_config_file",
        help="Selects the configuration file for the SampleGenerator",
    )
    parser.add_argument(
        "-p",
        required=True,
        choices=["bg", "bth"],
        help="Selects optimization for either the Bg parameter"
        " (good solvent parameter) or the Bth parameter (thermal blob parameter)",
    )
    parser.add_argument(
        "-m",
        choices=["Inception3", "Vgg13"],
        default="Inception3",
        metavar="model_name",
        help="Selects the neural network model to optimize training for"
        " (default: Inception3)",
    )
    parser.add_argument(
        "-d",
        action="store_true",
        help="Enables CUDA device offloading (default: use the CPU)",
    )
    parser.add_argument(
        "-w", action="store_true", help="Allows the output_file to be overwritten"
    )

    args = parser.parse_args()

    outfile = Path(args.output_file)
    overwrite = args.w
    if outfile.is_file() and not overwrite:
        raise FileExistsError(
            "Output file exists. Pass the -w flag to allow overwriting it."
        )

    if args.m == "Inception3":
        from psst.models import Inception3 as Model
    else:
        from psst.models import Vgg13 as Model

    parameter = Parameter.bg if args.p == "bg" else Parameter.bth
    optim_config_file = Path(args.c)
    gen_config_file = Path(args.g)

    if args.d:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    optim_config = OptimConfig.from_file(optim_config_file)
    generator_config = psst.GeneratorConfig.from_file(gen_config_file)
    generator = psst.SampleGenerator(parameter, generator_config, device=device)

    model = Model().to(device=device)
    loss_fn = torch.nn.MSELoss()

    params = run(optim_config, generator, model, loss_fn)
    adam_config = psst.AdamConfig(
        lr=params["lr"], eps=params["eps"], betas=(params["beta_1"], params["beta_2"])
    )
    adam_config.to_yaml(outfile, overwrite=overwrite)


if __name__ == "__main__":
    main()
