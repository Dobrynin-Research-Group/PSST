from argparse import ArgumentParser
from pathlib import Path

import torch

import psst
from psst import Parameter


def parse_args() -> (
    tuple[type[torch.nn.Module], Parameter, Path, Path, Path, Path, torch.device]
):
    parser = ArgumentParser()
    parser.add_argument(
        "checkpoint_file",
        help="Selects file location to store trained model and optimizer states",
    )
    parser.add_argument(
        "-c",
        required=True,
        metavar="training_config_file",
        help="Selects the configuration file for the training",
    )
    parser.add_argument(
        "-g",
        required=True,
        metavar="generator_config_file",
        help="Selects the configuration file for the SampleGenerator",
    )
    parser.add_argument(
        "-a",
        metavar="adam_config_file",
        help="Selects the configuration file for the Adam optimizer"
        " (if not set, a default set of values will be used)",
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
        "-l",
        metavar="checkpoint_file",
        help="Load training state from checkpoint file",
    )
    parser.add_argument(
        "-w", action="store_true", help="Allows the checkpoint_file to be overwritten"
    )

    args = parser.parse_args()

    checkpoint_file = Path(args.checkpoint_file)
    overwrite = args.w
    if checkpoint_file.is_file() and not overwrite:
        raise FileExistsError(
            "Checkpoint file exists. Pass the -w flag to allow overwriting it."
        )

    if args.m == "Inception3":
        from psst.models import Inception3 as Model
    else:
        from psst.models import Vgg13 as Model

    parameter = Parameter.bg if args.p == "bg" else Parameter.bth
    run_config_file = Path(args.c)
    adam_config_file = Path(args.a)
    gen_config_file = Path(args.g)

    if args.d:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return (
        Model,
        parameter,
        checkpoint_file,
        run_config_file,
        adam_config_file,
        gen_config_file,
        device,
    )


def main():
    (
        Model,
        parameter,
        checkpoint_file,
        run_config_file,
        adam_config_file,
        gen_config_file,
        device,
    ) = parse_args()

    if adam_config_file:
        adam_config = psst.AdamConfig.from_file(adam_config_file)
    else:
        adam_config = psst.AdamConfig()

    run_config = psst.RunConfig.from_file(run_config_file)
    generator_config = psst.GeneratorConfig.from_file(gen_config_file)
    generator = psst.SampleGenerator(parameter, generator_config, device=device)

    model = Model().to(device=device)
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), **adam_config)
    start_epoch = 0
    if checkpoint_file:
        chkpt: psst.Checkpoint = torch.load(checkpoint_file)
        start_epoch = chkpt.epoch
        model.load_state_dict(chkpt.model_state)
        optimizer.load_state_dict(chkpt.optimizer_state)

    loss_fn = torch.nn.MSELoss()
    generator = psst.SampleGenerator(Parameter.bg, generator_config, device=device)

    psst.train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        generator=generator,
        start_epoch=start_epoch,
        num_epochs=run_config.num_epochs,
        num_samples_train=run_config.num_samples_train,
        num_samples_test=run_config.num_samples_test,
        checkpoint_file=checkpoint_file,
        checkpoint_frequency=run_config.checkpoint_frequency,
    )

    torch.save(
        psst.Checkpoint(
            run_config.num_epochs,
            model.state_dict(),
            optimizer.state_dict(),
        ),
        checkpoint_file,
    )


if __name__ == "__main__":
    main()
