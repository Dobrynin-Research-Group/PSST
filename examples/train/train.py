from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import attrs
import torch

import psst
from psst.models import Inception3, Vgg13
from psst import Parameter
from psst.training import *


@attrs.define
class CmdLineArgs:
    model: torch.nn.Module
    parameter: Parameter
    train_config: TrainingConfig
    range_config: psst.RangeConfig
    gen_config: psst.GeneratorConfig
    adam_config: psst.AdamConfig
    device: torch.device
    from_checkpoint: Optional[Path] = None


def parse_args() -> CmdLineArgs:
    parser = ArgumentParser()
    parser.add_argument(
        "config_file",
        help="main configuration file",
    )
    parser.add_argument(
        "-p",
        required=True,
        choices=["bg", "bth"],
        help="blob parameter to train the model for; can either be 'bg'"
        " (good solvent parameter) or 'bth' (thermal blob parameter)",
    )
    parser.add_argument(
        "-r",
        metavar="range_config_file",
        help="separate configuration file for RangeConfig",
    )
    parser.add_argument(
        "-g",
        metavar="generator_config_file",
        help="separate configuration file for GeneratorConfig",
    )
    parser.add_argument(
        "-a",
        metavar="adam_config_file",
        help="separate configuration file for AdamConfig",
    )
    parser.add_argument(
        "-m",
        choices=["Inception3", "Vgg13"],
        metavar="model_name",
        help="neural network model to train (can be specified here or in"
        " config_file)",
    )
    parser.add_argument(
        "-d",
        metavar="device_name",
        default="cpu",
        help="name of the device on which to run training (default is 'cpu')",
    )
    parser.add_argument(
        "-l",
        metavar="checkpoint_file",
        help="checkpoint file from which to load training state",
    )

    args = parser.parse_args()

    if args.m == "Inception3":
        model = Inception3()
    else:
        model = Vgg13()

    parameter = Parameter.bg if args.p == "bg" else Parameter.bth

    train_config = TrainingConfig.from_file(args.config_file)
    range_config = psst.RangeConfig.from_file(args.r)
    if args.a:
        adam_config = psst.AdamConfig.from_file(args.a)
    else:
        adam_config = psst.AdamConfig()

    from_checkpoint = args.l
    if from_checkpoint:
        from_checkpoint = Path(from_checkpoint)
        assert from_checkpoint.is_file()

    if args.d:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return CmdLineArgs(
        model,
        parameter,
        train_config,
        range_config,
        generator_config,
        adam_config,
        device,
        from_checkpoint,
    )


def main():
    args = parse_args()

    generator = psst.SampleGenerator(
        args.parameter,
        args.range_config,
        args.gen_config,
        device=args.device,
    )

    model = args.model
    model.to(device=args.device)
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), **args.adam_config)
    start_epoch = 0
    if args.from_checkpoint:
        chkpt: psst.Checkpoint = torch.load(args.from_checkpoint)
        start_epoch = chkpt.epoch
        model.load_state_dict(chkpt.model_state)
        optimizer.load_state_dict(chkpt.optimizer_state)

    train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        generator=generator,
        train_config=args.train_config,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
