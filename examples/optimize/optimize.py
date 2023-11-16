from argparse import ArgumentParser
from pathlib import Path

import torch

import psst
from psst import Parameter
from psst.optimization import OptimConfig, optimize


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
        "-r",
        required=True,
        metavar="range_config_file",
        help="Selects the configuration file for Ranges",
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
    range_config_file = Path(args.r)

    if args.d:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    optim_config = OptimConfig.from_file(optim_config_file)
    generator_config = psst.GeneratorConfig.from_file(gen_config_file)
    range_config = psst.RangeConfig.from_file(range_config_file)
    generator = psst.SampleGenerator(
        parameter, range_config, generator_config, device=device
    )

    model = Model().to(device=device)
    loss_fn = torch.nn.MSELoss()

    params = optimize(optim_config, generator, model, loss_fn)
    adam_config = psst.AdamConfig(
        lr=params["lr"], eps=params["eps"], betas=(params["beta_1"], params["beta_2"])
    )
    adam_config.to_yaml(outfile, overwrite=overwrite)


if __name__ == "__main__":
    main()
