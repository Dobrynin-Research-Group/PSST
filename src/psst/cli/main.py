from argparse import ArgumentParser
from functools import singledispatch
from pathlib import Path
from typing import NamedTuple, Optional

import psst
from psst import models, optimization, training, evaluation

MODELS = [m for m in dir(models) if m[0] != "_"]


class ConfigCollection(NamedTuple):
    command: optimization.OptimConfig | training.TrainingConfig | evaluation.EvaluationConfig
    model_config: psst.ModelConfig
    range_config: psst.RangeConfig
    adam_config: Optional[psst.AdamConfig] = None
    generator_config: Optional[psst.GeneratorConfig] = None


def parse_config(config_file: Path) -> ConfigCollection:
    d = psst.configuration.get_dict_from_file(config_file)

    command_list = ["training", "optimization", "evaluation"]
    command = None
    for cmd in command_list:
        if cmd in d:
            if command is not None:
                raise ValueError(
                    f"Only one command is allowed in the config file, found '{command}'"
                    f" and '{cmd}'"
                )
            command = cmd

    if command is None:
        raise RuntimeError(
            "Config file must have one of 'training', 'optimization', or 'evaluation'"
            "as a key"
        )

    if command == "training":
        cmd_config = training.TrainingConfig.from_dict(d.pop(command))
    elif command == "optimization":
        cmd_config = optimization.OptimConfig.from_dict(d.pop(command))
    else:
        cmd_config = evaluation.EvaluationConfig.from_dict(d.pop(command))

    model_value = d["models"]
    if isinstance(model_value, str):
        model_config = psst.ModelConfig.from_file(model_value)
    elif isinstance(model_value, dict):
        model_config = psst.ModelConfig.from_dict(model_value)
    else:
        raise TypeError("blah")

    range_value = d["ranges"]
    if isinstance(range_value, str):
        range_config = psst.RangeConfig.from_file(range_value)
    elif isinstance(range_value, dict):
        range_config = psst.RangeConfig.from_dict(range_value)
    else:
        raise TypeError("blah")

    adam_value = d.get("adam")
    if adam_value is None:
        adam_config = None
    elif isinstance(adam_value, str):
        adam_config = psst.AdamConfig.from_file(adam_value)
    elif isinstance(adam_value, dict):
        adam_config = psst.AdamConfig.from_dict(adam_value)
    else:
        raise TypeError("blah")

    gen_value = d.get("generator")
    if gen_value is None:
        generator_config = None
    elif isinstance(gen_value, str):
        generator_config = psst.GeneratorConfig.from_file(gen_value)
    elif isinstance(gen_value, dict):
        generator_config = psst.GeneratorConfig.from_dict(gen_value)
    else:
        raise TypeError("blah")

    return ConfigCollection(
        cmd_config,
        model_config,
        range_config,
        adam_config,
        generator_config,
    )


@singledispatch
def run(cmd_config, **kwargs):
    raise NotImplementedError("How did this happen?")


@run.register(training.TrainingConfig)
def _(cmd_config: training.TrainingConfig, **kwargs):
    return training.run(cmd_config, **kwargs)


@run.register(optimization.OptimConfig)
def _(cmd_config: optimization.OptimConfig, **kwargs):
    return optimization.run(cmd_config, **kwargs)


@run.register(evaluation.EvaluationConfig)
def _(cmd_config: evaluation.EvaluationConfig, **kwargs):
    return evaluation.run(cmd_config, **kwargs)


def main():
    parser = ArgumentParser(prog="psst", description="TBD")
    parser.add_argument(
        "config_file",
        help="configuration file to be processed for the given command",
    )

    parser.add_argument(
        "-d",
        dest="device",
        metavar="device",
        default="cpu",
        help="",
    )
    parser.add_argument(
        "-w",
        dest="overwrite",
        action="store_true",
        help="flag to allow overwriting of output files",
    )
    parser.add_argument(  # not evaluate
        "-l",
        dest="load_file",
        metavar="load-checkpoint-file",
        help="",
    )

    args = parser.parse_args()

    config_file = Path(args.config_file)
    if not config_file.is_file():
        raise FileNotFoundError(f"Could not locate config_file: {config_file}")

    configs = parse_config(config_file)
    d = configs._asdict()
    command = d.pop("command")

    return run(
        command,
        **d,
        device=args.device,
        overwrite=args.overwrite,
        load_file=args.load_file,
    )
