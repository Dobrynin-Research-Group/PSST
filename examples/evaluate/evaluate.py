from argparse import ArgumentParser

import torch

import psst
import psst.evaluation as ev


def parse_args() -> tuple[ev.EvaluationConfig, psst.RangeConfig, torch.device]:
    parser = ArgumentParser()
    parser.add_argument(
        "configuration_file",
        help="File location of the configuration file for this evaluation.",
    )
    parser.add_argument(
        "-r",
        required=True,
        help="",
    )
    parser.add_argument(
        "-d",
        help="PyTorch device to use for inferencing (default: use the CPU).s",
    )

    args = parser.parse_args()

    eval_config = ev.EvaluationConfig.from_file(args.configuration_file)

    range_config = psst.RangeConfig.from_file(args.r)

    device = torch.device("cuda") if args.d else torch.device("cpu")

    return eval_config, range_config, device


def main():
    eval_config, range_config, device = parse_args()

    result = ev.evaluate_dataset(range_config, eval_config, device)

    return result


if __name__ == "__main__":
    main()
