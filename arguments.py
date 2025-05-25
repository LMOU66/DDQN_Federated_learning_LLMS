import argparse
import dataclasses
from dataclasses import dataclass
import logging

@dataclass
class Args:
    num_clients: int
    num_rounds: int
    lora_r: int
    client_epochs: int
    client_ckpt: str
    client_lr: float
    rank: int
    data_path: str
    data_name: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    # Add command-line
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--client_epochs', type=int, default=5)
    parser.add_argument('--client_ckpt', type=str, default="distilbert-base-uncased")

    parser.add_argument('--client_lr', type=float, default=5e-5)

    parser.add_argument('--rank', type=int, default=0)

    parser.add_argument('--data_path', type=str, default="glue")
    parser.add_argument('--data_name', type=str, default="cola")

    # Parse the command-line arguments
    args = parser.parse_args()

    return Args(**vars(args))

def pretty_print_args(args: Args):
    args_dict = dataclasses.asdict(args)
    args_str = "\n".join(f"{k}:\t{v}" for k, v in args_dict.items())
    logging.info(args_str)
