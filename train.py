import os
import argparse
from pathlib import Path
import numpy as np
import random
import toml
import logging
import torch

from models.DeepRanking import DeepRanking

PROJECT_PATH = Path(os.path.dirname(__file__))
EXPERIMENTS_PATH = Path(PROJECT_PATH / "experiments")


def print_cuda_status():
    # Set up device on GPU, if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    print()

    # Additional info for Cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage: ')
        print('Allocated: ', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached: ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def prep_experiment_dir(experiment):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(experiment))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)])
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return Path(experiment_dir)


def main() -> None:
    print_cuda_status()
    parser = argparse.ArgumentParser(description=
                                     """
                                     Train an experiment and save it as a checkpoint
                                     """)
    parser.add_argument("experiment", type=str, help=f"ID of the experiment.")
    args = parser.parse_args()
    config = toml.load(EXPERIMENTS_PATH / f"{args.experiment}.toml")["config"]

    experiment_dir = prep_experiment_dir(args.experiment)
    config["experiment_dir"] = experiment_dir

    # Make results reproducible
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    logging.basicConfig(filename=str(experiment_dir / 'output.log'), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    if config['model'] == "DeepRanking":
        if "checkpoint" in config:
            model = DeepRanking(config)
            checkpoint_path = config["checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location=model.device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.step = checkpoint['step']
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("loaded checkpoint", checkpoint_path)
        else:
            model = DeepRanking(config)
    else:
        raise Exception(f"Unknown model {config['model']}")
    logging.info("Training model...")
    model.train()


if __name__ == "__main__":
    main()
