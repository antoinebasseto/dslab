import argparse
import os
import toml
import joblib
import re
from pathlib import Path
from glob import glob

import torch
from utils.models_common import compute_similar_images, create_embeddings, FolderDataset
from models.AE import AE
import numpy as np

PROJECT_PATH = Path(os.path.dirname(__file__))
EXPERIMENTS_PATH = Path(PROJECT_PATH / "experiments")


def index_to_id(index):
    return index // 10000, index % 10000


def id_to_index(id):
    return id[0] * 10000 + id[1]


def find_dataset_breaks(num_frames, dataset):
    breaks = np.zeros(num_frames)
    for i in range(num_frames):
        break_point = np.argmax(dataset >> i * 10000 - 1)
        breaks[i] = break_point
    return breaks


def tracking(embeddings, id, caller):
    num_frames = caller.config["num_frames"] - 1
    index = id_to_index(id)
    track = [id]
    breaks = find_dataset_breaks(num_frames)
    for i in range(num_frames - 1):
        track += [compute_similar_images(index, embeddings[breaks[i]:breaks[i + 1]], caller)]
    return track


def main() -> None:
    parser = argparse.ArgumentParser(description=
                                     """
Load an experiment checkpoint and use it to generate a submission. Store the submission
in the experiment directory as 'submission.csv'.
""")
    parser.add_argument("experiments", type=str,
                        help=f"list of experiments whose predictions you want to average into a submission.")
    parser.add_argument("id", type=int)

    args = parser.parse_args()
    print(args.experiments)
    for experiment in args.experiments:
        config = toml.load(EXPERIMENTS_PATH / f"{experiment}.toml")["config"]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        if config['model'] == "ConvAE":
            model = AE(config)
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception(f"Unknown model {config['model']}")
        transforms = None
        train_dataset = FolderDataset(config["train_dataset"], transforms)

        embeddings = create_embeddings(train_dataset, config, model)
        indexes = tracking(embeddings, args.id, config)
        return indexes




if __name__ == "__main__":
    main()
