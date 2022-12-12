import argparse
import os
import toml
import joblib
import pandas as pd
import re
from pathlib import Path
from glob import glob

import torch
from utils.models_common import compute_similar_images, create_embeddings, FolderDataset
from models.AE import AE
from models.ViTMAE import ViTMAE
import numpy as np

PROJECT_PATH = Path(os.path.dirname(__file__))
PROJECT_PATH = Path(PROJECT_PATH)
import numpy as np

PROJECT_PATH = Path(os.path.dirname(__file__))
DATA_PATH = Path(PROJECT_PATH / "data")
EXPERIMENTS_PATH = Path(PROJECT_PATH / "experiments")
FEATURE_PATH = Path(DATA_PATH / "03_features")


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
                                     Train an experiment and save it as a checkpoint
                                     """)
    parser.add_argument("experiment", type=str, help=f"ID of the experiment.")
    args = parser.parse_args()
    config = toml.load(EXPERIMENTS_PATH / f"{args.experiment}.toml")["config"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    print(config['model'])
    if config['model'] == "ConvAE":
        model = AE(config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
    elif config['model'] == 'VitMae':
        model = ViTMAE(config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise Exception(f"Unknown model {config['model']}")
    transforms = None
    train_dataset = FolderDataset(config["train_dataset"], transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    embeddings = create_embeddings(device, train_dataloader, config, model.model)
    np.save("embeddings.npy", embeddings)
    embeddings = embeddings
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]))
    print(embeddings.shape)
    indexes = np.array([0,0])
    for i in sorted(os.listdir(config["train_dataset"])):
        print(i[4:8])
        indexes = np.vstack((indexes, np.array([i[1], i[4:8]])))
        #print(np.array([i[1], i[4:8]]))
    indexes = indexes[1:, :]
    embeddings = np.concatenate((indexes, embeddings), axis=1)
    pd.DataFrame(embeddings).to_csv(Path(FEATURE_PATH/ f"embeddings_{config['image_name']}.csv"))
    print(indexes.shape)

if __name__ == "__main__":
    main()
