import os
import argparse
import re
from pathlib import Path
import numpy as np
import toml
import pandas as pd
import random
import logging
import torch
from preprocessing.preprocessing import preprocess
from preprocessing.preprocessing import preprocess_alt_franc
from preprocessing.preprocessing import preprocess_alt_featextr
from utils.models_common import create_embeddings, FolderDataset
from models.ViTMAE import ViTMAE
from data_creation.droplets_and_cells import generate_output
from data_creation.droplets_and_cells import generate_output_from_ndarray
from data_creation.droplet_retriever import create_dataset_from_ndarray, resize_patch, create_dataset_cell_enhanced, create_dataset_cell_ndarray, create_dataset_cell_enhanced_from_ndarray


def raw_to_preprocessed(raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path) -> np.ndarray:
    preprocessed_image = preprocess(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_{image_name}.npy")
    np.save(path, preprocessed_image)
    return preprocessed_image


def raw_to_preprocessed_alt_franc(raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path) -> np.ndarray:
    preprocessed_image = preprocess_alt_franc(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_drpdtc_{image_name}.npy")
    np.save(path, preprocessed_image)
    return preprocessed_image

def raw_to_preprocessed_featextr(raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path) -> np.ndarray:
    preprocessed_image = preprocess_alt_featextr(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_bf_{image_name}.npy")
    np.save(path, preprocessed_image[:, 0, :, :])
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_dapi_{image_name}.npy")
    np.save(path, preprocessed_image[:, 1, :, :])
    return preprocessed_image
 
def save_droplet_images(dataset: np.ndarray, image_name: str, DROPLET_PATH: Path) -> None:
    folder_path = Path(DROPLET_PATH / image_name)

    try:
        os.mkdir(folder_path)
    except FileExistsError as _:
        pass

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            # Changed patch size from 100x100 to 64x64 as 100x100 is just way too big considering the droplets are ca 40 pixels across
            patch = np.float64(resize_patch(dataset[i][j]['patch'], 64))
            patch[patch == 0.0] = np.nan
            thresh = np.nanquantile(patch, 0.5, axis = (1, 2))
            patch[np.isnan(patch)] = 0.0
            patch = np.uint16(np.clip((patch - thresh[:, None, None]) / (2**16 - 1 - thresh[:, None, None]), 0, 1.0) * (2**16 - 1))
            patch[np.isnan(patch)] = 0.0
            np.save(Path(folder_path / ("f" + str(i) + "_d" + str(j).zfill(4))), patch)

def prep_experiment_dir(EXPERIMENTS_PATH, experiment):
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

def print_cuda_status():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

def populate(raw_image_path, image_name, FEATURE_PATH, PREPROCESSED_PATH, DROPLET_PATH, EXPERIMENT_PATH = None, omit_droplet_dataset = False, train_model = False, create_emb =  False) -> None:
    print("Image preprocessing for droplet detection ...")
    preprocessed_image = raw_to_preprocessed_alt_franc(raw_image_path, image_name, PREPROCESSED_PATH)
    print("Detecting droplets and cells...")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    # generate_output_from_ndarray(preprocessed_image, droplet_feature_path, cell_feature_path, True, str(PREPROCESSED_PATH) + "/", True)
    generate_output_from_ndarray(preprocessed_image, droplet_feature_path, cell_feature_path, True, "", False)

    print("Image preprocessing for feature extraction ...")
    preprocessed_image = raw_to_preprocessed_featextr(raw_image_path, image_name, PREPROCESSED_PATH)
        
    if not omit_droplet_dataset:
        print("Creating droplet images...")
        droplet_images_dataset = create_dataset_cell_enhanced_from_ndarray([0], preprocessed_image, droplet_feature_path, cell_feature_path, allFrames=True, buffer = -2, suppress_rest = True, suppression_slack = -3)
        save_droplet_images(droplet_images_dataset, image_name, DROPLET_PATH)
        path = Path(FEATURE_PATH/ f"fulldataset_{image_name}.npy")
        np.save(path, droplet_images_dataset)
    else:
        print("Omitting droplet dataset creation...")

    model = None
    config = None
    if train_model:
        print_cuda_status()
        experiment_dir = prep_experiment_dir(EXPERIMENT_PATH, image_name)
        config = toml.load(EXPERIMENT_PATH/"model.toml")["config"]

        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        if config["train_dataset"] == '':
            config["train_dataset"] = Path(DROPLET_PATH, f"{image_name}")
            config["val_dataset"] = Path(DROPLET_PATH, f"{image_name}")
        config['experiment_dir'] = experiment_dir

        logging.basicConfig(filename=str(experiment_dir / 'output.log'), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

        if "checkpoint" in config:
            model = ViTMAE(config)
            checkpoint_path = config["checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location=model.device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.step = checkpoint['step']
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("loaded checkpoint", checkpoint_path)
        else:
            model = ViTMAE(config)

        logging.info("Training model...")
        model.train()

    if create_emb:
        print("Creating Embeddings")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        if not train_model:
            config = toml.load(EXPERIMENT_PATH / "0003.toml")["config"]

            checkpoint = torch.load(config["checkpoint"], map_location=device)

            model = ViTMAE(config)
            model.model.load_state_dict(checkpoint['model_state_dict'])

        transforms = None
        train_dataset = FolderDataset(Path(DROPLET_PATH, f"{image_name}"), transforms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        embeddings = create_embeddings(device, train_dataloader, config, model.model)
        embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]))
        indexes = np.array([0,0])
        marker1, marker2, marker3 = 'f', '_d', '.npy'
        regex1 = marker1 + '(.+?)' + marker2
        regex2 = marker2 + '(.+?)' + marker3
        for i in sorted(os.listdir(Path(DROPLET_PATH, f"{image_name}"))):
            frame = int(re.search(regex1, i).group(1))
            droplet_id = int(re.search(regex2, i).group(1))
            indexes = np.vstack((indexes, np.array([frame, droplet_id])))

        indexes = indexes[1:, :]
        embeddings = np.concatenate((indexes, embeddings), axis=1)
        pd.DataFrame(embeddings).to_csv(Path(FEATURE_PATH / f"embeddings_{image_name}.csv"))




    # print("Preprocessing raw image...")
    # preprocessed_image = raw_to_preprocessed(raw_image_path, image_name, PREPROCESSED_PATH)

    # print("Detecting droplets and cells...")
    # droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    # cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    # generate_output(raw_image_path, droplet_feature_path, cell_feature_path, True, "", False)

    # print("Creating droplet images...")
    # #droplet_images_dataset = create_dataset_from_ndarray([0], ['BF'], preprocessed_image, droplet_feature_path, allFrames=True, allChannels=True)
    # droplet_images_dataset = np.array(create_dataset_cell_ndarray(None, ["BF"], preprocessed_image, droplet_feature_path, cell_feature_path, allFrames=True, buffer=-2, suppress_rest=True, suppression_slack=-3, median_filter_preprocess=False))
    # save_droplet_images(droplet_images_dataset, image_name, DROPLET_PATH)
    # path = Path(FEATURE_PATH/ f"fulldataset_{image_name}.npy")
    # np.save(path, droplet_images_dataset)



