import os
import argparse
import re
from pathlib import Path
import numpy as np
import toml
import pandas as pd
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

def populate(raw_image_path, image_name, FEATURE_PATH, PREPROCESSED_PATH, DROPLET_PATH, EXPERIMENT_PATH = None, omit_droplet_dataset = False) -> None:
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

    if EXPERIMENT_PATH is not None:
        print("Creating Embeddings")
        config = toml.load(EXPERIMENT_PATH / "0003.toml")["config"]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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



