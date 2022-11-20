import os
import argparse
from pathlib import Path
import numpy as np
from preprocessing.preprocessing import preprocess
from data_creation.droplets_and_cells import generate_output
from data_creation.droplet_retriever import create_dataset_from_ndarray, resize_patch, create_dataset_cell_enhanced, create_dataset_cell_ndarray


def raw_to_preprocessed(raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path) -> np.ndarray:
    preprocessed_image = preprocess(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_{image_name}.npy")
    np.save(path, preprocessed_image)
    return preprocessed_image

def save_droplet_images(dataset: np.ndarray, image_name: str, DROPLET_PATH: Path) -> None:
    folder_path = Path(DROPLET_PATH / image_name)

    try:
        os.mkdir(folder_path)
    except FileExistsError as _:
        pass

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            patch = resize_patch(dataset[i][j]['patch'], 100)
            np.save(Path(folder_path / (str(i + 1) + str(j).zfill(4))), patch)

def populate(raw_image_path, image_name, FEATURE_PATH, PREPROCESSED_PATH, DROPLET_PATH) -> None:
    print("Preprocessing raw image...")
    preprocessed_image = raw_to_preprocessed(raw_image_path, image_name, PREPROCESSED_PATH)

    print("Detecting droplets and cells...")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    generate_output(raw_image_path, droplet_feature_path, cell_feature_path, True, "", False)

    print("Creating droplet images...")
    #droplet_images_dataset = create_dataset_from_ndarray([0], ['BF'], preprocessed_image, droplet_feature_path, allFrames=True, allChannels=True)
    droplet_images_dataset = np.array(create_dataset_cell_ndarray(None, ["BF"], preprocessed_image, droplet_feature_path, cell_feature_path, allFrames=True, buffer=-2, suppress_rest=True, suppression_slack=-3, median_filter_preprocess=False))
    save_droplet_images(droplet_images_dataset, image_name, DROPLET_PATH)
    path = Path(FEATURE_PATH/ f"fulldataset_{image_name}.npy")
    np.save(path, droplet_images_dataset)



