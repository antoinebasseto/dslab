import os
import argparse
from pathlib import Path
import numpy as np
from preprocessing.preprocessing import preprocess
from preprocessing.preprocessing import preprocess_alt_franc
from preprocessing.preprocessing import preprocess_alt_featextr
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

def populate(raw_image_path, image_name, FEATURE_PATH, PREPROCESSED_PATH, DROPLET_PATH, omit_droplet_dataset = False) -> None:
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



