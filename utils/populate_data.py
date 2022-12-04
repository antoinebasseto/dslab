import os
import argparse
from pathlib import Path
import numpy as np

from preprocessing import preprocess
from preprocessing import preprocess_alt_franc
from preprocessing import preprocess_alt_featextr
from droplets_and_cells.droplets_and_cells import generate_output
from droplets_and_cells.droplets_and_cells import generate_output_from_ndarray
from droplet_retriever import create_dataset_from_ndarray, resize_patch

PROJECT_PATH = Path(os.path.dirname(os.getcwd()))
DATA_PATH = Path(PROJECT_PATH / "data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")

def raw_to_preprocessed(raw_image_path: Path, image_name: str) -> np.ndarray:
    preprocessed_image = preprocess(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_{image_name}.npy")
    np.save(path, preprocessed_image)
    return preprocessed_image

def raw_to_preprocessed_alt_franc(raw_image_path: Path, image_name: str) -> np.ndarray:
    preprocessed_image = preprocess_alt_franc(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_drpdtc_{image_name}.npy")
    np.save(path, preprocessed_image)
    return preprocessed_image

def raw_to_preprocessed_featextr(raw_image_path: Path, image_name: str) -> np.ndarray:
    preprocessed_image = preprocess_alt_featextr(raw_image_path)
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_bf_{image_name}.npy")
    np.save(path, preprocessed_image[:, 0, :, :])
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_dapi_{image_name}.npy")
    np.save(path, preprocessed_image[:, 1, :, :])
    return preprocessed_image

def save_droplet_images(dataset: np.ndarray, image_name: str) -> None:
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

def main() -> None:
    parser = argparse.ArgumentParser(description=
                                     """
                                     Generate data from raw.
                                     """)
    parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
    parser.add_argument("-o", action = argparse.BooleanOptionalAction, help=f"Omit droplet dataset generation")
    # parser.add_argument("--alt_franc", action = argparse.BooleanOptionalAction, help=f"Do alternative preprocessing (by Francesco, needed for droplet detection)")
    args = parser.parse_args()

    raw_image_path = Path(RAW_PATH / args.raw_image)
    image_name = args.raw_image[:-4].lower().replace(' ', '_')
    omit_droplet_dataset = args.o
    # alt_franc = args.alt_franc
    if omit_droplet_dataset is None:
        omit_droplet_dataset = False
    # if alt_franc is None:
    #     alt_franc = False

    print("Image preprocessing for droplet detection ...")
    preprocessed_image = raw_to_preprocessed_alt_franc(raw_image_path, image_name)
    print("Detecting droplets and cells...")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    generate_output_from_ndarray(preprocessed_image, droplet_feature_path, cell_feature_path, True, "", False)

    print("Image preprocessing for feature extraction ...")
    preprocessed_image = raw_to_preprocessed_featextr(raw_image_path, image_name)
        

    # print("Detecting droplets and cells...")
    # droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    # cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    # generate_output(raw_image_path, droplet_feature_path, cell_feature_path, True, "droplets_and_cells/raw_images/", True)

    if not omit_droplet_dataset:
        print("Creating droplet images...")
        droplet_images_dataset = create_dataset_from_ndarray([0], preprocessed_image, droplet_feature_path, allFrames=True, buffer = -2, suppress_rest = True, suppression_slack = -3)
        save_droplet_images(droplet_images_dataset, image_name)
    else:
        print("Omitting droplet dataset creation...")

if __name__ == "__main__":
    main()
