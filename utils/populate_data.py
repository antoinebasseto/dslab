import os
import argparse
from pathlib import Path
import numpy as np

from preprocessing import preprocess
from droplets_and_cells.droplets_and_cells import generate_output
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

def save_droplet_images(dataset: np.ndarray, image_name: str) -> None:
    folder_path = Path(DROPLET_PATH / image_name)

    try:
        os.mkdir(folder_path)
    except FileExistsError as _:
        pass

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            patch = resize_patch(dataset[i][j]['patch'], 100)
            np.save(Path(folder_path / (str(i + 1) + str(j).zfill(4))), patch)

def main() -> None:
    parser = argparse.ArgumentParser(description=
                                     """
                                     Generate data from raw.
                                     """)
    parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
    args = parser.parse_args()

    raw_image_path = Path(RAW_PATH / args.raw_image)
    image_name = args.raw_image[:-4].lower().replace(' ', '_')

    print("Preprocessing raw image...")
    preprocessed_image = raw_to_preprocessed(raw_image_path, image_name)

    print("Detecting droplets and cells...")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    generate_output(raw_image_path, droplet_feature_path, cell_feature_path, True, "", False)

    print("Creating droplet images...")
    droplet_images_dataset = create_dataset_from_ndarray([0], ['BF'], preprocessed_image, droplet_feature_path, allFrames=True, allChannels=True)
    save_droplet_images(droplet_images_dataset, image_name)

if __name__ == "__main__":
    main()
