import os
import argparse
from pathlib import Path

from visualizer.interactive_explorer import select_trajectories

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")
RESULT_PATH = Path(DATA_PATH/ "05_results")

def visualize_results(raw_image_filename: str, results_filename: str):
    assert(raw_image_filename[-4:] == '.nd2')
    assert(results_filename[-4:] == '.csv')

    raw_image_path = Path(RAW_PATH / raw_image_filename)
    results_path = Path(RESULT_PATH / results_filename)
    image_name = raw_image_filename[:-4].lower().replace(' ', '_')

    select_trajectories(raw_image_path, results_path, image_name)

def main() -> None:
    parser = argparse.ArgumentParser(description=
                                 """
                                 Generate data from raw.
                                 """)
    parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
    parser.add_argument("results", type=str, help=f"results file name")
    args = parser.parse_args()

    visualize_results(args.raw_image, args.results)

if __name__ == "__main__":
    main()
