import os
import argparse
from pathlib import Path
from nd2reader import ND2Reader
import cv2 as cv
import pandas as pd

PROJECT_PATH = Path(os.path.dirname(os.getcwd()))
RESULTS_PATH = Path(PROJECT_PATH / "results")
DATA_PATH = Path(PROJECT_PATH / "data")

def visualize_results(file: str, experiment: str):
    assert(file[-4:] == ".nd2")

    nd2_file = ND2Reader(os.path.join(DATA_PATH, file))
    pairing_df = pd.read_csv(os.path.join(RESULTS_PATH, f"pairing_{experiment}.csv"))
    # Expected df is x y coordinates for each of the 8 images
    assert(pairing_df.shape[1] == 16)

    # Create folder where result images are stored
    experiment_result_dir = os.path.join(RESULTS_PATH, experiment)
    try:
        os.mkdir(experiment_result_dir)
    except FileExistsError as _:
        pass

    font = cv.FONT_HERSHEY_PLAIN
    for i in range(8):
        print(f"Generating visualization for image {i}...")

        img = nd2_file.get_frame_2D(x=0, y=0, c=4, t=i, z=0, v=0)
        img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        # Write index number of droplet on each of them
        df = pairing_df.iloc[:, [2*i, 2*i+1]]
        for index, row in df.iterrows():
            cv.putText(img, str(index), (row[0],row[1]), font, 1, (0,255,0), 2, cv.LINE_AA)

        # Save the result image
        cv.imwrite(os.path.join(experiment_result_dir, f"img{i}.png"), img)

def main() -> None:
    parser = argparse.ArgumentParser(description=
                                     """
                                     Generate result images.
                                     """)
    parser.add_argument("file", type=str, help=f"ND2 image file experiment was ran on.")
    parser.add_argument("experiment", type=str, help=f"ID of the experiment.")
    args = parser.parse_args()

    return visualize_results(args.file, args.experiment)

if __name__ == "__main__":
    main()