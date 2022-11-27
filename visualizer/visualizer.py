import os
import argparse
from pathlib import Path
import cv2 as cv
import pandas as pd

from interactive_explorer import read_in_results

PROJECT_PATH = Path(os.path.dirname(os.getcwd()))
RESULTS_PATH = Path(PROJECT_PATH / "results")
DATA_PATH = Path(PROJECT_PATH / "data")

# imagepath is the path to the nd2 image 
# pairingpath is the path to the csv table with the pairings of droplets between frames
# dropletpath is the path to the csv table with the droplets
# outputpath is the optional path to where to save the images
# experimentid is the optional id of the experiment. Is set to "tmp" if no id is given
# focus_upper is the optional upper left corner of the window of interest
# focus_lower is the optional lower right corner of the window of interest
# ret_traj_table is a flag that indicates whether we also want to save the trajectory_id augmented droplet table.

# def visualize_results(file: str, experiment: str):
def visualize_results(imagepath: str, pairingpath: str, dropletpath: str, outputpath: str, experimentid: str, focus_upper = None, focus_lower = None, ret_traj_table = False):
    assert(imagepath[-4:] == ".nd2")
    assert(pairingpath[-4:] == ".csv")
    assert(dropletpath[-4:] == ".csv")

    experiment = experimentid
    if experiment is None:
        experiment = "tmp"

    # nd2_file = ND2Reader(os.path.join(DATA_PATH, file))
    # pairing_df = pd.read_csv(os.path.join(RESULTS_PATH, f"pairing_{experiment}.csv"))
    # # Expected df is x y coordinates for each of the 8 images
    # assert(pairing_df.shape[1] == 16)


    experiment_result_dir = outputpath
    if experiment_result_dir is None:
        # Create folder where result images are stored
        experiment_result_dir = os.path.join(RESULTS_PATH, experiment)
        try:
            os.mkdir(experiment_result_dir)
        except FileExistsError as _:
            pass

    read_in_results(imagepath, dropletpath, pairingpath, experiment_result_dir, focus_upper_arg = focus_upper, focus_lower_arg = focus_lower, id = experiment, store_traj_table = ret_traj_table)
    # font = cv.FONT_HERSHEY_PLAIN
    # for i in range(8):
    #     print(f"Generating visualization for image {i}...")

    #     img = nd2_file.get_frame_2D(x=0, y=0, c=4, t=i, z=0, v=0)
    #     img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    #     # Write index number of droplet on each of them
    #     df = pairing_df.iloc[:, [2*i, 2*i+1]]
    #     for index, row in df.iterrows():
    #         cv.putText(img, str(index), (row[0],row[1]), font, 1, (0,255,0), 2, cv.LINE_AA)

    #     # Save the result image
    #     cv.imwrite(os.path.join(experiment_result_dir, f"img{i}.png"), img)


# pairing table must have columns "framePrev", "frameNext", "dropletIdPrev", and "dropletIdNext"
# droplet table and pairing table must obviosuly correspond to each other in terms of frames and droplet IDs
def main() -> None:
    # parser = argparse.ArgumentParser(description=
    #                                  """
    #                                  Generate result images.
    #                                  """)
    # parser.add_argument("file", type=str, help=f"ND2 image file experiment was ran on.")
    # parser.add_argument("experiment", type=str, help=f"ID of the experiment.")
    # args = parser.parse_args()

    # return visualize_results(args.file, args.experiment)
    parser = argparse.ArgumentParser(description=
                                     """
                                     Generate result images.
                                     """)
    parser.add_argument("imagepath", type=str, help=f"Path to ND2 image")
    parser.add_argument("dropletpath", type=str, help=f"Path to droplet table")
    parser.add_argument("--pairingpath", type=str, help=f"Path to pairing table. Strictly speaking not required, IFF the droplet table already has a column with 'trajectory_id'. Otherwise, this is very much needed.", required = False)
    parser.add_argument("--experimentid", type=str, help=f"Experiment ID. If not given, \"tmp\" is used automatically.", required = False)
    parser.add_argument("--outputpath", type=str, help=f"Path to output directory if needed to override. Otherwise default output path used.", required = False)
    parser.add_argument("--width", type=int, help=f"Width of window of interest. If not given, whole image is taken.", required = False)
    parser.add_argument("--height", type=int, help=f"Height of window of interest. If not given, whole image is taken.", required = False)
    parser.add_argument("--row", type=int, help=f"Row of upper left corner of window of interest. If not given, is zero.", required = False)
    parser.add_argument("--col", type=int, help=f"Column of upper left corner of window of interest. If not given, is zero.", required = False)
    parser.add_argument("--returntrajectories", type=bool, help=f"Whether or not to return a droplet table with additional trajectory id column. Default is false.", required = False)
    args = parser.parse_args()

    row = args.row
    col = args.col
    width = args.width
    height = args.height
    focus_lower = (height, width)
    ret_traj_table = args.returntrajectories

    if ret_traj_table is None:
        ret_traj_table = False
    if row is None:
        row = 0
    if col is None:
        col = 0
    if width is None:
        focus_lower = None
    if height is None:
        focus_lower = None
    if focus_lower is not None:
        focus_lower = (focus_lower[0] + row, focus_lower[1] + col)
    focus_upper = (row, col)

    return visualize_results(args.imagepath, args.pairingpath, args.dropletpath, args.outputpath, args.experimentid, focus_upper, focus_lower, ret_traj_table)

if __name__ == "__main__":
    main()
