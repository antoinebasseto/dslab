from data_creation.populate_data import populate
from tracking.hierarchical_linking import linking
from tracking.hierarchical_linking import droplet_linking_feature_based_voting
import argparse
from pathlib import Path
import os

import time
start_time = time.time()



PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")
RESULT_PATH = Path(DATA_PATH/ "05_results")

parser = argparse.ArgumentParser(description=
                                 """
                                 Generate data from raw.
                                 """)
parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
args = parser.parse_args()

raw_image_path = Path(RAW_PATH / args.raw_image)
image_name = args.raw_image[:-4].lower().replace(' ', '_')

print("----Creating Preprocessed Dataset----")
populate(raw_image_path,image_name,FEATURE_PATH,PREPROCESSED_PATH,DROPLET_PATH)

print("----Applying Tracking Methods----")
#linking(image_name,FEATURE_PATH,RESULT_PATH)
droplet_linking_feature_based_voting(image_name,FEATURE_PATH,RESULT_PATH)

print("--- %s seconds ---" % (time.time() - start_time))