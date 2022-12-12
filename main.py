from data_creation.populate_data import populate
from tracking.hierarchical_linking import linking
from tracking.hierarchical_linking import vote_based_linking
# from tracking.hierarchical_linking import droplet_linking_feature_based_voting
import argparse
from pathlib import Path
import numpy as np
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
EXPERIMENT_PATH = Path(PROJECT_PATH / "experiments")

parser = argparse.ArgumentParser(description=
                                 """
                                 Generate data from raw.
                                 """)
parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
parser.add_argument("-g", action='store_true', help=f"Generate Embeddings")
parser.add_argument("-s",action='store_true', help=f"Skip dataset generation")
parser.add_argument("-e",action='store_true', help=f"Use embeddings")
args = parser.parse_args()

raw_image_path = Path(RAW_PATH / args.raw_image)
image_name = args.raw_image[:-4].lower().replace(' ', '_')

skip_dataset = args.s
use_embeddings = args.e
if not args.g:
    EXPERIMENT_PATH = None

if not skip_dataset:
    print("----Creating Preprocessed Dataset----")
    populate(raw_image_path,image_name,FEATURE_PATH,PREPROCESSED_PATH,DROPLET_PATH, EXPERIMENT_PATH)
print("----Applying Tracking Methods----")
print("feature path", FEATURE_PATH)
vote_based_linking(image_name,FEATURE_PATH,RESULT_PATH, use_embeddings=True)
# linking(image_name,FEATURE_PATH,RESULT_PATH, use_embeddings)
# droplet_linking_feature_based_voting (droplet_table_path, cell_table_path, bf_image_path, dapi_image_path,tracking_table_path)

print("--- %s seconds ---" % (time.time() - start_time))