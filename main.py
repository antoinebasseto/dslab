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

parser = argparse.ArgumentParser(description=
                                 """
                                 Generate data from raw.
                                 """)
parser.add_argument("raw_image", type=str, help=f"ND2 image file name")
parser.add_argument("-s",action='store_true', help=f"Skip dataset generation")
parser.add_argument("-e",action='store_true', help=f"Use embeddings")
parser.add_argument('--similarity_weight', type=float, help='a number between 0 and 1. 1 means we only link droplets if they have exactly the same features. 0 means we allow linking of droplets even if they look very different. Default is 0.5')
parser.add_argument('--vicinity_weight', type=float, help='a number between 0 and 1. 1 means we prefer only spacial vicinity. 0 means we consider spacial vicinity and visual similarity. Default is 0.0')
parser.add_argument('--max_dist', type=int, help='a positive integer that indicates the maximal distance in pixels that a droplet can move between frames. Default is 250')
parser.add_argument('--movement_variability', type=float, help='a positive floating point number (typically close to 1). Close to 0 means we prefer few droplets that move a lot. 1 means we are neutral. Greater than 1 means we prefer if many droplets move a bit. Default is 1 and values close to 1 seem to work best.')
args = parser.parse_args()

raw_image_path = Path(RAW_PATH / args.raw_image)
image_name = args.raw_image[:-4].lower().replace(' ', '_')

skip_dataset = args.s
use_embeddings = args.e
if not skip_dataset:
    print("----Creating Preprocessed Dataset----")
    populate(raw_image_path,image_name,FEATURE_PATH,PREPROCESSED_PATH,DROPLET_PATH)
similarity_weight = args.similarity_weight
max_dist = args.max_dist
vicinity_weight = args.vicinity_weight
movement_variability = args.movement_variability
if similarity_weight is None:
    similarity_weight = 0.5
similarity_weight = min(max(similarity_weight, 0.1), 0.9)
if max_dist is None:
    max_dist = 250
max_dist = max(max_dist, 20)
if vicinity_weight is None:
    vicinity_weight = 0.0
vicinity_weight = min(max(vicinity_weight, 0.0), 1.0)
if movement_variability is None:
    movement_variability = 1.0
movement_variability = max(movement_variability, 0.1)
print("----Applying Tracking Methods----")
vote_based_linking(image_name,FEATURE_PATH,RESULT_PATH, use_embeddings, similarity_weight, max_dist, vicinity_weight, movement_variability)
# linking(image_name,FEATURE_PATH,RESULT_PATH, use_embeddings)
# droplet_linking_feature_based_voting (droplet_table_path, cell_table_path, bf_image_path, dapi_image_path,tracking_table_path)

print("--- %s seconds ---" % (time.time() - start_time))