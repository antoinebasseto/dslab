import numpy as np
import cv2 as cv
import pandas as pd
from raw_image_reader import get_image_as_ndarray

def get_patch(image, center_row, center_col, radius, buffer = 3, suppress_rest = True, suppression_slack = 1, discard_boundaries = True):
    s = image.shape
    # cv.imshow('test', image[0, : ,:] / image[0, : ,:].max())
    # cv.waitKey(0)
    if len(s) == 3:
        # We are in the case where we have channel, image_row and image_col as axes.
        window_dim = radius + buffer
        window_rows = np.asarray((max(0, center_row - window_dim), min(s[1], center_row + window_dim + 1)), dtype = np.int32)
        window_cols = np.asarray((max(0, center_col - window_dim), min(s[2], center_col + window_dim + 1)), dtype = np.int32)
        if ((window_rows[1] - window_rows[0] != 2 * window_dim + 1) or (window_cols[1] - window_cols[0] != 2 * window_dim + 1)) and discard_boundaries:
            return np.zeros((0, 0, 0))
        else:
            ans = np.zeros((s[0], 2 * window_dim + 1, 2 * window_dim + 1), dtype = np.uint16)
            # cv.imshow('test', ans[0, :, :])
            # cv.waitKey(0)
            # cv.imshow('test', image[0, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]])
            # cv.waitKey(0)
            target_rows = window_rows - (center_row - window_dim)
            target_cols = window_cols - (center_col - window_dim)
            # print(target_rows[0] - target_rows[1])
            # print(target_cols[0] - target_cols[1])
            # print(window_rows[0] - window_rows[1])
            # print(window_cols[0] - window_cols[1])
            ans[:, target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[:, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]]
            # cv.imshow('test', ans[0, :, :])
            # cv.waitKey(0)
            if suppress_rest:
                mask = np.zeros(ans.shape[1:3], dtype = np.uint16)
                cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
                ans = ans * mask
                # cv.imshow('test', ans[0, :, :])
                # cv.waitKey(0)
            return ans
    elif len(s) == 2:
        window_dim = radius + buffer
        window_rows = np.asarray((max(0, center_row - window_dim), min(s[0], center_row + window_dim + 1)), dtype = np.int32)
        window_cols = np.asarray((max(0, center_col - window_dim), min(s[1], center_col + window_dim + 1)), dtype = np.int32)
        if ((window_rows[1] - window_rows[0] != 2 * window_dim + 1) or (window_cols[1] - window_cols[0] != 2 * window_dim + 1)) and discard_boundaries:
            return np.zeros((0, 0, 0))
        else:
            ans = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.uint16)
            target_rows = window_rows - (center_row - window_dim)
            target_cols = window_cols - (center_col - window_dim)
            ans[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]]
            if suppress_rest:
                mask = np.zeros(ans.shape, dtype = np.uint16)
                cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
                ans = ans * mask
            return ans
    else:
        assert(False and 'Too many axes in image')

# THIS IS THE PRIMARY FUNCTION
# frames is the list of frames (innteger index) you want to retreive
# channels is the list of channels (strings) you want to retreive. 
#   'BF' is the identifier for the brightfield images
#   'DAPI' is the identifier for the DAPI channel
# image_path is the full path to the nd2 image, suffix included
# droplet_table_path is the path to the csv table ith the detected droplets
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure) 
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet

def create_dataset(frames, channels, image_path, droplet_table_path):
    droplet_table = pd.read_csv(droplet_table_path, index_col = False)
    raw_images = get_image_as_ndarray(frames, channels, image_path, allFrames = False, allChannels = False)
    ans = []
    for i, j in enumerate(frames):
        droplets_in_frame = droplet_table[droplet_table['frame'] == j]
        image_frame = raw_images[i, :, :, :]
        frame_ans = []
        for idx, droplet in droplets_in_frame.iterrows():
            tmp_ans = droplet
            tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'], droplet['radius'], buffer = 3, suppress_rest = True, suppression_slack = 1, discard_boundaries = False)
            frame_ans.append(tmp_ans)
        ans.append(frame_ans)
    return ans