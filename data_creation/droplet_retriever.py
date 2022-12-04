import numpy as np
import cv2 as cv
import pandas as pd
from preprocessing.raw_image_reader import get_image_as_ndarray
from tqdm.auto import tqdm

def create_dataset_from_ndarray(frames, image_ndarray, droplet_table_path, allFrames = False, buffer = 3, suppress_rest = True, suppression_slack = 1, discard_boundaries = False):
    droplet_table = pd.read_csv(droplet_table_path, index_col = False)
    new_frames = frames
    if allFrames:
        new_frames = range(image_ndarray.shape[0])
    ans = []
    for i, j in enumerate(new_frames):
        droplets_in_frame = droplet_table[droplet_table['frame'] == j]
        image_frame = image_ndarray[i, :, :, :]
        frame_ans = []
        for _, droplet in droplets_in_frame.iterrows():
            tmp_ans = droplet
            tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'], droplet['radius'],
                                         buffer, suppress_rest, suppression_slack, discard_boundaries)
            frame_ans.append(tmp_ans)
        ans.append(frame_ans)
    return ans


def get_patch(image, center_row, center_col, radius, buffer=3, suppress_rest=True, suppression_slack=1,
              discard_boundaries=True):
    s = image.shape
    # cv.imshow('test', image[0, : ,:] / image[0, : ,:].max())
    # cv.waitKey(0)
    if len(s) == 3:
        # We are in the case where we have channel, image_row and image_col as axes.
        window_dim = radius + buffer
        window_rows = np.asarray((max(0, center_row - window_dim), min(s[1], center_row + window_dim + 1)),
                                 dtype=np.int32)
        window_cols = np.asarray((max(0, center_col - window_dim), min(s[2], center_col + window_dim + 1)),
                                 dtype=np.int32)
        if ((window_rows[1] - window_rows[0] != 2 * window_dim + 1) or (
                window_cols[1] - window_cols[0] != 2 * window_dim + 1)) and discard_boundaries:
            return np.zeros((0, 0, 0))
        else:
            ans = np.zeros((s[0], 2 * window_dim + 1, 2 * window_dim + 1), dtype=np.uint16)
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
            ans[:, target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[:,
                                                                                     window_rows[0]: window_rows[1],
                                                                                     window_cols[0]: window_cols[1]]
            # cv.imshow('test', ans[0, :, :])
            # cv.waitKey(0)
            if suppress_rest:
                mask = np.zeros(ans.shape[1:3], dtype=np.uint16)
                cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
                ans = ans * mask[None, :, :]
                # cv.imshow('test', ans[0, :, :])
                # cv.waitKey(0)
            return ans
    elif len(s) == 2:
        window_dim = radius + buffer
        window_rows = np.asarray((max(0, center_row - window_dim), min(s[0], center_row + window_dim + 1)),
                                 dtype=np.int32)
        window_cols = np.asarray((max(0, center_col - window_dim), min(s[1], center_col + window_dim + 1)),
                                 dtype=np.int32)
        if ((window_rows[1] - window_rows[0] != 2 * window_dim + 1) or (
                window_cols[1] - window_cols[0] != 2 * window_dim + 1)) and discard_boundaries:
            return np.zeros((0, 0, 0))
        else:
            ans = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype=np.uint16)
            target_rows = window_rows - (center_row - window_dim)
            target_cols = window_cols - (center_col - window_dim)
            ans[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[window_rows[0]: window_rows[1],
                                                                                  window_cols[0]: window_cols[1]]
            if suppress_rest:
                mask = np.zeros(ans.shape, dtype=np.uint16)
                cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
                ans = ans * mask
            return ans
    else:
        assert (False and 'Too many axes in image')


# frames is the list of frames (innteger index) you want to retreive
# channels is the list of channels (strings) you want to retreive.
#   'BF' is the identifier for the brightfield images
#   'DAPI' is the identifier for the DAPI channel
# image_path is the full path to the nd2 image, suffix included
# droplet_table_path is the path to the csv table with the detected droplets
# allFrames indicates whether we want to retreive all frames.
# allChannels indicates whether we want to retreive all channels.
# buffer is the number of extra pixels of slack that we want to have when cutting out the droplets. So the dimension of the returned patches is 2 * (slack + radius) + 1.
# suppress_rest indicates whether we want to suppress the pixels outside of the radius
# suppression_slack is the distance in pixels outside of teh detected radius, that we still consider to be part of the droplet and which we dont suppress.
#   So, everything that is farther away than radius + suppression_slack from the droplet center, gets suppressed
# discard_boundaries indicates whether to not cut out patches that are not 100% included in the image. If set to false, regions of the patch that exceed image boundaries are filled with zeros.
#   If set to true, droplets whose image patches are not contained in the image get a patch of 0x0 pixels.
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure)
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet with according channels

# Example use:
"""
    image_path = 'raw_images/smallMovement1.nd2'
    table_path = 'finished_outputs/smallMovement1_droplets.csv'
    dataset = create_dataset([0], ['BF'], image_path, table_path, allFrames = True, allChannels = True)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0]['patch'].shape)
    # Iterate over frames
    for fr in dataset:
        # Upscale one patch and all its channels at once
        upscaled = resize_patch(fr[0]['patch'], 100)
        print(upscaled.shape)
        # Display channels
        for ch in upscaled:
            cv.imshow("test", ch)
            cv.waitKey(0)
"""


def create_dataset(frames, channels, image_path, droplet_table_path, allFrames=False, allChannels=False, buffer=3,
                   suppress_rest=True, suppression_slack=1, discard_boundaries=False):
    droplet_table = pd.read_csv(droplet_table_path, index_col=False)
    raw_images = get_image_as_ndarray(frames, channels, image_path, allFrames, allChannels)
    # print(raw_images.shape)
    new_frames = frames
    if allFrames:
        new_frames = range(raw_images.shape[0])
    ans = []
    for i, j in enumerate(new_frames):
        droplets_in_frame = droplet_table[droplet_table['frame'] == j]
        image_frame = raw_images[i, :, :, :]
        frame_ans = []
        for idx, droplet in droplets_in_frame.iterrows():
            tmp_ans = droplet
            tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'], droplet['radius'],
                                         buffer, suppress_rest, suppression_slack, discard_boundaries)
            frame_ans.append(tmp_ans)
        ans.append(frame_ans)
    return ans


# Will return a new patch which is the old patch down or upscaled to have height and width 'diameter'. Only square input-patches are allowed.
# Will assume that the last two axes of the input patch are y and x. Channels supported.
# Does both up and downscaling

# Example use:
"""
    image_path = 'raw_images/smallMovement1.nd2'
    table_path = 'finished_outputs/smallMovement1_droplets.csv'
    dataset = create_dataset([0], ['BF'], image_path, table_path, allFrames = True, allChannels = True)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0]['patch'].shape)
    # Iterate over frames
    for fr in dataset:
        # Upscale one patch and all its channels at once
        upscaled = resize_patch(fr[0]['patch'], 100)
        print(upscaled.shape)
        # Display channels
        for ch in upscaled:
            cv.imshow("test", ch)
            cv.waitKey(0)
"""


def resize_patch(patch, diameter):
    s = np.asarray(patch.shape)
    # print(s)
    numaxes = len(s)
    dims = (s[-2], s[-1])
    # print(dims)
    assert (dims[0] == dims[1])
    are_we_upscaling = (dims[0] <= diameter)
    # ans = np.zeros_like(patch)
    if numaxes == 2:
        if are_we_upscaling:
            return cv.resize(patch, (diameter, diameter), interpolation=cv.INTER_CUBIC)
        else:
            return cv.resize(patch, (diameter, diameter), interpolation=cv.INTER_AREA)
    else:
        target_dim = np.copy(s)
        target_dim[-2] = diameter
        target_dim[-1] = diameter
        ans = np.zeros(target_dim, dtype=patch.dtype)
        # print("Hell1o")
        # print(ans.shape)
        for i in range(np.prod(np.asarray(s[0: -2]))):
            idx = np.unravel_index(i, s[0: -2])
            # print("Hell2o")
            # print(idx)
            # print(patch[idx].shape)
            # print(ans[idx].shape)
            if are_we_upscaling:
                ans[idx] = cv.resize(patch[idx], (diameter, diameter), interpolation=cv.INTER_CUBIC)
            else:
                ans[idx] = cv.resize(patch[idx], (diameter, diameter), interpolation=cv.INTER_AREA)
        return ans


# frames is the list of frames (innteger index) you want to retreive
# channels is the list of channels (strings) you want to retreive.
#   'BF' is the identifier for the brightfield images
#   'DAPI' is the identifier for the DAPI channel
# image_path is the full path to the nd2 image, suffix included
# droplet_table_path is the path to the csv table with the detected droplets
# cell_table_path is the path to the csv table with the detected cells / signals
# allFrames indicates whether we want to retreive all frames.
# allChannels indicates whether we want to retreive all channels.
# buffer is the number of extra pixels of slack that we want to have when cutting out the droplets. So the dimension of the returned patches is 2 * (slack + radius) + 1.
# suppress_rest indicates whether we want to suppress the pixels outside of the radius
# suppression_slack is the distance in pixels outside of teh detected radius, that we still consider to be part of the droplet and which we dont suppress.
#   So, everything that is farther away than radius + suppression_slack from the droplet center, gets suppressed
# discard_boundaries indicates whether to not cut out patches that are not 100% included in the image. If set to false, regions of the patch that exceed image boundaries are filled with zeros.
#   If set to true, droplets whose image patches are not contained in the image get a patch of 0x0 pixels.
# median_filter_preprocess is a lag that if set true, will do some basic median sharpening on the image. Typically this preproccing step seems decent at removing a background whose intensity varies.
#   This is not meant to be a good solution but it needs to be done before the droplets are cut out.
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure)
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet with according channels

# Example use:
"""
    image_path = 'raw_images/smallMovement1.nd2'
    droplet_table_path = 'finished_outputs/smallMovement1_droplets.csv'
    cell_table_path = 'finished_outputs/smallMovement1_cells.csv'
    dataset = create_dataset_cell_enhanced([0], ['BF', 'DAPI'], image_path, droplet_table_path, cell_table_path, allFrames = False, allChannels = False, suppression_slack = 0)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0]['patch'].shape)
    for fr in dataset:
        for drplt in fr:
            # This will print all the signal spikes detected in the droplet
            print(drplt['cell_signals'])
            upscaled = resize_patch(drplt['patch'], 300)
            print(upscaled.shape)
            for ch in upscaled:

                cv.imshow("test", (ch - ch.min())/(ch.max() - ch.min()))
                cv.waitKey(0)
"""


def create_dataset_cell_enhanced(frames, channels, image_path, droplet_table_path, cell_table_path, allFrames=False,
                                 allChannels=False, buffer=3, suppress_rest=True, suppression_slack=1,
                                 discard_boundaries=False, omit_patches=False, median_filter_preprocess=False):
    droplet_table = pd.read_csv(droplet_table_path, index_col=False)
    cell_table = pd.read_csv(cell_table_path, index_col=False)
    if not omit_patches:
        raw_images = get_image_as_ndarray(frames, channels, image_path, allFrames, allChannels)
        # print(raw_images.shape)
        new_frames = frames
        if allFrames:
            new_frames = range(raw_images.shape[0])
        ans = []
        for i, j in enumerate(new_frames):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            image_frame = raw_images[i, :, :, :]
            if median_filter_preprocess:
                for ch_idx in range(image_frame.shape[0]):
                    if (allChannels and ch_idx == 4) or ((not allChannels) and channels[ch_idx] == "BF"):
                        ch = np.uint16(2 ** 16 - np.int32(image_frame[ch_idx, :, :]) - 1)
                        normalized_raw_patch = np.float64(ch - ch.min()) / (ch.max() - ch.min())
                        flatfield_normalized_raw_base = np.float64(
                            cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 10 + 1) / 255.0)
                        corrected_raw_patch = np.int64(ch) - (
                                    flatfield_normalized_raw_base * (ch.max() - ch.min()) + ch.min())
                        corrected_raw_patch = np.uint16(np.clip(corrected_raw_patch, 0, 2 ** 16 - 1))
                        image_frame[ch_idx, :, :] = corrected_raw_patch
                    else:
                        ch = image_frame[ch_idx, :, :]
                        normalized_raw_patch = np.float64(ch - ch.min()) / (ch.max() - ch.min())
                        flatfield_normalized_raw_base = np.float64(
                            cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 10 + 1) / 255.0)
                        corrected_raw_patch = np.int64(ch) - (
                                    flatfield_normalized_raw_base * (ch.max() - ch.min()) + ch.min())
                        corrected_raw_patch = np.uint16(np.clip(corrected_raw_patch, 0, 2 ** 16 - 1))
                        image_frame[ch_idx, :, :] = corrected_raw_patch
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'],
                                             droplet['radius'], buffer, suppress_rest, suppression_slack,
                                             discard_boundaries)
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans
    else:
        new_frames = frames
        if allFrames:
            new_frames = range(np.max(droplet_table['frame'].to_numpy(dtype=np.int32)) + 1)
        ans = []
        for i, j in tqdm(enumerate(new_frames)):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans


def create_dataset_cell_ndarray(frames, channels, image, droplet_table_path, cell_table_path, allFrames=False,
                                allChannels=False, buffer=3, suppress_rest=True, suppression_slack=1,
                                discard_boundaries=False, omit_patches=False, median_filter_preprocess=False):
    droplet_table = pd.read_csv(droplet_table_path, index_col=False)
    cell_table = pd.read_csv(cell_table_path, index_col=False)
    new_image = np.zeros((image.shape[0], 2, image.shape[2], image.shape[3]))
    new_image[:, 0, :, :] = image[:, 0, :, :]
    new_image[:, 1, :, :] = image[:, 4, :, :]
    if not omit_patches:
        # raw_images = get_image_as_ndarray(frames, channels, image_path, allFrames, allChannels)
        # print(raw_images.shape)
        new_frames = frames
        if allFrames:
            new_frames = range(new_image.shape[0])
        ans = []
        for i, j in enumerate(new_frames):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            image_frame = new_image[i, :, :, :]
            if median_filter_preprocess:
                for ch_idx in range(image_frame.shape[0]):
                    if (allChannels and ch_idx == 4) or ((not allChannels) and channels[ch_idx] == "BF"):
                        ch = np.uint16(2 ** 16 - np.int32(image_frame[ch_idx, :, :]) - 1)
                        normalized_raw_patch = np.float64(ch - ch.min()) / (ch.max() - ch.min())
                        flatfield_normalized_raw_base = np.float64(
                            cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 10 + 1) / 255.0)
                        corrected_raw_patch = np.int64(ch) - (
                                    flatfield_normalized_raw_base * (ch.max() - ch.min()) + ch.min())
                        corrected_raw_patch = np.uint16(np.clip(corrected_raw_patch, 0, 2 ** 16 - 1))
                        image_frame[ch_idx, :, :] = corrected_raw_patch
                    else:
                        ch = image_frame[ch_idx, :, :]
                        normalized_raw_patch = np.float64(ch - ch.min()) / (ch.max() - ch.min())
                        flatfield_normalized_raw_base = np.float64(
                            cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 10 + 1) / 255.0)
                        corrected_raw_patch = np.int64(ch) - (
                                    flatfield_normalized_raw_base * (ch.max() - ch.min()) + ch.min())
                        corrected_raw_patch = np.uint16(np.clip(corrected_raw_patch, 0, 2 ** 16 - 1))
                        image_frame[ch_idx, :, :] = corrected_raw_patch
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'],
                                             droplet['radius'], buffer, suppress_rest, suppression_slack,
                                             discard_boundaries)
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans
    else:
        new_frames = frames
        if allFrames:
            new_frames = range(np.max(droplet_table['frame'].to_numpy(dtype=np.int32)) + 1)
        ans = []
        for i, j in tqdm(enumerate(new_frames)):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans


# frames is the list of frames (innteger index) you want to retreive
# image is the f-c-h-w dimensional image from which the droplets are going to be cut out. f are the frames, c are the channels, h is the image height and w its width
# droplet_table_path is the path to the csv table with the detected droplets
# cell_table_path is the path to the csv table with the detected cells / signals
# allFrames indicates whether we want to retreive all frames.
# buffer is the number of extra pixels of slack that we want to have when cutting out the droplets. So the dimension of the returned patches is 2 * (slack + radius) + 1.
# suppress_rest indicates whether we want to suppress the pixels outside of the radius
# suppression_slack is the distance in pixels outside of teh detected radius, that we still consider to be part of the droplet and which we dont suppress.
#   So, everything that is farther away than radius + suppression_slack from the droplet center, gets suppressed
# discard_boundaries indicates whether to not cut out patches that are not 100% included in the image. If set to false, regions of the patch that exceed image boundaries are filled with zeros.
#   If set to true, droplets whose image patches are not contained in the image get a patch of 0x0 pixels.
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure)
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet with according channels

def create_dataset_cell_enhanced_from_ndarray(frames, image, droplet_table_path, cell_table_path, allFrames=False,
                                              buffer=3, suppress_rest=True, suppression_slack=1,
                                              discard_boundaries=False, omit_patches=False):
    droplet_table = pd.read_csv(droplet_table_path, index_col=False)
    cell_table = pd.read_csv(cell_table_path, index_col=False)
    if not omit_patches:
        raw_images = image
        # print(raw_images.shape)
        new_frames = frames
        if allFrames:
            new_frames = range(raw_images.shape[0])
        ans = []
        for i, j in enumerate(new_frames):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            image_frame = raw_images[i, :, :, :]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'],
                                             droplet['radius'], buffer, suppress_rest, suppression_slack,
                                             discard_boundaries)
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans
    else:
        new_frames = frames
        if allFrames:
            new_frames = range(np.max(droplet_table['frame'].to_numpy(dtype=np.int32)) + 1)
        ans = []
        for i, j in tqdm(enumerate(new_frames)):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[
                    ['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans


# frames is the list of frames (innteger index) you want to retreive
# image is the f-c-h-w dimensional image from which the droplets are going to be cut out. f are the frames, c are the channels, h is the image height and w its width
# droplet_table_path is the path to the csv table with the detected droplets
# cell_table_path is the path to the csv table with the detected cells / signals
# allFrames indicates whether we want to retreive all frames.
# buffer is the number of extra pixels of slack that we want to have when cutting out the droplets. So the dimension of the returned patches is 2 * (slack + radius) + 1.
# suppress_rest indicates whether we want to suppress the pixels outside of the radius
# suppression_slack is the distance in pixels outside of teh detected radius, that we still consider to be part of the droplet and which we dont suppress.
#   So, everything that is farther away than radius + suppression_slack from the droplet center, gets suppressed
# discard_boundaries indicates whether to not cut out patches that are not 100% included in the image. If set to false, regions of the patch that exceed image boundaries are filled with zeros.
#   If set to true, droplets whose image patches are not contained in the image get a patch of 0x0 pixels.
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure) 
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet with according channels

def create_dataset_cell_enhanced_from_ndarray(frames, image, droplet_table_path, cell_table_path, allFrames = False, buffer = 3, suppress_rest = True, suppression_slack = 1, discard_boundaries = False, omit_patches = False):
    droplet_table = pd.read_csv(droplet_table_path, index_col = False)
    cell_table = pd.read_csv(cell_table_path, index_col = False)
    if not omit_patches:
        raw_images = image
        # print(raw_images.shape)
        new_frames = frames
        if allFrames:
            new_frames = range(raw_images.shape[0])
        ans = []
        for i, j in enumerate(new_frames):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            image_frame = raw_images[i, :, :, :]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['patch'] = get_patch(image_frame, droplet['center_row'], droplet['center_col'], droplet['radius'], buffer, suppress_rest, suppression_slack, discard_boundaries)
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans
    else:
        new_frames = frames
        if allFrames:
            new_frames = range(np.max(droplet_table['frame'].to_numpy(dtype = np.int32)) + 1)
        ans = []
        for i, j in tqdm(enumerate(new_frames)):
            droplets_in_frame = droplet_table[droplet_table['frame'] == j]
            cells_in_frame = cell_table[cell_table['frame'] == j]
            frame_ans = []
            for idx, droplet in droplets_in_frame.iterrows():
                droplet_id = droplet['droplet_id']
                cells_in_frame_in_droplet = (cells_in_frame[cells_in_frame['droplet_id'] == droplet_id])[['cell_id', 'center_row', 'center_col', 'intensity_score', 'persistence_score']]
                tmp_ans = droplet
                tmp_ans['cell_signals'] = cells_in_frame_in_droplet
                frame_ans.append(tmp_ans)
            ans.append(frame_ans)
        return ans