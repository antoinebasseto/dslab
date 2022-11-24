import nd2
import numpy as np
import cv2 as cv
from tqdm import tqdm

# THIS IS THE PRIMARY FUNCTION
# frames is the list of frames (innteger index) you want to retreive
# channels is the list of channels (strings) you want to retreive. 
#   'BF' is the identifier for the brightfield images
#   'DAPI' is the identifier for the DAPI channel
# path_to_image is the full path to the nd2 image, suffix included
# If allFrames == True, frames is ignored and all frames should be processed
# If allChannels == True, all channels will be output (not working atm)
# Returns a 4d numpy array (uint16 so be careful) with the following axes: Frames, Channels, Y and X.

def get_image_as_ndarray(frames, channels, path_to_image, allFrames = True, allChannels = False):



    f = nd2.ND2File(path_to_image)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']
    nr_rows = f.sizes['Y']
    nr_cols = f.sizes['X']

    print('Nr frames: ' + str(nr_frames))
    print('Nr channels: ' + str(nr_channels))
    print('Image dimensions: ' + str(nr_rows) + 'x' + str(nr_cols))


    channel_idx_lookup = {}
    if allChannels:
        channels = []
    print('Channel names:')
    for c in f.metadata.channels:
        channelname = c.channel.name
        if allChannels:
            channels.append(channelname)
        print('\t' + channelname)
        channelidx = c.channel.index
        if (channelname in channels):
            channel_idx_lookup[channelname] = channelidx
    channel_idx_precompute = []
    for ch_name in channels:
        channel_idx_precompute.append(channel_idx_lookup[ch_name])
    if allFrames:
        frames = range(nr_frames)
    fullimage = f.asarray()
    output = (fullimage[frames, :, :, :])[:, channel_idx_precompute, :, :]
    return output


def get_image_as_ndarray_old(frames, channels, path_to_image, allFrames = True, allChannels = False):


    f = nd2.ND2File(path_to_image)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']
    nr_rows = f.sizes['Y']
    nr_cols = f.sizes['X']

    print('Nr frames: ' + str(nr_frames))
    print('Nr channels: ' + str(nr_channels))
    print('Image dimensions: ' + str(nr_rows) + 'x' + str(nr_cols))


    channel_idx_lookup = {}
    if allChannels:
        channels = []
    print('Channel names:')
    for c in f.metadata.channels:
        channelname = c.channel.name
        channels.append(channelname)
        print('\t' + channelname)
        channelidx = c.channel.index
        if (channelname in channels):
            channel_idx_lookup[channelname] = channelidx

    if allFrames:
        frames = range(nr_frames)
    output = np.zeros((len(frames), len(channels), nr_rows, nr_cols), dtype = np.uint16)
    for j, frame_nr in tqdm(enumerate(frames)):
        for i, ch_name in enumerate(channels):
            # print(frame_nr)
            # print(channel_idx_lookup[ch_name])
            output[j, i, :, :] = f.asarray()[frame_nr, channel_idx_lookup[ch_name], :, :]
            # cv.imshow("test", output[j, i, :, :])
            # cv.waitKey(0)
    return output