import numpy as np
import cv2 as cv
from tqdm.auto import tqdm

from skimage.filters import rank
from skimage.morphology import disk

from preprocessing.raw_image_reader import get_image_as_ndarray
from data_creation.nms import canny_nms

def preprocess(image_path: str) -> np.ndarray:
    image = get_image_as_ndarray([0], ['BF'], image_path, allFrames=True, allChannels=True)

    # For each frame, preprocess channels inplace
    for frame in image:

        # Brightfield preprocessing
        frame[4] = rank.equalize(frame[4], footprint=disk(30))

        # DAPI preprocessing
        # FITC preprocessing
        # TRITC preprocessing
        # Cy5 preprocessing

    return image


def preprocess_alt_franc(image_path: str) -> np.ndarray:
    image = get_image_as_ndarray([0], ['BF', 'DAPI'], image_path, allFrames=True, allChannels=False)

    # For each frame, preprocess channels inplace
    image[:, 0, :, :] = np.uint16(2**16 - (np.int32(image[:, 0, :, :]) + 1))

    kernel = np.ones((3, 3))

    for frame in image:
        # print(np.median(frame[4]))
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)
        pullback_min = bf_chan.min()
        pullback_max = bf_chan.max()
        bf_pullback = (bf_chan - pullback_min) / (pullback_max - pullback_min)
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)

        bf_pullback = np.clip((bf_pullback - np.quantile(bf_pullback, 0.5)) / (1.0 - np.quantile(bf_pullback, 0.5)), 0.0, 1.0)
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)

        equalized = rank.equalize(bf_pullback, footprint=disk(10)) / 255.0
        # cv.imshow('test', equalized[:1000, :1000])
        # cv.waitKey(0)

        bf_pullback = bf_pullback * equalized
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)
        smoothed = cv.GaussianBlur(bf_pullback, (3, 3), 0)
        frame[0, :, :] = np.uint16(smoothed * (2**16 - 1))
    return image

def preprocess_alt_featextr(image_path: str) -> np.ndarray:
    image = get_image_as_ndarray([0], ['BF', 'DAPI'], image_path, allFrames=True, allChannels=False)
    image[:, 0, :, :] = np.uint16(2**16 - (np.int32(image[:, 0, :, :]) + 1))

    # For each frame, preprocess channels inplace
    for frame in tqdm(image):
        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(np.uint8(bf_chan * 255), 2 * 50 + 1) / 255.0)
        img_mediansharpened = np.clip(bf_chan - img_medianblurred, 0.0, 1.0)
        equalized_bf = rank.equalize(img_mediansharpened, footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.5)
        img_mediansharpened[img_mediansharpened > thresh] = bf_chan[img_mediansharpened > thresh]

        frame[0] = cv.GaussianBlur(np.uint16(img_mediansharpened * (2**16 - 1)), (3, 3), 0)
        # cv.imshow('test', bf_chan[:1000, :1000])
        # cv.waitKey(0)
        # cv.imshow('test', frame[0, :1000, :1000])
        # cv.waitKey(0)

        # DAPI preprocessing
        dapi_chan = np.float64(frame[1, :, :])
        dapi_chan_low = np.quantile(dapi_chan, 0.8)
        dapi_chan = np.clip((dapi_chan - dapi_chan_low) / ((2**16 - 1) - dapi_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(np.uint8(dapi_chan * 255), 2 * 20 + 1) / 255.0)
        img_mediansharpened = np.clip(dapi_chan - img_medianblurred, 0.0, 1.0)
        equalized_dapi = rank.equalize(img_mediansharpened, footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.8)
        img_mediansharpened[img_mediansharpened > thresh] = dapi_chan[img_mediansharpened > thresh]

        frame[1] = np.uint16(img_mediansharpened * (2**16 - 1))
        # cv.imshow('test', dapi_chan[:1000, :1000])
        # cv.waitKey(0)
        # cv.imshow('test', frame[1, :1000, :1000])
        # cv.waitKey(0)
    image[np.isnan(image)] = 0.0
    return image