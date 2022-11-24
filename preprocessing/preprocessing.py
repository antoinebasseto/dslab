import numpy as np

from skimage.filters import rank
from skimage.morphology import disk
from .raw_image_reader import get_image_as_ndarray

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
