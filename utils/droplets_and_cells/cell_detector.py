import cv2 as cv
import numpy as np

from cell_finding import cell_finding

def cell_detector(dapi_channel, detected_droplets):

    detected_cells = cell_finding(detected_droplets, dapi_channel)


    print("\n\nCells at intensity level. High to low intensity:")
    print(np.sum(detected_cells == 1.0))
    print(np.sum(detected_cells == 0.9))
    print(np.sum(detected_cells == 0.8))
    print(np.sum(detected_cells == 0.7))
    print(np.sum(detected_cells == 0.6))
    print(np.sum(detected_cells == 0.5))
    print(np.sum(detected_cells == 0.4))
    print(np.sum(detected_cells == 0.3))
    print(np.sum(detected_cells == 0.2))
    print(np.sum(detected_cells == 0.1))
    print("\n")

    return detected_cells