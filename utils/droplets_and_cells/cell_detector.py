import cv2 as cv
import numpy as np

from cell_finding import cell_finding
from cell_finding import cell_finding2

def cell_detector(dapi_channel, bf_channel, detected_droplets):

    cell_detection_scores = cell_finding2(detected_droplets, dapi_channel, bf_channel)
    detected_cells_mask = cell_detection_scores[0, :, :]
    intensity_scores = cell_detection_scores[1, :, :]
    persistency_scores = cell_detection_scores[2, :, :]


    print("\n\nSignals at intensity level. High to low intensity:")
    print(np.sum(intensity_scores >= 0.9))
    print(np.sum(intensity_scores >= 0.8) - np.sum(intensity_scores >= 0.9))
    print(np.sum(intensity_scores >= 0.7) - np.sum(intensity_scores >= 0.8))
    print(np.sum(intensity_scores >= 0.6) - np.sum(intensity_scores >= 0.7))
    print(np.sum(intensity_scores >= 0.5) - np.sum(intensity_scores >= 0.6))
    print(np.sum(intensity_scores >= 0.4) - np.sum(intensity_scores >= 0.5))
    print(np.sum(intensity_scores >= 0.3) - np.sum(intensity_scores >= 0.4))
    print(np.sum(intensity_scores >= 0.2) - np.sum(intensity_scores >= 0.3))
    print(np.sum(intensity_scores >= 0.1) - np.sum(intensity_scores >= 0.2))
    print(np.sum(intensity_scores > 0.0) - np.sum(intensity_scores >= 0.1))
    print("\n\nSignals at persistency level. High to low intensity:")
    print(np.sum(persistency_scores >= 0.9))
    print(np.sum(persistency_scores >= 0.8) - np.sum(persistency_scores >= 0.9))
    print(np.sum(persistency_scores >= 0.7) - np.sum(persistency_scores >= 0.8))
    print(np.sum(persistency_scores >= 0.6) - np.sum(persistency_scores >= 0.7))
    print(np.sum(persistency_scores >= 0.5) - np.sum(persistency_scores >= 0.6))
    print(np.sum(persistency_scores >= 0.4) - np.sum(persistency_scores >= 0.5))
    print(np.sum(persistency_scores >= 0.3) - np.sum(persistency_scores >= 0.4))
    print(np.sum(persistency_scores >= 0.2) - np.sum(persistency_scores >= 0.3))
    print(np.sum(persistency_scores >= 0.1) - np.sum(persistency_scores >= 0.2))
    print(np.sum(persistency_scores > 0.0) - np.sum(persistency_scores >= 0.1))
    print("\n\nTotal signals detected:")
    print(np.sum(detected_cells_mask > 0.0))
    print("\n\nAverage signals per droplet:")
    print(np.sum(detected_cells_mask > 0.0) / len(detected_droplets))
    print('')
    print('')

    # assert(False)

    return detected_cells_mask, intensity_scores, persistency_scores