import cv2 as cv
import numpy as np
from nms import nms
from tqdm import tqdm

def cell_finding (droplet_circles, raw_dapi):
    # tmp_dapi = cv.GaussianBlur(raw_dapi, (5, 5), 0)
    # cannied = cv.Canny(np.uint8(tmp_dapi * 255), 2, 1, L2gradient = True)



    tmp_dapi = cv.GaussianBlur(raw_dapi, (3, 3), 0)
    raw_nms = nms(tmp_dapi)

    droplet_mask = np.zeros(raw_dapi.shape, dtype = np.float32)
    for i in droplet_circles:
        center = (i[0], i[1])
        radius = i[2]
        cv.circle(droplet_mask, np.flip(center), radius - 1, 1.0, -1)
    
    raw_nms = raw_nms * droplet_mask
    thresh = np.quantile(raw_dapi, 0.9)
    raw_nms[raw_dapi <= thresh] = 0.0

    s = raw_nms.shape
    window_dim = 80
    to_iterate = np.nonzero(raw_nms)
    minpoints = 160**2
    for k in tqdm(range(to_iterate[0].size)):
        i = to_iterate[0][k]
        j = to_iterate[1][k]
        if (raw_nms[i, j] != 0.0):
            nms_peak = tmp_dapi[i, j]
            window_x = (max(0, i - window_dim), min(s[0] - 1, i + window_dim))
            window_y = (max(0, j - window_dim), min(s[1] - 1, j + window_dim))
            relevant_points = tmp_dapi[window_x[0]: window_x[1], window_y[0]: window_y[1]]
            minpoints = min(minpoints, np.sum(relevant_points <= np.quantile(relevant_points, 0.8)))
            # print(relevant_points.shape)
            # print(np.sum(relevant_points <= np.quantile(relevant_points, 0.8)))
            relevant_points = relevant_points[relevant_points <= np.quantile(relevant_points, 0.8)]
            sample_sd = np.std(relevant_points)
            sample_mean = np.mean(relevant_points)
            if (nms_peak <= sample_mean + 10 * sample_sd):
                raw_nms[i, j] = 0.0
            elif (nms_peak <= sample_mean + 20 * sample_sd):
                raw_nms[i, j] = 0.1
            elif (nms_peak <= sample_mean + 40 * sample_sd):
                raw_nms[i, j] = 0.2
            elif (nms_peak <= sample_mean + 80 * sample_sd):
                raw_nms[i, j] = 0.3
            elif (nms_peak <= sample_mean + 160 * sample_sd):
                raw_nms[i, j] = 0.4
            elif (nms_peak <= sample_mean + 320 * sample_sd):
                raw_nms[i, j] = 0.5
            elif (nms_peak <= sample_mean + 640 * sample_sd):
                raw_nms[i, j] = 0.6
            elif (nms_peak <= sample_mean + 1280 * sample_sd):
                raw_nms[i, j] = 0.7
            elif (nms_peak <= sample_mean + 2560 * sample_sd):
                raw_nms[i, j] = 0.8
            elif (nms_peak <= sample_mean + 5120 * sample_sd):
                raw_nms[i, j] = 0.9

    # print("Min Points: " + str(minpoints))
    return raw_nms