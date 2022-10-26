import cv2 as cv
import numpy as np
from nms import nms
from nms import nms4
from nms import canny_nms
from nms import grad_nms
from tqdm.auto import tqdm
import math

import sys

sys.path.insert(0, '/Users/francescodadalt/Documents/ETH/Programmieren/Git Repositiories/dslab/utils')
from droplet_retreiver import resize_patch

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

def masked_dilate(image, mask, kernel):
    return cv.morphologyEx(image, cv.MORPH_DILATE, kernel, iterations = 1) * mask

def create_custom_kernel (rad):
    ans = np.zeros((2 * rad + 1, 2 * rad + 1), dtype = np.float32)
    for i in range(2 * rad + 1):
        for j in range(2 * rad + 1):
            dist = np.linalg.norm(np.asarray([i, j]) - np.asarray([rad, rad]))
            ans[i, j] = math.exp(-dist/((rad / 2)))
    ans = ans / np.sum(ans)
    return ans

def cell_finding2 (droplet_circles, raw_dapi, raw_bf):
    # tmp_dapi = cv.GaussianBlur(raw_dapi, (3, 3), 0)
    # tmp_dapi = np.float32(tmp_dapi * 2.0**(-16))
    # tmp_bf = cv.GaussianBlur(raw_bf, (3, 3), 0)
    # tmp_bf = np.float32(tmp_bf * 2.0**(-16))
    s = raw_dapi.shape

    kernel = np.ones((3, 3))

    ans = np.zeros((3, s[0], s[1]), dtype = np.float32)

    # droplet_mask = np.zeros(s, dtype = np.float32)

    # counter = 0
    for i in tqdm(droplet_circles):
        # print(counter)
        # counter = counter + 1
        center = np.asarray((i[0], i[1]), dtype = np.int32)
        radius = i[2]
        # if (center[0] == 2403 and center[1] == 788):
        # if (center[0] == 4121 and center[1] == 1855):
        if True:
        # if counter == 56:
            window_dim = radius + 10
            window_rows = np.asarray((max(0, center[0] - window_dim), min(s[0], center[0] + window_dim + 1)), dtype = np.int32)
            window_cols = np.asarray((max(0, center[1] - window_dim), min(s[1], center[1] + window_dim + 1)), dtype = np.int32)
            target_rows = window_rows - (center[0] - window_dim)
            target_cols = window_cols - (center[1] - window_dim)
            # patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.uint16)
            local_mask = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
            cv.circle(local_mask, np.asarray([window_dim, window_dim]), radius, 1.0, -1)
            raw_patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
            raw_patch[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = raw_dapi[window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] * 2.0**(-16)
            signal_mask = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
            signal_mask[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = 1.0
            raw_bf_patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
            raw_bf_patch[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = raw_bf[window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] * 2.0**(-16)
            median_of_relevant_pixels = np.median(raw_patch[signal_mask * local_mask == 1.0])
            raw_patch[signal_mask == 0.0] = median_of_relevant_pixels
            
            normalized_raw_patch = (raw_patch - raw_patch.min()) / (raw_patch.max() - raw_patch.min())

            flatfield_normalized_raw_base = np.float32(cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 5 + 1) / 255.0)

            corrected_raw_patch = raw_patch - (flatfield_normalized_raw_base * (raw_patch.max() - raw_patch.min()) + raw_patch.min())
            corrected_raw_patch = corrected_raw_patch - corrected_raw_patch.min()

            corrected_patch = cv.GaussianBlur(corrected_raw_patch, (3, 3), 0)


            # raw_patch_eroded = cv.morphologyEx(raw_patch * 1.0, cv.MORPH_ERODE, kernel, iterations = 1)

            relevant_pixels_list = corrected_patch[local_mask == 1.0]

            background_threshold = np.quantile(relevant_pixels_list, 0.8)

            guaranteed_background_mask = (corrected_patch <= background_threshold) * 1.0
            # guaranteed_background_mask_dilated = cv.morphologyEx(guaranteed_background_mask, cv.MORPH_DILATE, kernel, iterations = 1)
            guaranteed_background_mask_dilated = np.copy(guaranteed_background_mask)

            # patch_canny = cv.Canny(np.uint8(raw_patch * 255), 0, 5, L2gradient = True)
            patch_nms = nms(corrected_patch) * local_mask
            # patch_grad_nms = grad_nms(corrected_patch) * local_mask
            # patch_raw_nms = nms(corrected_raw_patch) * local_mask
            # patch_nms4 = nms4(raw_patch) * local_mask
            # patch_canny_nms = canny_nms(patch)
            inversepatch_nms = canny_nms(-corrected_raw_patch)
            for i in range(20):
                guaranteed_background_mask_dilated = masked_dilate(guaranteed_background_mask_dilated, inversepatch_nms, kernel)
            guaranteed_background_mask_final = np.logical_or(guaranteed_background_mask, guaranteed_background_mask_dilated)
            guaranteed_background_mask_final = cv.morphologyEx(guaranteed_background_mask_final * 1.0, cv.MORPH_CLOSE, kernel, iterations = 1)

            for i in range(5):
                nullhypothesis = corrected_raw_patch[guaranteed_background_mask_final == 1.0]
                hypo_mean = np.mean(nullhypothesis)
                hypo_std = np.std(nullhypothesis)
                guaranteed_background_mask_final[corrected_patch <= hypo_mean + 3 * hypo_std] = 1.0
            peaks_nms = patch_nms * (1.0 - guaranteed_background_mask_final)           
            peaks = np.zeros((3, peaks_nms.shape[0], peaks_nms.shape[1]), dtype = np.float32)
            peaks[0, :, :] = peaks_nms
            peaks[1, :, :] = peaks_nms
            noise_std = np.std(corrected_raw_patch[guaranteed_background_mask_final == 1.0])
            # print(noise_std)
            noise_mean = np.mean(corrected_raw_patch[guaranteed_background_mask_final == 1.0])
            # print(noise_mean)
            # print(corrected_patch[peaks[0, :, :] == 1.0].shape)
            # peaks[1, :, :] = (1.0 - np.exp(-(corrected_patch - noise_mean) / (10 * noise_std))) * peaks[0, :, :]
            peaks[1, :, :] = (corrected_patch - noise_mean) / (10.0 * noise_std) * peaks[0, :, :]


            peaks_detected_idxs = np.argwhere(peaks_nms != 0.0)
            if peaks_detected_idxs.shape[0] > 0:
                guaranteed_background_idxs = np.argwhere(guaranteed_background_mask_final == 1.0)
                # print('')
                # print('')
                # print(guaranteed_background_idxs.shape)
                distance_vecs = guaranteed_background_idxs[:, None, :] - peaks_detected_idxs[None, :, :]
                # print(distance_vecs.shape)
                distances = np.linalg.norm(distance_vecs, axis = 2)
                distances.partition(10, axis = 0)
                mean_distances = np.mean(distances[: 10, :], axis = 0)
                # distances = np.min(np.linalg.norm(distance_vecs, axis = 2), axis = 0)
                # print(mean_distances.shape)
                # print(mean_distances)


                # presistency_score = (1.0 - np.exp(-(mean_distances) / 1.5))
                presistency_score = mean_distances


                # print(presistency_score)
                # print((np.repeat(2, mean_distances.size), peaks_detected_idxs[:, 0], peaks_detected_idxs[:, 1]))
                peaks[(np.repeat(2, mean_distances.size), peaks_detected_idxs[:, 0], peaks_detected_idxs[:, 1])] = presistency_score

            ans[:, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] = ans[:, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] + peaks[:, target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]]

            to_display7 = raw_bf_patch
            to_display = np.asarray([patch_nms * local_mask, inversepatch_nms * local_mask, guaranteed_background_mask * local_mask])
            to_display2 = np.asarray([patch_nms * local_mask, inversepatch_nms * local_mask, guaranteed_background_mask_dilated * local_mask])
            to_display3 = np.asarray([peaks[0, :, :] , inversepatch_nms * local_mask, guaranteed_background_mask_final * local_mask])
            to_display5 = (corrected_patch - corrected_patch.min()) / (corrected_patch.max() - corrected_patch.min())


            # cv.imshow('test',np.float32(np.transpose(resize_patch(to_display7, 500))))
            # cv.waitKey(0)
            # cv.imshow('test', np.float32(np.transpose(resize_patch(to_display5, 500))))
            # cv.waitKey(0)
            # cv.imshow('test', np.float32(np.transpose(resize_patch(to_display, 500))))
            # cv.waitKey(0)
            # cv.imshow('test', np.float32(np.transpose(resize_patch(to_display2, 500))))
            # cv.waitKey(0)
            # cv.imshow('test', np.float32(np.transpose(resize_patch(to_display3, 500))))
            # cv.waitKey(0)
    # cv.imshow('test',np.float32(np.transpose(resize_patch(tmp_bf, 500))))
    # cv.waitKey(0)
    # cv.imshow('test',np.float32(np.transpose(resize_patch(ans, 500))))
    # cv.waitKey(0)
    return ans