from find_hough_circle import circle_RANSAC
from find_hough_circle import circle_RANSAC3
import cv2 as cv
import numpy as np
from nms import canny_nms
from tqdm import tqdm

# Transforms float32 images to uint8 images. Assumes the range of the input image is correct
def f32_to_uint8 (img):
    return np.uint8(img * 255)

# Transforms uint8 images to float32 images. Assumes the range of both images are correct
def uint8_to_f32 (img):
    return np.float32(img / 255.0)

# input is float32 greyscale base image
def manual_circle_hough (img, refine):
    
    # For the BF channel, bottom 80% of pixels is pretty much background 
    noise_level = 0.8

    img_denoised = 1.0 - (img - img.min()) / (img.max() - img.min())
    img_denoised = np.clip(img_denoised - np.quantile(img_denoised, noise_level), 0.0, 1.0)
    img_denoised = (img_denoised - img_denoised.min()) / (img_denoised.max() - img_denoised.min())

    # Depending on option, do refinement or not
    detected_circles = []
    if (not refine):
        img_denoised = cv.GaussianBlur(img_denoised, (3, 3), 0)
        preliminary_hough = np.uint16(np.around(cv.HoughCircles(f32_to_uint8(img_denoised), cv.HOUGH_GRADIENT, 1, 25, param1 = 40, param2 = 30, minRadius = 15, maxRadius = 25)))
        for i in tqdm(preliminary_hough[0, :]):
            # Switcheroo here to have row and then col 
            center = (i[1], i[0])
            radius = i[2]
            detected_circles.append((center[0], center[1], radius))
    else:
        img_denoised = cv.GaussianBlur(img_denoised, (3, 3), 0)
        img_edged = canny_nms(img_denoised)
        preliminary_hough = np.uint16(np.around(cv.HoughCircles(f32_to_uint8(img_denoised), cv.HOUGH_GRADIENT, 1, 25, param1 = 40, param2 = 30, minRadius = 15, maxRadius = 35)))
        preliminary_mask = np.zeros(img_denoised.shape, dtype = np.float32)
        for i in tqdm(preliminary_hough[0, :]):
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(preliminary_mask, center, radius, 1.0, -1)

        # Create a mask that can suppress all detected circles
        preliminary_mask_negative = 1.0 - preliminary_mask
        erosion_kernel = np.ones((3, 3), dtype = np.float32)
        for i in tqdm(preliminary_hough[0, :]):
            # Refine each circle
            center = (i[1], i[0])
            radius = i[2]
            # Some indexing madness to extract the relevant patch
            patch_x = (max(int(center[0]) - radius - 5, 0), min(int(center[0]) + radius + 5, img_denoised.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 5, 0), min(int(center[1]) + radius + 5, img_denoised.shape[1] - 1))
            patch_keypoints = img_edged[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            patch_edges = img_edged[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            patch_mask = preliminary_mask_negative[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            # Compute where the circle is estimated to be in the new patch
            center_in_patch = center - np.asarray([max(int(center[0]) - radius - 5, 0), max(int(center[1]) - radius - 5, 0)])
            # This operation results in patch_mask being a mask that suppresses all other circles except for the current one
            cv.circle(patch_mask, np.flip(center_in_patch) , radius, 1.0, -1)
            patch_mask = cv.morphologyEx(patch_mask, cv.MORPH_ERODE, erosion_kernel, iterations = 1)
            patch_mask = patch_mask * 0.9 + 0.1
            # This operation makes sure that we also ignore the center region of where we think the circle is in order to avoid noise from the beads
            cv.circle(patch_mask, np.flip(center_in_patch) , 10, 0.0, -1)
            patch_keypoints = patch_keypoints * patch_mask
            # Get the refined circle estimate
            # refined_circle = circle_RANSAC(patch_keypoints, patch_edges, 15, 35)
            refined_circle = circle_RANSAC3(patch_keypoints, patch_edges, 15, 35)
            refined_circle = (refined_circle[0] + max(int(center[0]) - radius - 5, 0), refined_circle[1] + max(int(center[1]) - radius - 5, 0), int(refined_circle[2]))
            center = (refined_circle[0], refined_circle[1])
            radius = refined_circle[2]
            detected_circles.append((refined_circle[0], refined_circle[1], radius))

    return detected_circles