from math import exp
import cv2
import numpy as np
import utils
import config


def preprocess(img, ksize):
    """
    Completes the preprocessing of the image using Gaussian blur
    """
    blurred = cv2.GaussianBlur(img, ksize, 0)

    if len(blurred.shape) == 3:
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    return blurred


def sobel(img):
    """
    Apply Sobel edge detection to the image
    """
    sobel_hori = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vert = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    sobel_hori_mag = cv2.convertScaleAbs(sobel_hori)
    sobel_vert_mag = cv2.convertScaleAbs(sobel_vert)

    combined_mag = cv2.addWeighted(sobel_hori_mag, 0.5, sobel_vert_mag, 0.5, 0)

    return combined_mag


def find_boundary_points(img, threshold):
    """
    Find the boundary points of the building
    """
    boundary_points = []
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if img[row, col] > threshold:
                boundary_points.append((row, col))
                break
    
    return boundary_points


def generate_mask(img, boundary_points):
    """
    Generate a mask of the sky
    """
    mask = np.full(img.shape, 255, dtype=np.uint8)
    for point in boundary_points:
        mask[point[0]:, point[1]] = 0
    return mask


def calculate_boundary_threshold(img, row_percentage, day_night_thresh, day_night_multi = (1, 1)):
    """
    Calculate the threshold of the image
    """
    row_count = int(img.shape[0] * (row_percentage / 100))
    row_mean = np.mean(img[:row_count, :])

    day_multi, night_multi = day_night_multi

    if row_mean > day_night_thresh:
        return row_mean * day_multi, True
    else:
        return row_mean * night_multi, False


def initial_segment_sky(img, display=False):
    # Variables for image preprocessing
    gaussian_ksize = config.gaussian_ksize

    # Variables for boundary thresholding
    row_percentage = config.row_percentage
    day_night_threshold = config.day_night_threshold
    day_night_multi = config.day_night_multi

    # Preprocess image
    preprocessed_img = preprocess(img, gaussian_ksize)

    # Get boundary threshold
    boundary_threshold, _ = calculate_boundary_threshold(preprocessed_img, row_percentage, day_night_threshold, day_night_multi)

    # Apply Sobel edge detection
    sobel_img = sobel(preprocessed_img)

    # print(np.mean(sobel_img))

    # Find boundary points
    boundary_points = find_boundary_points(sobel_img, boundary_threshold)

    # Generate mask
    mask = generate_mask(sobel_img, boundary_points)

    # Segment sky
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sky_img = cv2.bitwise_and(img, mask)

    # Plot images
    if display:
        utils.plot_subplots((2, 2), [img, sobel_img, mask, sky_img], ["Original Image", "Sobel Edges", "Sky Segmentation Mask", "Sky Image"])

    return mask, sky_img
