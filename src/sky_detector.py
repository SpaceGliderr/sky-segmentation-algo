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

    cv2.imwrite("./report/sobel_hori.png", sobel_hori_mag)
    cv2.imwrite("./report/sobel_vert.png", sobel_vert_mag)
    cv2.imwrite("./report/sobel_combined.png", combined_mag)

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
    # print(f"Boundary threshold: {boundary_threshold}")

    # Apply Sobel edge detection
    sobel_img = sobel(preprocessed_img)

    # print(np.mean(sobel_img))

    # Find boundary points
    boundary_points = find_boundary_points(sobel_img, boundary_threshold)

    # Generate mask
    mask = generate_mask(sobel_img, boundary_points)
    # print(set(mask.flatten()))

    # Segment sky
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sky_img = cv2.bitwise_and(img, mask)

    # Plot images
    if display:
        utils.plot_subplots((2, 2), [img, sobel_img, mask, sky_img], ["Original Image", "Sobel Edges", "Sky Segmentation Mask", "Sky Image"])

    return mask, sky_img


def z_score(score, mean, std):
    return (score - mean) / (5 * std)
    # return (score - mean) / std


def split_channels(img, coords):
    """
    Split the image into three channels
    """
    b, g, r = cv2.split(img)

    b_values, g_values, r_values = [], [], []

    for coord in coords:
        b_values.append(b[coord[0], coord[1]])
        g_values.append(g[coord[0], coord[1]])
        r_values.append(r[coord[0], coord[1]])

    return b_values, g_values, r_values


def calculate_stats(channel_pixels):
    """
    Calculate the mean and standard deviation
    """
    b, g, r = channel_pixels

    return (np.mean(b), np.std(b)), (np.mean(g), np.std(g)), (np.mean(r), np.std(r))


def calculate_color_probability(pixel, stats):
    b, g, r = pixel
    # print(f"b: {b}, g: {g}, r: {r}")

    b_stats, g_stats, r_stats = stats

    b_score = z_score(b, b_stats[0], b_stats[1])
    g_score = z_score(g, g_stats[0], g_stats[1])
    r_score = z_score(r, r_stats[0], r_stats[1])

    # print(f"b: {b_score}, g: {g_score}, r: {r_score}")
    
    # probability = b_score + g_score + r_score
    probability = exp(-1 * ( b_score ** 2 + g_score ** 2 + r_score ** 2 ))
    # print(probability)

    # print(probability)

    return probability


def generate_probability_mask(img):
    """
    Generate a mask based on the probability of a pixel being a sky pixel
    """
    # Variables for probability masking
    probability_threshold = config.probability_threshold

    # Apply initial segmentation
    mask, sky = initial_segment_sky(img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Split the initial sky pixels into channels and calculate their stats
    channel_pixels = split_channels(sky, np.transpose(np.where(mask == 255)))
    stats = calculate_stats(channel_pixels)

    # Generate probability mask
    probability_mask = np.zeros(img.shape, dtype=np.uint8)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if calculate_color_probability(img[row, col], stats) > probability_threshold:
                probability_mask[row, col] = 255
    
    return probability_mask, mask


def postprocess(mask, se, iterations=1):
    """
    Postprocess the mask, using morphological operations like closing
    """
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=iterations)


def segment_sky(img, display=False):
    """
    Segment the sky using the probability mask and initial sky segmentation
    """
    # Apply probability mask
    probability_mask, initial_mask = generate_probability_mask(img)

    # Postprocess the mask
    probability_mask = postprocess(probability_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 3)

    # Combine the probability mask with the initial mask
    # print("Initial mask:", initial_mask.shape)
    # print("Probability mask:", probability_mask.shape)
    probability_mask = cv2.cvtColor(probability_mask, cv2.COLOR_BGR2GRAY)
    final_mask = cv2.bitwise_and(initial_mask, probability_mask)

    # Segment sky
    final_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    sky_img = cv2.bitwise_and(img, final_mask)

    # Plot images
    if display:
        utils.plot_subplots((3, 2), [img, initial_mask, probability_mask, final_mask, sky_img], ["Original Image", "Initial Mask", "Probability Mask", "Final Mask", "Sky Image"])

    return final_mask, sky_img
