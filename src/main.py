from math import exp
from os import walk
import cv2
from cv2 import split
# from cv2 import exp
from matplotlib import pyplot as plt
import numpy as np
import utils
import sky_detector as detector
import performance
import config


if __name__ == '__main__':
    curr_sample_dir_name = "training-samples" # Set the target directory. Needs to be a variable since the directory can be changed to training samples or testing samples.

    sample_dir_names = next(walk(f'./{curr_sample_dir_name}'), (None, None, []))[1] # Gets all the subdirectories in the target sample directory.
    # print(sample_dir_names)

    curr_imgs_dir_name = sample_dir_names[3] # Can be easily changed into a for loop in the future. Have to make it flexible.

    imgs_filenames = next(walk(f'./{curr_sample_dir_name}/{curr_imgs_dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    imgs_filenames.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1]))) # Sorts image filenames in ascending order

    # First index is selected for now (second image in 1093 directory) because it provides a good separation between sky and building
    curr_img_filename = imgs_filenames[1] # Can be easily changed into a for loop in the future. Have to make it flexible.
    # print(curr_img_filename)
    # curr_img_filename = "20140503_162404.jpg"

    # Read selected image
    curr_img_path = f'./{curr_sample_dir_name}/{curr_imgs_dir_name}/{curr_img_filename}'
    # print(curr_img_path)
    curr_img_path = "./training-samples/1093/20140419_002403.jpg"
    ori_img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)

    # with open(curr_img_path, 'rb') as f:
    #     check_chars = f.read()[-2:]

    # if check_chars != b'\xff\xd9':
    #     print("Not complete image")
    # else:
    #     ori_img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)

    # performance.get_corrupted()

    # Performance test
    # print("Initial Segment Sky")
    performance.evaluate(detector.initial_segment_sky)
    # performance.evaluate_results()

    # print("Segment Sky")
    # performance.evaluate(detector.segment_sky)

    # Segment sky
    # detector.segment_sky(ori_img, True)
    # mask, sky = detector.initial_segment_sky(ori_img)
    # cv2.imwrite("./report/mask.jpg", mask)
    # cv2.imwrite("./report/sky.jpg", sky)
    # probability_map = generate_probability_map(ori_img)

    # utils.plot_subplots((2, 2), [ori_img, sky, mask, probability_map], ["Original Image", "Initial Sky Segment Algo", "Initial Mask", "Probability Algo"])

    # Research on color averages
    # mask, sky = detector.initial_segment_sky(ori_img)

    # # sky_pixels = np.where(mask == 255)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # # print(mask.shape)
    # # print(np.transpose(np.where(mask == 255)))

    # # b, g, r = cv2.split(sky)
    # channel_pixels = split_channels(sky, np.transpose(np.where(mask == 255)))
    # # print(b)
    # # print(g)
    # # print(r)

    # # Calculate the mean and standard deviation
    # # b_stats, g_stats, r_stats = calculate_stats(channel_pixels)
    # stats = calculate_stats(channel_pixels)

    # # Calculate the color score
    # calculate_color_probability(ori_img[151][332], stats)

    # b, g, r = ori_img[0][0]

    # print(b)
    # print(g)
    # print(r)

    # print(f"b: {b_stats[0]} g: {g_stats[0]} r: {r_stats[0]}")
    # print(f"b: {b_stats[1]} g: {g_stats[1]} r: {r_stats[1]}")
    # print(f"b: {z_score(30, b_stats[0], b_stats[1])} g: {z_score(40, g_stats[0], g_stats[1])} r: {z_score(20, r_stats[0], r_stats[1])}")


    # b, g, r = cv2.split(ori_img)
    # cv2.imshow("Blue Channel", b)
    # cv2.imshow("Green Channel", g)
    # cv2.imshow("Red Channel", r)
    # cv2.imshow("Sobel", detector.sobel(r))
    # cv2.waitKey()

    # Using the probability model to segment sky
    # 1. Obtain the color center of the sky pixels

    # print(mask.shape)

    # Performance test
    # performance.evaluate(detector.initial_segment_sky)


    # =====================================================================================

    # Variables for image preprocessing
    # gaussian_ksize = (15, 15)

    # # Preprocess image
    # preprocessed_img = detector.preprocess(ori_img, gaussian_ksize)

    # # Variables for boundary thresholding
    # row_percentage = 15
    # day_night_threshold = 70
    # day_night_multi = (0.1, 0.3)

    # # Get boundary threshold
    # boundary_threshold = detector.calculate_boundary_threshold(preprocessed_img, 15, 70, (0.1, 0.25))
    # print(f"Boundary threshold: {boundary_threshold}")

    # # Apply Sobel edge detection
    # sobel_img = detector.sobel(preprocessed_img)

    # # Find boundary points
    # boundary_points = detector.find_boundary_points(sobel_img, boundary_threshold)

    # # Generate mask
    # mask = detector.generate_mask(sobel_img, boundary_points)

    # # Segment sky
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # sky_img = cv2.bitwise_and(ori_img, mask)

    # # Plot images
    # utils.plot_subplots((2, 2), [ori_img, sobel_img, mask, sky_img], ["Original Image", "Sobel Edges", "Sky Segmentation Mask", "Sky Image"])

    # =====================================================================================

    # blur = cv2.GaussianBlur(ori_img,(15,15),0)
    # grayscale = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # brighter_img = grayscale + 50

    # utils.plot_subplots((2, 2), [ori_img, blur, grayscale, brighter_img], ["Original", "Blurred", "Grayscale", "Brighter"])

    # cv2.imshow("Blur", blur)
    # cv2.imshow("Grayscale Image", grayscale)
    # cv2.imshow("Brighter Image", brighter_img)
    # cv2.waitKey(0)

    # Test with Channels
    # ori_img_color = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
    # b, g, r = cv2.split(ori_img_color)
    # grayscale = cv2.cvtColor(ori_img_color, cv2.COLOR_BGR2GRAY)
    # hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV) # HSV
    # h, s, v1 = cv2.split(hsv_img) # v1 is the gray scale image of HSV
    # # cv2.imshow("Grayscale Image", grayscale)
    # # cv2.imshow("HSV Gray Image", v1)
    # # cv2.waitKey(0)
    # # utils.plot_subplots((1, 3), [ori_img, grayscale, v1], ["Original", "Grayscale", "HSV Gray"])
    # ori_img = cv2.GaussianBlur(ori_img_color,(15,15),0)

    # Display original image
    # cv2.imshow("Original Sky Image", ori_img)

    # Explore the different color maps
    # hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV) # HSV
    # lab_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2LAB) # LAB
    # yuv_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV) # YUV
    # hsl_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HLS) # HSL
    # b, g, r = cv2.split(ori_img) # BGR

    # cv2.imshow("Blue Channel", b)
    # cv2.imshow("Green Channel", g)
    # cv2.waitKey(0)

    # plt.figure()
    # plt.subplot(3,3,1)
    # plt.imshow(ori_img)

    # plt.subplot(3,3,2)
    # plt.imshow(hsv_img)

    # plt.subplot(3,3,3)
    # plt.imshow(lab_img)

    # plt.subplot(3,3,4)
    # plt.imshow(yuv_img)

    # plt.subplot(3,3,5)
    # plt.imshow(hsl_img)

    # plt.subplot(3,3,6)
    # plt.imshow(b)

    # plt.subplot(3,3,7)
    # plt.imshow(g)

    # plt.subplot(3,3,8)
    # plt.imshow(r)

    # plt.show()


    # Watershed
    # ori_img_copy = ori_img.copy()
    # imgGray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", imgGray)

    # _, thrImg = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Threshold Image", thrImg)

    # sE = np.ones((3,3), dtype=np.uint8)

    # sureBg = cv2.dilate(thrImg, sE, iterations=3)
    # cv2.imshow("Dilated Image", sureBg)

    # distTransform = cv2.distanceTransform(thrImg, cv2.DIST_L2, 5)
    # _, sureFg = cv2.threshold(distTransform, 0.7 * distTransform.max(), 255, 0)

    # sureFg = np.uint8(sureFg)
    # unknownRegion = cv2.subtract(sureBg, sureFg)

    # _, markers = cv2.connectedComponents(sureFg)
    # markers = markers+1
    # markers[unknownRegion==255] = 0

    # watershedMarkers = cv2.watershed(ori_img,markers.copy())
    # # ori_img[watershedMarkers == -1] = [255,255,0]
    # row, col, ch = ori_img.shape
    # ori_img = ori_img[1:row, 1:col]
    # watershedMarkers = watershedMarkers[1:row, 1:col]
    # ori_img[watershedMarkers == 2] = [255, 0, 0]

    # plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.imshow(ori_img_copy)

    # plt.figure(2)
    # plt.subplot(2,1,2)
    # plt.imshow(ori_img)
    # plt.show()


    # Display watershed image
    # plt.figurer


    # Apply Sobel edge detection to the image
    # sobel_hori = cv2.Sobel(ori_img, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_vert = cv2.Sobel(ori_img, cv2.CV_64F, 0, 1, ksize=3)

    # sobel_hori_mag = np.abs(sobel_hori)
    # sobel_vert_mag = np.abs(sobel_vert)
    # sobel_hori_mag = cv2.convertScaleAbs(sobel_hori)
    # sobel_vert_mag = cv2.convertScaleAbs(sobel_vert)

    # # Show magnitude of horizontal and vertical Sobel edges
    # # cv2.imshow("Sobel Horizontal Magnitude", sobel_hori_mag)
    # # cv2.imshow("Sobel Vertical Magnitude", sobel_vert_mag)

    # combined_mag = (0.5 * sobel_hori_mag) + (0.5 * sobel_vert_mag)
    # combined_mag = combined_mag.astype(np.uint8)
    # combined_mag = cv2.addWeighted(sobel_hori_mag, 0.5, sobel_vert_mag, 0.5, 0)
    # combined_mag = cv2.cvtColor(combined_mag, cv2.COLOR_BGR2GRAY)

    # # cv2.imshow("Sobel Edge Detection", combined_mag)
    # # plt.imshow(combined_mag, cmap='gray')
    # # plt.show()
    # # print("Lowest value in Sobel edge detection: ", np.min(combined_mag))

    # # Once the combined magnitude image is obtained, go through each column and find the first pixel that exceeds the threshold.
    # # The returned set will be the boundary points of the building.
    # threshold = 30
    # boundary_points = []
    # print(combined_mag.shape)
    # for col in range(combined_mag.shape[1]):
    #     for row in range(combined_mag.shape[0]):
    #         if combined_mag[row, col] > threshold:
    #             boundary_points.append((row, col))
    #             break

    # # print("Boundary points: ", boundary_points)

    # # For each boundary point, generate a mask of the sky.
    # mask = np.full(combined_mag.shape, 255, dtype=np.uint8)
    # for point in boundary_points:
    #     mask[point[0]:, point[1]] = 0

    # # cv2.imshow("Sky Segmentation Mask", mask)

    # # Use the generated mask to remove the sky from the original image.
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # # sky_img = cv2.bitwise_and(ori_img, mask)
    # sky_img = cv2.bitwise_and(ori_img_color, mask)

    # # Display the sky image
    # # cv2.imshow("Sky Image", sky_img)

    # utils.plot_subplots((2, 2), [ori_img, combined_mag, mask, sky_img], ["Original Image", "Combined Magnitude", "Sky Segmentation Mask", "Sky Image"])

    # Uncomment below when you want to show the image
    # cv2.waitKey()
