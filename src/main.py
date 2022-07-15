import cv2
import sky_detector as detector
import performance


if __name__ == '__main__':
    # Read selected image
    curr_img_path = "./training-samples/1093/20140419_002403.jpg"
    ori_img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)

    # Segment the sky
    sky_mask, sky = detector.initial_segment_sky(ori_img, True)

    # Performance test
    performance.evaluate(detector.initial_segment_sky)

    # To evaluate the masks in the results folder
    performance.evaluate_results("1093")
