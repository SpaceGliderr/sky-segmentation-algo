from os import walk

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sky_detector as detector
import config
from itertools import repeat

def get_corrupted():
    """
    Returns a list of corrupted images.
    """
    dir_name = "test-samples"
    curr_imgs_dir_name = "4232"
    imgs_filenames = next(walk(f'./{dir_name}/{curr_imgs_dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    corrupted = 0

    for curr_img_filename in imgs_filenames:
        # Read selected image
        curr_img_path = f'./{dir_name}/{curr_imgs_dir_name}/{curr_img_filename}'

        with open(curr_img_path, 'rb') as f:
            check_chars = f.read()[-2:]

        if check_chars != b'\xff\xd9':
            corrupted += 1

    print("Corrupted images: ", corrupted)


def generate_confusion_matrix(ground_truth, mask):
    """
    Calculates the confusion matrix for the algorithm.
    """
    return confusion_matrix(ground_truth.flatten(), mask.flatten(), labels=[0, 255]).ravel()


def generate_classification_report(ground_truth, mask):
    """
    Calculates the classification report for the algorithm.
    """
    return classification_report(ground_truth.flatten(), mask.flatten(), labels=[0, 255])


def evaluate(algorithm):
    """
    Evaluates the performance of any algorithm on a dataset.
    """
    dir_name = "test-samples"
    sample_dir_names = next(walk(f'./{dir_name}'), (None, None, []))[1] # Gets all the subdirectories in the target sample directory.

    for curr_imgs_dir_name in sample_dir_names:
        print("Currently evaluating: ", curr_imgs_dir_name)

        imgs_filenames = next(walk(f'./{dir_name}/{curr_imgs_dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.

        # Performance metrics
        day_list = []
        night_list = []

        days, nights = 0, 0

        for curr_img_filename in imgs_filenames:
            # Read selected image
            curr_img_path = f'./{dir_name}/{curr_imgs_dir_name}/{curr_img_filename}'

            with open(curr_img_path, 'rb') as f:
                check_chars = f.read()[-2:]

            if check_chars != b'\xff\xd9':
                print("Skipping incomplete image")
            else:
                curr_img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)

                _, day = detector.calculate_boundary_threshold(curr_img, config.row_percentage, config.day_night_threshold)

                # Run algorithm on selected image
                mask, _ = algorithm(curr_img)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(f'results/{curr_imgs_dir_name}/{curr_img_filename}', mask) # Log results

                if day:
                    day_list.extend(mask.flatten())
                    days += 1
                else:
                    night_list.extend(mask.flatten())
                    nights += 1


        # Get ground truth image
        ground_truth = cv2.imread(f'./ground_truths/{curr_imgs_dir_name}.png', cv2.IMREAD_GRAYSCALE)

        day_gt_list = []
        day_gt_list.extend(repeat(ground_truth.flatten(), days))

        night_gt_list = []
        night_gt_list.extend(repeat(ground_truth.flatten(), nights))

        day = np.array(day_list)
        night = np.array(night_list)

        day_gt = np.array(day_gt_list)
        night_gt = np.array(night_gt_list)

        print("Calculating Performance")

        dtn, dfp, dfn, dtp = generate_confusion_matrix(day_gt, day)
        print(f'Days: {days} - TN:{dtn} FP:{dfp} FN:{dfn} TP:{dtp}')

        ntn, nfp, nfn, ntp = generate_confusion_matrix(night_gt, night)
        print(f'Night: {nights} - TN:{ntn} FP:{nfp} FN:{nfn} TP:{ntp}')

        totals = days + nights

        print(f'Total: {totals} - TN:{dtn + ntn} FP:{dfp + nfp} FN:{dfn + nfn} TP:{dtp + ntp}')


def evaluate_results(results_dir_name):
    dir_name = f'results/{results_dir_name}'

    imgs_filenames = next(walk(f'./{dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.

    # Performance metrics
    total_list = []

    total = 0

    for curr_img_filename in imgs_filenames:
        # Read selected image
        curr_img_path = f'./{dir_name}/{curr_img_filename}'

        with open(curr_img_path, 'rb') as f:
            check_chars = f.read()[-2:]

        if check_chars != b'\xff\xd9':
            print("Skipping incomplete image")
        else:
            print(curr_img_filename)
            curr_img = cv2.imread(curr_img_path, cv2.IMREAD_GRAYSCALE)

            total_list.extend(curr_img.flatten())
            total += 1

    # Get ground truth image
    ground_truth = cv2.imread(f'./ground_truths/{results_dir_name}.png', cv2.IMREAD_GRAYSCALE)

    total_gt_list = []
    total_gt_list.extend(repeat(ground_truth.flatten(), total))

    total_gt = np.array(total_gt_list)
    total = np.array(total_list)

    print("Calculating Performance")

    total_classification_report = generate_classification_report(total_gt, total)

    print("Total Classification Report\n", total_classification_report)
