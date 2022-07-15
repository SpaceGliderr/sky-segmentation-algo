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
    dir_name = "10917-samples"
    sample_dir_names = next(walk(f'./{dir_name}'), (None, None, []))[1] # Gets all the subdirectories in the target sample directory.

    # for curr_imgs_dir_name in sample_dir_names:
    curr_imgs_dir_name = "10917"
    print("Currently evaluating: ", curr_imgs_dir_name)

    imgs_filenames = next(walk(f'./{dir_name}/{curr_imgs_dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    # print(imgs_filenames)
    # imgs_filenames = next(walk(f'./{dir_name}/4232'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    # imgs_filenames.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1]))) # Sorts image filenames in ascending order
    
    # Performance metrics
    day_list = []
    night_list = []

    days, nights = 0, 0

    for curr_img_filename in imgs_filenames:
        # Read selected image
        curr_img_path = f'./{dir_name}/{curr_imgs_dir_name}/{curr_img_filename}'
        # curr_img_path = f'./{dir_name}/4232/{curr_img_filename}'
        # print(curr_img_path)

        with open(curr_img_path, 'rb') as f:
            check_chars = f.read()[-2:]

        if check_chars != b'\xff\xd9':
            print("Skipping incomplete image")
        else:
            # print(curr_img_filename)
            curr_img = cv2.imread(curr_img_path, cv2.IMREAD_COLOR)
            # print(curr_img.shape)

            _, day = detector.calculate_boundary_threshold(curr_img, config.row_percentage, config.day_night_threshold)

            # Run algorithm on selected image
            mask, _ = algorithm(curr_img)
            # print(mask.shape)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # print(set(mask.flatten()))

            # cv2.imwrite(f'results/4232/{curr_img_filename}', mask)
            cv2.imwrite(f'results/{curr_imgs_dir_name}/{curr_img_filename}', mask)

            # total_list.extend(curr_img.flatten())
            # totals += 1

            if day:
                day_list.extend(mask.flatten())
                days += 1
            else:
                night_list.extend(mask.flatten())
                nights += 1

    # print(np.array(day_list).shape)
    # print(days)
    # print(np.array(night_list).shape)
    # print(nights)

    # Get ground truth image
    ground_truth = cv2.imread(f'./ground_truths/{curr_imgs_dir_name}.png', cv2.IMREAD_GRAYSCALE)
    # ground_truth = cv2.imread(f'./ground_truths/4232.png', cv2.IMREAD_GRAYSCALE)
    # print(ground_truth.flatten().shape)

    day_gt_list = []
    day_gt_list.extend(repeat(ground_truth.flatten(), days))

    night_gt_list = []
    night_gt_list.extend(repeat(ground_truth.flatten(), nights))

    # total_gt_list = []
    # total_gt_list.extend(repeat(ground_truth.flatten(), totals))

    # print(np.array(day_gt_list).flatten().shape)
    # print(np.array(night_gt_list).flatten().shape)

    day = np.array(day_list)
    night = np.array(night_list)
    # total = np.array(total_list)

    day_gt = np.array(day_gt_list)
    night_gt = np.array(night_gt_list)
    # total_gt = np.array(total_gt_list)

    # print(set(day.flatten()) - set(day_gt.flatten()))

    print("Calculating Performance")

    dtn, dfp, dfn, dtp = generate_confusion_matrix(day_gt, day)
    print(f'Days: {days} - TN:{dtn} FP:{dfp} FN:{dfn} TP:{dtp}')

    ntn, nfp, nfn, ntp = generate_confusion_matrix(night_gt, night)
    print(f'Night: {nights} - TN:{ntn} FP:{nfp} FN:{nfn} TP:{ntp}')

    # total = np.concatenate((day.flatten(), night.flatten()))
    # total_gt = np.concatenate((day_gt.flatten(), night_gt.flatten()))
    totals = days + nights

    # ttn, tfp, tfn, ttp = generate_confusion_matrix(total_gt, total)

    # day_classification_report = generate_classification_report(day_gt, day)
    # print("Day Classification Report\n", day_classification_report)

    # night_classification_report = generate_classification_report(night_gt, night)
    # print("Night Classification Report\n", night_classification_report)

    # ttn, tfp, tfn, ttp = generate_confusion_matrix(total_gt, total)
    # total_classification_report = generate_classification_report(total_gt, total)


    print(f'Total: {totals} - TN:{dfn + ntn} FP:{dfp + nfp} FN:{dfn + nfn} TP:{dtp + ntp}')
    # print("Total Classification Report\n", total_classification_report)


def evaluate_results():
    # dir_name = "small-test-samples"
    # sample_dir_names = next(walk(f'./{dir_name}'), (None, None, []))[1] # Gets all the subdirectories in the target sample directory.

    # for curr_imgs_dir_name in sample_dir_names:
    results_dir_name = "10917"
    dir_name = f'results/{results_dir_name}'
    print("Currently evaluating: ", dir_name)

    imgs_filenames = next(walk(f'./{dir_name}'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    # imgs_filenames = next(walk(f'./{dir_name}/4232'), (None, None, []))[2] # Gets all the filenames in the target image set directory.
    imgs_filenames.sort(key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[1]))) # Sorts image filenames in ascending order

    # Performance metrics
    day_list = []
    night_list = []
    total_list = []

    days, nights, total = 0, 0, 0

    for curr_img_filename in imgs_filenames:
        # Read selected image
        curr_img_path = f'./{dir_name}/{curr_img_filename}'
        # curr_img_path = f'./{dir_name}/4232/{curr_img_filename}'
        # print(curr_img_path)

        with open(curr_img_path, 'rb') as f:
            check_chars = f.read()[-2:]

        if check_chars != b'\xff\xd9':
            print("Skipping incomplete image")
        else:
            print(curr_img_filename)
            curr_img = cv2.imread(curr_img_path, cv2.IMREAD_GRAYSCALE)
            # print(curr_img.shape)

            total_list.extend(curr_img.flatten())
            total += 1

            # _, day = detector.calculate_boundary_threshold(curr_img, config.row_percentage, config.day_night_threshold)

            # # Run algorithm on selected image
            # mask, _ = algorithm(curr_img)
            # # print(mask.shape)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # # print(set(mask.flatten()))

            # # cv2.imwrite(f'results/4232/{curr_img_filename}', mask)
            # cv2.imwrite(f'results/{curr_imgs_dir_name}/{curr_img_filename}', mask)

            # if day:
            #     day_list.extend(mask.flatten())
            #     days += 1
            # else:
            #     night_list.extend(mask.flatten())
            #     nights += 1

    # print(np.array(day_list).shape)
    # print(days)
    # print(np.array(night_list).shape)
    # print(nights)

    # Get ground truth image
    ground_truth = cv2.imread(f'./ground_truths/{results_dir_name}.png', cv2.IMREAD_GRAYSCALE)
    # ground_truth = cv2.imread(f'./ground_truths/4232.png', cv2.IMREAD_GRAYSCALE)
    # print(ground_truth.flatten().shape)

    # day_gt_list = []
    # day_gt_list.extend(repeat(ground_truth.flatten(), days))

    # night_gt_list = []
    # night_gt_list.extend(repeat(ground_truth.flatten(), nights))

    # # print(np.array(day_gt_list).flatten().shape)
    # # print(np.array(night_gt_list).flatten().shape)

    # day = np.array(day_list)
    # night = np.array(night_list)

    # day_gt = np.array(day_gt_list)
    # night_gt = np.array(night_gt_list)

    # print(set(day.flatten()) - set(day_gt.flatten()))

    total_gt_list = []
    total_gt_list.extend(repeat(ground_truth.flatten(), total))

    total_gt = np.array(total_gt_list)
    total = np.array(total_list)

    print("Calculating Performance")

    # dtn, dfp, dfn, dtp = generate_confusion_matrix(day_gt, day)
    # ntn, nfp, nfn, ntp = generate_confusion_matrix(night_gt, night)

    # day_classification_report = generate_classification_report(day_gt, day)
    # print("Day Classification Report\n", day_classification_report)

    # night_classification_report = generate_classification_report(night_gt, night)
    # print("Night Classification Report\n", night_classification_report)

    # total = np.concatenate((day.flatten(), night.flatten()))
    # total_gt = np.concatenate((day_gt.flatten(), night_gt.flatten()))

    # ttn, tfp, tfn, ttp = generate_confusion_matrix(total_gt, total)
    total_classification_report = generate_classification_report(total_gt, total)

    # print(f'Days: {days} - TN:{dtn} FP:{dfp} FN:{dfn} TP:{dtp}')
    # print(f'Night: {nights} - TN:{ntn} FP:{nfp} FN:{nfn} TP:{ntp}')
    # print(f'Total - TN:{ttn} FP:{tfp} FN:{tfn} TP:{ttp}')
    print("Total Classification Report\n", total_classification_report)
