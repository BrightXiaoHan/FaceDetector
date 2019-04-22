import os
import sys
import cv2
import shutil
import progressbar

import numpy as np
import numpy.random as npr
import pandas as pd

from mtcnn.utils.functional import IoU

here = os.path.dirname(__file__)


def generate_training_data_for_pnet(meta_data, output_folder, crop_size=12, suffix='pnet'):
    """
    For training P-net, crop positive(0), negative(1) and partface(2) from original images. 
    The Generated file will be saved in "output_folder"

    Args:
        meta_data (list): Each item contains a dict with file_name, num_bb (Number of bounding box), meta_data(x1, y1, w, h, **).
        output_folder (str): Directory to save the result.
        crop_size (int): image size to crop.
    """
    pnet_data_folder = os.path.join(output_folder, suffix)

    positive_dest = os.path.join(pnet_data_folder, 'positive')
    negative_dest = os.path.join(pnet_data_folder, 'negative')
    part_dest = os.path.join(pnet_data_folder, 'part')

    [shutil.rmtree(x) for x in (positive_dest, negative_dest,
                              part_dest) if os.path.exists(x)]

    # Make dest dir recursively
    [os.makedirs(x) for x in (positive_dest, negative_dest,
                              part_dest) if not os.path.exists(x)]

    positive_meta_file = open(os.path.join(
        pnet_data_folder, 'positive_meta.csv'), 'w')
    part_meta_file = open(os.path.join(pnet_data_folder, 'part_meta.csv'), 'w')
    negative_meta_file = open(os.path.join(
        pnet_data_folder, 'negative_meta.csv'), 'w')

    # print("Start generate training data for pnet.")
    bar = progressbar.ProgressBar(max_value=len(meta_data) - 1)

    total_pos_num = 0
    total_neg_num = 0
    total_part_num = 0

    # Traverse all images in training set.
    for index, item in enumerate(meta_data):
        bar.update(index)
        # Read the image
        file_name = item['file_name']
        img = cv2.imread(file_name)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Do this for compatible with caffe model

        # Get boxes. (x1, y1, w, h) -> (x1, y1, x2, y2)
        boxes = np.array(item['meta_data'])[:, :4]
        boxes = boxes[boxes[:,2] >= 0]   # filter error box (w <0)
        boxes = boxes[boxes[:,3] >= 0]  # filter error box (h <0)

        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Origin image height and width
        height, width, _ = img.shape

        # Record the total number of positive, negative and part examples.
        neg_num = 0
        pos_num = 0
        part_num = 0

        # Random pick 50 negative examples
        while neg_num < 50:

            size = npr.randint(crop_size, min(width, height) / 2)

            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)

            crop_box = np.array([nx, ny, nx + size, ny + size])

            iou = IoU(crop_box, boxes)

            if np.max(iou) < 0.3:
                # Iou with all gts must below 0.3
                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (crop_size, crop_size),
                                        interpolation=cv2.INTER_LINEAR)

                total_neg_num += 1
                neg_num += 1

                negative_meta_file.write(','.join([str(total_neg_num) + '.jpg']) + '\n')
                cv2.imwrite(os.path.join(negative_dest, str(total_neg_num) + '.jpg'), resized_im)

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # generate negative examples that have overlap with gt
            for i in range(5):
                size = npr.randint(crop_size,  min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)

                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(
                    cropped_im, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    neg_num += 1
                    total_neg_num += 1
                    negative_meta_file.write(','.join([str(total_neg_num) + '.jpg']) + '\n')
                    cv2.imwrite(os.path.join(negative_dest, str(total_neg_num) + '.jpg'), resized_im)

            # generate positive examples and part faces
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8),
                                   np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv2.resize(
                    cropped_im, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    pos_num += 1
                    total_pos_num += 1
                    positive_meta_file.write(
                        ','.join([str(total_pos_num) + '.jpg', str(offset_x1), str(offset_y1), str(offset_x2), str(offset_y2)]) + '\n')

                    cv2.imwrite(os.path.join(positive_dest, str(total_pos_num) + '.jpg'), resized_im)


                elif IoU(crop_box, box_) >= 0.4:
                    part_num += 1
                    total_part_num += 1
                    part_meta_file.write(
                        ','.join([str(total_part_num) + '.jpg', str(offset_x1), str(offset_y1), str(offset_x2), str(offset_y2)]) + '\n')
                    
                    cv2.imwrite(os.path.join(part_dest, str(total_part_num) + '.jpg'), resized_im)
    bar.update()
    # print("\nDone")

    # Close the meta data files
    [x.close() for x in (positive_meta_file, part_meta_file, negative_meta_file)]
