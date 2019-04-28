import os
import sys
import cv2
import random
import shutil
import progressbar

import numpy as np
import numpy.random as npr
import pandas as pd

from mtcnn.deploy.detect import FaceDetector
from mtcnn.utils.functional import IoU

here = os.path.dirname(__file__)

def generate_training_data_for_onet(pnet, rnet, meta_data, output_folder, crop_size=48, suffix='onet'):
    """
    For training P-net, crop positive(0), negative(1) and partface(2) from original images. 
    The Generated file will be saved in "output_folder"

    Args:
        pnet (Pnet): Pre-trained pnet network.
        rnet (Rnet): Pre-trained rnet network.
        meta_data (list): Each item contains a dict with file_name, num_bb (Number of bounding box), meta_data(x1, y1, w, h, **).
        output_folder (str): Directory to save the result.
        crop_size (int): image size to crop.
        suffix (str): Create a folder named $suffix in $output_folder to save the result.
    """

    # Construct FaceDetector manually 
    detector = FaceDetector.__new__(FaceDetector)
    detector.pnet = pnet
    detector.rnet = rnet
    detector.device = pnet.device 

    # Prepare for output folder.
    rnet_output_folder = os.path.join(output_folder, suffix)

    positive_dest = os.path.join(rnet_output_folder, 'positive')
    negative_dest = os.path.join(rnet_output_folder, 'negative')
    part_dest = os.path.join(rnet_output_folder, 'part')

    [shutil.rmtree(x) for x in (positive_dest, negative_dest,
                              part_dest) if os.path.exists(x)]

    # Make dest dir recursively
    [os.makedirs(x) for x in (positive_dest, negative_dest,
                              part_dest) if not os.path.exists(x)]

    positive_meta_file = open(os.path.join(
        rnet_output_folder, 'positive_meta.csv'), 'w')
    part_meta_file = open(os.path.join(rnet_output_folder, 'part_meta.csv'), 'w')
    negative_meta_file = open(os.path.join(
        rnet_output_folder, 'negative_meta.csv'), 'w')

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

        # Get boxes. (x1, y1, w, h) -> (x1, y1, x2, y2)
        boxes = np.array(item['meta_data'])[:, :4]
        boxes = boxes[boxes[:,2] >= 0]   # filter error box (w <0)
        boxes = boxes[boxes[:,3] >= 0]  # filter error box (h <0)

        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Origin image height and width
        height, width, _ = img.shape

        processed_img = detector._preprocess(img)
        candidate_boxes = detector.stage_one(processed_img, 0.5, 0.707, 12, 0.7)
        try:
            candidate_boxes = detector.stage_two(processed_img, candidate_boxes, 0.5, 0.7)
        except RuntimeError:
            print("Out of memory on process img '%s.'" % file_name)
            continue
        candidate_boxes = detector._convert_to_square(candidate_boxes).cpu().numpy()

        neg_examples = []
        part_examples = []
        part_offsets = []
        pos_num = 0
        part_num = 0
        neg_num = 0

        for c_box in candidate_boxes:
            nx1 = c_box[0]
            ny1 = c_box[1]
            nx2 = c_box[2]
            ny2 = c_box[3]

            w = nx2 - nx1 + 1
            h = ny2 - ny1 + 1

            if nx2 > width or ny2 > height or nx1 < 0 or ny1<0:
                continue

            cropped_im = img[c_box[1]: c_box[3], c_box[0]: c_box[2], :]
            resized_im = cv2.resize(
                    cropped_im, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

            iou = IoU(c_box, boxes)
            max_iou = iou.max()

            if max_iou < 0.3:
                neg_num += 1
                neg_examples.append(resized_im)
                continue

            max_index = iou.argmax()

            x1, y1, x2, y2 = boxes[max_index]

            offset_x1 = (x1 - nx1) / float(w)
            offset_y1 = (y1 - ny1) / float(h)
            offset_x2 = (x2 - nx2) / float(w)
            offset_y2 = (y2 - ny2) / float(h)

            if max_iou >= 0.65:
                pos_num += 1
                total_pos_num += 1
                positive_meta_file.write(
                    ','.join([str(total_pos_num) + '.jpg', str(offset_x1), str(offset_y1), str(offset_x2), str(offset_y2)]) + '\n')

                cv2.imwrite(os.path.join(positive_dest, str(total_pos_num) + '.jpg'), resized_im)


            elif max_iou >= 0.4:
                part_num += 1
                part_examples.append(resized_im)
                part_offsets.append([str(offset_x1), str(offset_y1), str(offset_x2), str(offset_y2)])

        # Prevent excessive negative samples
        if neg_num > 4 * pos_num:
            neg_examples = random.sample(neg_examples, k=3*pos_num)
        
        for i in neg_examples:
            total_neg_num += 1
            negative_meta_file.write(','.join([str(total_neg_num) + '.jpg']) + '\n')
            cv2.imwrite(os.path.join(negative_dest, str(total_neg_num) + '.jpg'), i)

        # Prevent excessive part samples
        if part_num > 2 * pos_num:
            choiced_index = random.sample(list(range(part_num)), k=2*pos_num)
            part_examples = [part_examples[i] for i in choiced_index]
            part_offsets = [part_offsets[i] for i in choiced_index]

        for i, offsets in zip(part_examples, part_offsets):
            total_part_num += 1
            part_meta_file.write(str(total_part_num) + '.jpg,' + ','.join(offsets) + '\n')
                
            cv2.imwrite(os.path.join(part_dest, str(total_part_num) + '.jpg'), i)

    bar.update()

    # Close the meta data files
    [x.close() for x in (positive_meta_file, part_meta_file, negative_meta_file)]
