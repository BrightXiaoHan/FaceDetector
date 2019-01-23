import os
import sys
import cv2

import numpy as np

here = os.path.dirname(__file__)


class CelebA(object):

    def __init__(self, dataset_folder=os.path.join(here, 'CelebA')):
        """
        Make sure the WIDER_FACE dataset saved in $SOURCE_ROOT/datasets/CelebA folder.
        """
        self.dataset_folder = dataset_folder

    def _load_all(self):
        """Load metadata of CelebA dataset.

        Returns:
            list: Each item contains a dict with file_name, num_bb (Number of bounding box.), meta_data(x1, y1, w, h), landmarks(lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y).
        """
        ret = []

        anno_folder = os.path.join(self.dataset_folder, 'Anno')
        image_folder = os.path.join(self.dataset_folder, 'img_celeba')
        box_anno = os.path.join(anno_folder, 'list_bbox_celeba.txt')
        landmarks_algn_anno = os.path.join(
            anno_folder, 'list_landmarks_celeba.txt')

        f_box_anno = open(box_anno)
        f_landmarks_anno = open(landmarks_algn_anno)

        for i, (f_box_line, f_landmarks_line) in enumerate(zip(f_box_anno, f_landmarks_anno)):
            if i < 2:  # skip the top two lines in anno files
                continue
            image_name = f_box_line.strip().split(' ')[0]

            boxes = f_box_line.strip().split(' ')[1:]
            boxes = list(filter(lambda x: x != '', boxes))
            boxes = np.array(boxes).astype(int)

            landmarks = f_landmarks_line.strip().split(' ')[1:]
            landmarks = list(filter(lambda x: x != '', landmarks))
            landmarks = np.array(landmarks).astype(int)

            img_path = os.path.join(image_folder, image_name)

            item = {
                'file_name': img_path,
                'num_bb': 1,
                'meta_data': [boxes],
                'landmarks': [landmarks]
            }
            ret.append(item)

        f_box_anno.close()
        f_landmarks_anno.close()

        return ret

    def _split(self):
        """
        Split all_data into train, dev, test parts.
        """
        ret = self._load_all()
        partition_file = os.path.join(
            self.dataset_folder, 'Eval', 'list_eval_partition.txt')
        f_partition = open(partition_file)

        train = []
        dev = []
        test = []

        for line, item in zip(f_partition, ret):
            dtype = int(line.split(' ')[1])
            if dtype == 0:
                train.append(item)
            elif dtype == 1:
                dev.append(item)
            elif dtype == 2:
                test.append(item)

        return train, dev, test

    def get_train_meta(self):
        train, _, _ = self._split()
        return train

    def get_val_meta(self):
        _, dev, _ = self._split()
        return dev

    def get_test_meta(self):
        _, _, test = self._split()
        return test
