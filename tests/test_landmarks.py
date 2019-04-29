"""
Test cases for generate facial landmark localization training data
"""

import os
import sys
import random
import unittest
import cv2

from mtcnn.datasets import get_by_name
import mtcnn.train.gen_landmark as gl
from mtcnn.train.data import get_landmark_data
from mtcnn.utils import draw

DEFAULT_DATASET = 'CelebA'

here = os.path.dirname(__file__)


class TestGenLandmarks(unittest.TestCase):

    def setUp(self):
        self.datasets = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test')
        self.top = 1000
        self.crop_size = 24
        self.suffix = 'rnet'
        self.argument = False

    def test_gen_landmark_data(self):
        meta = self.datasets.get_train_meta()
        meta = random.choices(meta, k=self.top)
        eval_meta = self.datasets.get_val_meta()
        eval_meta = random.choices(eval_meta, k=self.top)
        gl.gen_landmark_data(eval_meta, self.crop_size, self.output_folder, argument=self.argument, suffix=self.suffix + '_eval')
        gl.gen_landmark_data(meta, self.crop_size, self.output_folder, argument=self.argument, suffix=self.suffix)

    def test_get_landmark_data(self):
        data = get_landmark_data(self.output_folder, suffix=self.suffix)

        images, landmarks = data.images, data.landmarks

        self.assertEqual(len(images), len(landmarks))

        # Random sampling 10 pictures and draw landmark points on them.
        output_folder = os.path.join(self.output_folder, 'sample_images', 'landmarks')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # convert from (n, 10) to (n, 5, 2) 
        landmarks = landmarks.reshape(-1, 2, 5).transpose(0, 2, 1)

        for i, (im, lm) in enumerate(zip(images[:10], landmarks[:10])):
            im = cv2.imread(im)
            w = im.shape[0]
            h = im.shape[1]

            lm[:, 0] *= w
            lm[:, 1] *= h

            lm = lm.astype(int)

            draw.draw_landmarks(im, lm)
            cv2.imwrite(os.path.join(output_folder, '%d.jpg' % i), im)
