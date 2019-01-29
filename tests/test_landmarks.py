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
from mtcnn.utils import draw

DEFAULT_DATASET = 'CelebA'

here = os.path.dirname(__file__)


class TestGenLandmarks(unittest.TestCase):

    def setUp(self):
        self.datasets = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test/pnet')
        self.top = 100

    def test_gen_landmark_data(self):
        meta = self.datasets.get_train_meta()
        meta = random.choices(meta, k=self.top)
        gl.gen_landmark_data(meta, 12, self.output_folder, argument=True)

    def test_get_landmark_data(self):
        images, landmarks = gl.get_landmark_data(self.output_folder)
        self.assertEqual(len(images), len(landmarks))

        # Random sampling 10 pictures and draw landmark points on them.
        output_folder = os.path.join(self.output_folder, 'sample_images')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        for i, (im, lm) in enumerate(zip(images[:10], landmarks[:10])):
            w = im.shape[0]
            h = im.shape[1]

            lm[:, 0] *= w
            lm[:, 1] *= h

            lm = lm.astype(int)

            draw.draw_landmarks(im, lm)
            cv2.imwrite(os.path.join(output_folder, '%d.jpg' % i), im)
