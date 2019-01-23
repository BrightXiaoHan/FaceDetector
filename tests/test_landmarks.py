"""
Test cases for generate facial landmark localization training data
"""

import os
import sys
import unittest
import cv2

from mtcnn.datasets import get_by_name
import mtcnn.utils.gen_landmark as gl

DEFAULT_DATASET = 'CelebA'

here = os.path.dirname(__file__)

class TestGenLandmarks(unittest.TestCase):
    
    def setUp(self):
        self.datasets = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test/landmarks_12')

    def test_gen_landmark_data(self):
        meta = self.datasets.get_train_meta()
        gl.gen_landmark_data(meta, 48, self.output_folder)

    def test_get_landmark_data(self):
        images, landmarks = gl.get_landmark_data(self.output_folder)
        self.assertEqual(len(images), len(landmarks))

        # Random sampling 100 pictures and draw landmark points on them.
        output_folder = os.path.join(self.output_folder, 'sample_images')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        for im, lm in zip(images[:100], landmarks[:100]):
            pass
