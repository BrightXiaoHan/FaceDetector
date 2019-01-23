"""
Test Cases for generating pnet training data.
"""

import os
import sys
import unittest
import cv2

from mtcnn.datasets import get_by_name
import mtcnn.utils.generate_pnet_training_data as gptd

DEFAULT_DATASET = 'WiderFace'

here = os.path.dirname(__file__)


class TestGenTrain(unittest.TestCase):

    def setUp(self):
        self.dataset = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test')

    def test_gen_pnet_train(self):
        meta = self.dataset.get_train_meta()
        gptd.generate_training_data_for_pnet(
            meta, output_folder=self.output_folder)

    def test_get_pnet_train(self):
        images, meta = gptd.get_training_data_for_pnet(self.output_folder)
        self.assertEqual(len(images), 3)
        self.assertEqual(len(meta), 3)

        # Random sampling 100 pictures from "positive", "negative" and "part" examples.
        output_folder = os.path.join(self.output_folder, 'pnet', 'sample_images')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        for i, (pos, neg, part) in enumerate(zip(*images)):
            cv2.imwrite(os.path.join(
                output_folder, 'pos_%d.jpg' % i), pos)
            cv2.imwrite(os.path.join(
                output_folder, 'neg_%d.jpg' % i), neg)
            cv2.imwrite(os.path.join(
                output_folder, 'part_%d.jpg' % i), part)

            if i > 100:
                break
