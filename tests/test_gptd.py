"""
Test Cases for generating pnet training data.
"""

import os
import sys
import random
import unittest
import cv2

from mtcnn.datasets import get_by_name
import mtcnn.train.gen_pnet_train as gptd

DEFAULT_DATASET = 'WiderFace'

here = os.path.dirname(__file__)


class TestGenTrain(unittest.TestCase):

    def setUp(self):
        self.dataset = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test')
        self.top = 100

    def test_gen_pnet_train(self):
        meta = self.dataset.get_train_meta()
        meta = random.choices(meta, k=self.top)
        gptd.generate_training_data_for_pnet(
            meta, output_folder=self.output_folder, crop_size=48)

    def test_get_pnet_train(self):
        pnet_data = gptd.get_training_data_for_pnet(self.output_folder)
        
        part, part_reg = pnet_data.part, pnet_data.part_reg
