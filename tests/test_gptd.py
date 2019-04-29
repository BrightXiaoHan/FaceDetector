"""
Test Cases for generating pnet training data.
"""

import os
import sys
import random
import unittest
import cv2

from mtcnn.datasets import get_by_name
from mtcnn import get_net_caffe
import mtcnn.train.gen_pnet_train as gptd
import mtcnn.train.gen_rnet_train as grtd
import mtcnn.train.gen_onet_train as gotd
from mtcnn.train.data import get_training_data


DEFAULT_DATASET = 'WiderFace'

here = os.path.dirname(__file__)


class TestGenTrain(unittest.TestCase):

    def setUp(self):
        self.dataset = get_by_name(DEFAULT_DATASET)
        self.output_folder = os.path.join(here, '../output/test')
        self.top = 100
        self.pnet, self.rnet, _ = get_net_caffe(os.path.join(here, '../output/converted'))

    def test_gen_pnet_train(self):
        meta = self.dataset.get_train_meta()
        meta = random.choices(meta, k=self.top)
        gptd.generate_training_data_for_pnet(
            meta, output_folder=self.output_folder, crop_size=12)
        eval_meta = self.dataset.get_val_meta()
        eval_meta = random.choices(eval_meta, k=self.top)
        gptd.generate_training_data_for_pnet(eval_meta, output_folder=self.output_folder, crop_size=12, suffix='pnet_eval')

    def test_get_pnet_train(self):
        pnet_data = get_training_data(self.output_folder, suffix='pnet')
        pnet_eval_data = get_training_data(self.output_folder, suffix='pnet_eval')

    def test_gen_rnet_train(self):
        meta = self.dataset.get_train_meta()
        meta = random.choices(meta, k=self.top)
        grtd.generate_training_data_for_rnet(self.pnet, meta, self.output_folder, suffix='rnet')
        eval_meta = self.dataset.get_val_meta()
        eval_meta = random.choices(eval_meta, k=self.top)
        grtd.generate_training_data_for_rnet(self.pnet, eval_meta, self.output_folder, suffix='rnet_eval')

    def test_get_rnet_train(self):
        rnet_data = get_training_data(self.output_folder, suffix='rnet')
        rnet_eval = get_training_data(self.output_folder, suffix='rnet_eval')

    def test_gen_onet_train(self):
        meta = self.dataset.get_train_meta()
        meta = random.choices(meta, k=self.top)
        gotd.generate_training_data_for_onet(self.pnet, self.rnet, meta, self.output_folder, suffix='onet')
        eval_meta = self.dataset.get_val_meta()
        eval_meta = random.choices(eval_meta, k=self.top)
        gotd.generate_training_data_for_onet(self.pnet, self.rnet, eval_meta, self.output_folder, suffix='onet_eval') 

    def test_get_onet_train(self):
        rnet_data = get_training_data(self.output_folder, suffix='onet')
        rnet_eval = get_training_data(self.output_folder, suffix='onet_eval')