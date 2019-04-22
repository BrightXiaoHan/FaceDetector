"""
Test case for data.py
"""
import os
import unittest

import mtcnn.train.data as data

here = os.path.dirname(__file__)


class TestData(unittest.TestCase):

    def setUp(self):
        self.output_folder = os.path.join(here, '../output/test/')
        self.net_stage = 'pnet'
        self.batch_size = 128

    def test_data(self):
        dataset = data.MtcnnDataset(self.output_folder, self.net_stage, self.batch_size)

        for batch in dataset.get_iter():
            self.assertEqual(len(batch), 4)
            print(1)
