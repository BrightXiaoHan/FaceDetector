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
        self.batch_size = 32

    def test_data(self):
        loader = data.get_data_loader(self.output_folder, self.net_stage, self.batch_size)
        for batch in loader:
            self.assertEqual(batch[0].shape[0], self.batch_size)
            self.assertEqual(tuple(batch[1].shape), (self.batch_size,))
            self.assertEqual(tuple(batch[2].shape), (self.batch_size, 4))
            self.assertEqual(tuple(batch[3].shape), (self.batch_size, 10))
