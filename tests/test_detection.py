"""
Test cases for mtcnn.deploy.detect.py
"""
import os
import unittest

import numpy as np
import mtcnn.deploy.detect as detect
import mtcnn.network.mtcnn_pytorch as mtcnn

here = os.path.dirname(__file__)


class TestDetection(unittest.TestCase):

    def setUp(self):
        weight_folder = os.path.join(here, '../output/converted')

        pnet = mtcnn.PNet()
        rnet = mtcnn.RNet()
        onet = mtcnn.ONet()

        pnet.load_caffe_model(
            np.load(os.path.join(weight_folder, 'pnet.npy'))[()])
        rnet.load_caffe_model(
            np.load(os.path.join(weight_folder, 'rnet.npy'))[()])
        onet.load_caffe_model(
            np.load(os.path.join(weight_folder, 'onet.npy'))[()])

        self.detector = detect.FaceDetector(pnet, rnet, onet)
        self.test_img = os.path.join(here, 'asset/images/bksomels.jpg')

    def test_detection(self):
        self.detector.detect(self.test_img)
