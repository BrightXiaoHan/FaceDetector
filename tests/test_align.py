"""
Test cases for mtcnn.deploy.detect.py
"""
import os
import time
import cv2
import unittest

import numpy as np
import mtcnn.deploy.detect as detect
import mtcnn.utils.draw as draw

from mtcnn.deploy import get_net_caffe
from mtcnn.deploy.align import align_multi

here = os.path.dirname(__file__)


class TestDetection(unittest.TestCase):

    def setUp(self):
        pnet, rnet, onet = get_net_caffe("output/converted")

        self.detector = detect.FaceDetector(pnet, rnet, onet)
        self.test_img = os.path.join(here, 'asset/images/office5.jpg')

    def test_detection(self):
        img = cv2.imread(self.test_img)  
        boxes, landmarks = self.detector.detect(self.test_img)
        boxes, faces = align_multi(img, boxes, landmarks, (92, 112))
 
        for face in faces: 
            cv2.imshow("Aligned Faces", face)
            cv2.waitKey(0)


        
