"""
Test cases for mtcnn.deploy.detect.py
"""
import os
import time
import cv2
import unittest

import numpy as np
import mtcnn.deploy.detect as detect
import mtcnn.network.mtcnn_pytorch as mtcnn
import mtcnn.utils.draw as draw

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

        self.detector = detect.FaceDetector(pnet, rnet, onet, "cuda:0")
        self.test_img = os.path.join(here, 'asset/images/office5.jpg')

    def test_detection(self):
        img = cv2.imread(self.test_img)
        boxes, landmarks = self.detector.detect(self.test_img)
        draw.draw_boxes2(img, boxes
        )
        draw.batch_draw_landmarks(img, landmarks)
        cv2.imshow('Stage One Boxes', img)
        cv2.waitKey(0)

    def test_stage_one(self):
        img = cv2.imread(self.test_img)
        norm_img = self.detector._preprocess(self.test_img)
        stage_one_boxes = self.detector.stage_one(norm_img, 0.6 , 0.707, 12, 0.7)
        draw.draw_boxes2(img, stage_one_boxes)
        cv2.imshow('Stage One Boxes', img)
        cv2.waitKey(0)

    def test_stage_two(self):
        # Running this test case after 'test_stage_one' passed.
        img = cv2.imread(self.test_img)
        norm_img = self.detector._preprocess(self.test_img)
        stage_one_boxes = self.detector.stage_one(norm_img, 0.6, 0.707, 12, 0.7)
        stage_two_boxes = self.detector.stage_two(norm_img, stage_one_boxes, 0.7, 0.7)
        draw.draw_boxes2(img, stage_two_boxes)
        cv2.imshow('Stage One Boxes', img)
        cv2.waitKey(0)

    def test_stage_three(self):
        # Running this test  case after 'test_stage_three' passed.
        img = cv2.imread(self.test_img)
        norm_img = self.detector._preprocess(self.test_img)
        stage_one_boxes = self.detector.stage_one(norm_img, 0.6, 0.707, 12, 0.7)
        stage_two_boxes = self.detector.stage_two(norm_img, stage_one_boxes, 0.7, 0.7)
        stage_three_boxes, landmarks = self.detector.stage_three(norm_img, stage_two_boxes, 0.7, 0.3)
        draw.draw_boxes2(img, stage_three_boxes)
        draw.batch_draw_landmarks(img, landmarks)
        cv2.imshow('Stage One Boxes', img)
        cv2.waitKey(0)

    def test_performance(self):
        start = time.time()
        for i in range(10):
            self.detector.detect(self.test_img)
        end = time.time()
        avg_time = (end - start) / 10
        print("Average time cost is %f. " % avg_time)
