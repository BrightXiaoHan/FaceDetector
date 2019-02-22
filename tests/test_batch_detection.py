"""
Test cases for mtcnn.deploy.detect.py
"""
import os
import time
import cv2
import torch
import unittest

import numpy as np
import mtcnn.deploy.batch_detect as detect
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

        self.detector = detect.BatchImageDetector(pnet, rnet, onet)
        self.batch_imgs = [os.path.join(here, 'asset/images/office5.jpg')] * 10

    def test_stage_one(self):
        imgs = [cv2.imread(img) for img in self.batch_imgs]

        norm_imgs = self.detector._preprocess(imgs)
        stage_one_boxes = self.detector.stage_one(
            norm_imgs, 0.6, 0.707, 12, 0.7)
        img_labels = stage_one_boxes[:, -1]

        for i, img in enumerate(imgs):
            mask = img_labels == i
            boxes = stage_one_boxes[mask]
            draw.draw_boxes2(img, boxes)
            cv2.imshow('Stage One Boxes', img)
            cv2.waitKey(0)

    def test_stage_two(self):
        # Running this test case after 'test_stage_one' passed.
        imgs = [cv2.imread(img) for img in self.batch_imgs]
        
        norm_imgs = self.detector._preprocess(imgs)
        stage_one_boxes = self.detector.stage_one(norm_imgs, 0.6, 0.707, 12, 0.7)
        stage_two_boxes = self.detector.stage_two(norm_imgs, stage_one_boxes, 0.7, 0.7)
        img_labels = stage_two_boxes[:, -1]
        
        for i, img in enumerate(imgs):
            mask = img_labels == i
            boxes = stage_two_boxes[mask]
            draw.draw_boxes2(img, boxes)
            cv2.imshow('Stage One Boxes', img)
            cv2.waitKey(0)

    def test_stage_three(self):
        # Running this test case after 'test_stage_one' passed.
        imgs = [cv2.imread(img) for img in self.batch_imgs]
        
        norm_imgs = self.detector._preprocess(imgs)
        stage_one_boxes = self.detector.stage_one(norm_imgs, 0.6, 0.707, 12, 0.7)
        stage_two_boxes = self.detector.stage_two(norm_imgs, stage_one_boxes, 0.7, 0.7)
        stage_three_boxes, landmarks = self.detector.stage_three(norm_imgs, stage_two_boxes, 0.7, 0.3)
        img_labels = stage_three_boxes[:, -1]
        
        for i, img in enumerate(imgs):
            mask = img_labels == i
            boxes = stage_three_boxes[mask]
            draw.draw_boxes2(img, boxes)
            cv2.imshow('Stage One Boxes', img)
            cv2.waitKey(0)
 
    def test_performance(self):
        imgs = [cv2.imread(img) for img in self.batch_imgs]
        start = time.time()
        self.detector.detect(imgs)
        end = time.time()
        avg_time = (end - start) / len(imgs)
        print("Average time per image is %fs." % avg_time)