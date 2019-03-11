"""
Test Cases mtcnn.utils.functional
"""

import os
import torch
import sys
import unittest
import cv2
import numpy as np

import mtcnn.utils.functional as func

here = os.path.dirname(__file__)

class TestFunctional(unittest.TestCase):

    def setUp(self):
        # construct a list containing the images that will be examined
        # along with their respective bounding boxes
        self.images = [
            (os.path.join(here, 'asset/images/audrey.jpg'), np.array([
            (12, 84, 140, 212, 1),
            (24, 84, 152, 212, 1),
            (36, 84, 164, 212, 1),
            (12, 96, 140, 224, 1),
            (24, 96, 152, 224, 1),
            (24, 108, 152, 236, 1)]), 1),
            (os.path.join(here, 'asset/images/bksomels.jpg'), np.array([
            (114, 60, 178, 124, 1),
            (120, 60, 184, 124, 1),
            (114, 66, 178, 130, 1)]), 1),
            (os.path.join(here, 'asset/images/gpripe.jpg'), np.array([
            (12, 30, 76, 94, 1),
            (12, 36, 76, 100, 1),
            (72, 36, 200, 164, 1),
            (84, 48, 212, 176, 1)]), 2)]

    def test_iou(self):
        pass

    def test_nms(self):

        # loop over the images
        for (imagePath, boundingBoxes, num_face) in self.images:
            # load the image and clone it
            print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
            image = cv2.imread(imagePath)
            orig = image.copy()

            # loop over the bounding boxes for each image and draw them
            for (startX, startY, endX, endY, _) in boundingBoxes:
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # perform non-maximum suppression on the bounding boxes
            pick = func.nms(boundingBoxes[:, :4], boundingBoxes[:, 4], 0.3)
            print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

            # loop over the picked bounding boxes and draw them
            for i in pick:
                (startX, startY, endX, endY) = boundingBoxes[i][:4]
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # # display the images
            # cv2.imshow("Original", orig)
            # cv2.imshow("After NMS", image)
            # cv2.waitKey(0)
            self.assertEqual(len(pick), num_face)

    def test_iou_torch(self):
        boxes = self.images[0][1][:, :4]
        b = torch.IntTensor(boxes[0])
        boxes = torch.IntTensor(boxes)

        over = func.iou_torch(b, boxes).numpy().tolist()
        self.assertEqual(over[0], 1)
        