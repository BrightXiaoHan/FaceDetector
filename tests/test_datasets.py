"""
Test cases for datasets module.
"""
import os
import cv2
import sys
import unittest
import random
import numpy as np

from mtcnn.datasets import get_by_name
from mtcnn.utils import draw

here = os.path.dirname(__file__)


class TestWiderFace(unittest.TestCase):
    def setUp(self):
        self.datasets = get_by_name("WiderFace")
        self.output_folder = os.path.join(here, '../output/test/wider_face')

    def test_get_widerface_train(self):
        ret = self.datasets.get_train_meta()

        for item in ret:
            self.assertIn('file_name', item)
            self.assertTrue(os.path.exists(item.get('file_name')))
            self.assertIn('num_bb', item)
            self.assertIn('meta_data', item)
            self.assertEqual(len(item.get('meta_data')), item.get('num_bb'))

        # Sameple 10 image and save them in output/test/wider_face folder
        examples = random.choices(ret, k=10)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

        for i, item in enumerate(examples):
            img = cv2.imread(item['file_name'])
            boxes = np.array(item['meta_data'])[:, :4]
            draw.draw_boxes(img, boxes)
            saved_path = os.path.join(self.output_folder, "%d.jpg" % i)
            cv2.imwrite(saved_path, img)

    def test_get_widerface_test(self):
        ret = self.datasets.get_test_meta()

        for item in ret:
            self.assertTrue(os.path.exists(item))


class TestCelebA(unittest.TestCase):

    def setUp(self):
        self.datasets = get_by_name("CelebA")
        self.output_folder = os.path.join(here, '../output/test/celeba')

    def test_get_celeba_train(self):
        ret = self.datasets.get_train_meta()

        for item in ret:
            self.assertIn('file_name', item)
            if not os.path.exists(item['file_name']):
                print(1)
            self.assertTrue(os.path.exists(item['file_name']))
            self.assertIn('num_bb', item)
            self.assertIn('meta_data', item)
            self.assertEqual(len(item.get('meta_data')), item.get('num_bb'))

        # Sameple 10 image and save them in output/test/wider_face folder
        examples = random.choices(ret, k=10)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

        for i, item in enumerate(examples):
            img = cv2.imread(item['file_name'])

            boxes = np.array(item['meta_data'])[:, :4]
            draw.draw_boxes(img, boxes)

            landmarks = np.reshape(np.array(item['landmarks']), (-1, 2))
            draw.draw_landmarks(img, landmarks)

            saved_path = os.path.join(self.output_folder, "%d.jpg" % i)
            cv2.imwrite(saved_path, img)
