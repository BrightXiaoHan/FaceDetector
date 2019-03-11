import os
import cv2
import unittest

import numpy as np
import mtcnn.network.mtcnn_pytorch as mtcnn

from mtcnn.deploy.detect import FaceDetector
from mtcnn.deploy.tracker import FaceTracker

here = os.path.dirname(__file__)

class TestTracker(unittest.TestCase):


    def setUp(self):
        self.test_video = os.path.join(here, './asset/video/school.avi')
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

        self.detector = FaceDetector(pnet, rnet, onet)  
        self.tracker = FaceTracker(self.detector)

    def test_video(self):
        cap = cv2.VideoCapture(self.test_video)
        
        res, frame = cap.read()

        i = 0
        while res:
            self.tracker.track(frame) 
            res, frame = cap.read()
            i += 1
            print("The %sth frame has been processed." % i)
            if i > 50:
                break
        cache = self.tracker.get_cache()

        saved_folder = "/home/hanbing/Desktop/image_folder"
        for key, images in cache.items():
            if len(images) < 5:
                continue
            dir_name = os.path.join(saved_folder, str(key))
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            
                for i, img in enumerate(images):
                    file_name = os.path.join(dir_name, str(i) + '.jpg')
                    cv2.imwrite(file_name, img)
        
        



