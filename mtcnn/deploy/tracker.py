import torch
import numpy as np
import mtcnn.deploy.detect as detect
import mtcnn.utils.functional as func

class FaceTracker(object):

    def __init__(self, detector, re_detect_every=10, min_interval=2, iou_thres=0.7):
        """Set hyper parameters for FaceTracker
        
        Keyword Arguments:
            detector {mtcnn.deploy.detect.FaceDetector} -- FaceDetector object.
            re_detect_every {int} -- How often does this tracker do full mtcnn detection.(default: {10})
            min_interval {int} -- If we can't detect any face in some areas, we drop these boexs out. (default: {2})
            iou_thres (float) -- Iou < iou_thres is regard as the same person.
        """

        self.detector = detector
        self.re_detect_every = re_detect_every
        self.min_interval = min_interval

        self.reset()

    def track(self, frame):
        if self.cur_count % self.cur_count == 0:
            boxes, _ = self.detector.detect(frame)

            for b in self.boxes_cache:
                func.IoU(b, boxes)
        

    def reset(self):
        self.cur_count = 0
        self.boxes_cache = []
        self.label_cache = []
        self.interval_cache = []
        self.image_cache = dict()
