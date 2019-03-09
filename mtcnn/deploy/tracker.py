import torch
import uuid
import numpy as np
from collections import defaultdict
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
        self.iou_thres = iou_thres

        self.reset()
        self.image_cache = defaultdict(list)

        # Set params for detector. This can be modefied by "set_detect_params"
        self.default_detect_params = dict(
            threshold=[0.6, 0.7, 0.9], 
            factor=0.7, 
            minsize=12, 
            nms_threshold=[0.7, 0.7, 0.3]
        )

    def set_detect_params(self, **kwargs):
        self.default_detect_params.update(kwargs)
        

    def track(self, frame):
        if self.cur_count % self.min_interval == 0 or len(self.boxes_cache) == 0:
            boxes, _ = self.detector.detect(frame, **self.default_detect_params)

            update_cache = {}
            if boxes.shape[0] != 0:
                
                for i, b in enumerate(self.boxes_cache):
                    ovr = func.IoU(b, boxes)
                    max_ovr = ovr.max()
                    max_index = ovr.argmax()
                    if ovr >= self.iou_thres:
                        update_cache[max_index] = self.label_cache[i]

            self.reset()
            for b in boxes:
                self.label_cache.append(uuid.uuid1())
                self.interval_cache.append(0)
                self.boxes_cache.append(b)

            for k, v in update_cache.items():
                self.label_cache[k] = v

            for b, label in zip(self.boxes_cache, self.label_cache):
                self.image_cache[label].append(frame[b[1]: b[3], b[0]: b[2]])
            
            self.cur_count += 1
        
        else:
            boxes = self.detector.stage_three(frame, torch.stack(self.boxes_cache), self.default_detect_params)
            update_cache = {}
            for b in boxes:
                ovr = func.IoU(b, self.boxes_cache)
                max_index = ovr.argmax()
                update_cache[max_index] = b

            revome_list = []
            for i, b in enumerate(self.boxes_cache):
                if i in update_cache:
                    self.boxes_cache[i] = update_cache[i]
                    self.image_cache[self.label_cache[i]].append(frame[b[1]: b[3], b[0]: b[2]])
                else:
                    if self.interval_cache[i] <= self.min_interval:
                        self.interval_cache[i] += 1
                    else:
                        revome_list.append(i)

            for i in revome_list:
                self.label_cache.pop(i)
                self.interval_cache.pop(i)
                self.boxes_cache.pop(i)

            self.cur_count += 1


    def reset(self):
        self.cur_count = 0
        self.boxes_cache = []
        self.label_cache = []
        self.interval_cache = []

    def get_cache(self):
        """
        Get the images in image_cache and clear the images in cache.
        """
        tmp = self.image_cache
        self.image_cache = defaultdict(list)
        return tmp

        
