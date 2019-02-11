import cv2
import torch

import mtcnn.utils.func_pytorch as func


class FaceDetector(object):

    def __init__(self, pnet, onet, rnet, device='cpu'):
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet

        self.device = device
        [net.to(device) for net in [pnet, rnet, onet]]

    @staticmethod
    def _preprocess(img):

        if isinstance(img, str):
            img = cv2.imread(img)

        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img)  
        img = func.imnormalize(img)
        img = img.unsqueeze(0)

        return img

    def detect(self, img, threshold=[0.6, 0.7, 0.7], factor=0.7, minsize=12):

        img = self._preprocess(img)
        stage_one_boxes = self.stage_one(img, threshold[0], factor, minsize)
        stage_two_boxes = self.stage_two(img, stage_one_boxes, threshold[1])
        stage_three_boxes, landmarks = self.stage_three(img, stage_two_boxes, threshold[2])

        return stage_three_boxes, landmarks
        
    def _generate_boxes(self, w, h, stride=2):
        x1, y1, x2, y2 = 0, 0, 12, 12
        boxes = []
        while x2 <= w:
            while y2 <= h:
                boxes.append((x1, y1, x2, y2))
                y2 += stride
                y1 += stride
            x1 += stride
            x2 += stride
        boxes = torch.LongTensor(boxes).to(self.device)
        return boxes

    @staticmethod
    def _filter_boxes(boxes, p_distribution, box_regs, threshold, landmarks=None):
        scores = p_distribution[:, 1]

        # Select positive example
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]

        # Compute actually coordinate
        final_boxes = boxes - (box_regs * 12)

        # Set value zero for negative value
        final_boxes[final_boxes < 0] = 0

        # filter boxes that x2 <= x1 or y2 <= y1
        mask_x2gtx1 = final_boxes[:, 2] > final_boxes[:, 0]
        mask_y2gtx1 = final_boxes[:, 3] > final_boxes[:, 1]
        mask_error = (mask_x2gtx1 + mask_y2gtx1) == 0

        candidate = final_boxes[mask_error]
        scores = scores[mask_error]

        if landmarks is not None:
            landmarks = landmarks[mask][mask_error]
            return candidate, scores, landmarks

        return candidate, scores

    def stage_one(self, img, threshold, factor, minsize):
        width = img.shape[2]
        height = img.shape[3]

        # Compute valid scales
        scales = []
        cur_width = width
        cur_height = height
        cur_factor = factor
        while cur_width >= minsize and cur_height >= minsize:
            scales.append((cur_width, cur_height, cur_factor))
            cur_width *= factor
            cur_height *= factor
            factor *= factor

        candidate_boxes = torch.empty((0, 4), dtype=torch.int64)
        candidate_scores = torch.empty((0))
        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(
                img, size=(w, h), mode='bilinear')
            boxes = self._generate_boxes(w, h)
            p_distribution, box_regs, _ = self.pnet(resize_img)

            candidate, scores = self._filter_boxes(
                boxes, p_distribution, box_regs, threshold)

            candidate = torch.ceil(candidate.float()/f).int()

            candidate_boxes = torch.cat([candidate_boxes, candidate])
            candidate_scores = torch.cat([candidate_scores, scores])

        # nms
        keep = func.nms(candidate_boxes, candidate_scores, 0.3)
        return candidate_boxes[keep]

    def stage_two(self, img, boxes, threshold):

        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        # get candidate faces
        candidate_faces = torch.empty((0, 3, 24, 24))

        for box in boxes:
            im = img[:, :, box[0]: box[2], box[1], box[3]]
            im = torch.nn.functional.interpolate(
                im, size=(24, 24), mode='bilinear')
            candidate_faces = torch.cat([candidate_faces, im])

        p_distribution, box_regs = self.rnet(candidate_faces)

        candidate, scores, _ = self._filter_boxes(
            boxes, p_distribution, box_regs, threshold)

        # nms
        keep = func.nms(candidate, scores, 0.3)
        return candidate[keep]

    def stage_three(self, img, boxes, threshold):
        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        # get candidate faces
        candidate_faces = torch.empty((0, 3, 48, 48))

        for box in boxes:
            im = img[:, :, box[0]: box[2], box[1], box[3]]
            im = torch.nn.functional.interpolate(
                im, size=(48, 48), mode='bilinear')
            candidate_faces = torch.cat([candidate_faces, im])

        p_distribution, box_regs, landmarks = self.rnet(candidate_faces)
        candidate, scores, landmarks = self._filter_boxes(
            boxes, p_distribution, box_regs, threshold, landmarks)

        # nms
        keep = func.nms(candidate, scores, 0.3)
        return candidate[keep], landmarks[keep]