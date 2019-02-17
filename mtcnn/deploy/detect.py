import math
import cv2
import torch
import time

import mtcnn.utils.functional as func

def _no_grad(func):

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


class FaceDetector(object):

    def __init__(self, pnet, rnet, onet, device='cpu'):
        
        self.device = torch.device(device)
        
        self.pnet = pnet.to(self.device)
        self.rnet = rnet.to(self.device)
        self.onet = onet.to(self.device)

        self.onet.eval()  # Onet has dropout layer.

    def _preprocess(self, img):

        if isinstance(img, str):
            img = cv2.imread(img)

        # Convert image from rgb to bgr for Compatible with original caffe model.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img).to(self.device)
        img = func.imnormalize(img)
        img = img.unsqueeze(0)

        return img

    def detect(self, img, threshold=[0.6, 0.7, 0.7], factor=0.7, minsize=12):

        img = self._preprocess(img)
        stage_one_boxes = self.stage_one(img, threshold[0], factor, minsize)
        stage_two_boxes = self.stage_two(img, stage_one_boxes, threshold[1])
        stage_three_boxes, landmarks = self.stage_three(
            img, stage_two_boxes, threshold[2])

        return stage_three_boxes, landmarks

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a FloatTensor of shape [1, 2, n, m].
            offsets: a FloatTensor array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            boxes: LongTensor with shape [x, 4].
            score: FloatTensor with shape [x].
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # extract positive probability and resize it as [n, m] dim tensor.
        probs = probs[0, 1, :, :]

        # indices of boxes where there is probably a face
        inds = (probs > threshold).nonzero()

        if inds.shape[0] == 0:
            return torch.empty((0, 4), dtype=torch.int32, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device), torch.empty((0, 4), dtype=torch.float32, device=self.device)

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[:, 0], inds[:, 1]]
                              for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = torch.cat([tx1, ty1, tx2, ty2], 0)
        score = probs[inds[:, 0], inds[:, 1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = torch.stack([
            stride*inds[:, 1] + 1.0,
            stride*inds[:, 0] + 1.0,
            stride*inds[:, 1] + 1.0 + cell_size,
            (stride*inds[:, 0] + 1.0 + cell_size),
        ], 0).transpose(0, 1).float()

        bounding_boxes = torch.round(bounding_boxes / scale).int()
        return bounding_boxes, score, offsets

    def calibrate_box(bboxes, offsets):
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.

        Arguments:
            bboxes: a float numpy array of shape [n, 5].
            offsets: a float numpy array of shape [n, 4].

        Returns:
            a float numpy array of shape [n, 5].
        """
        pass

    @_no_grad
    def stage_one(self, img, threshold, factor, minsize):
        width = img.shape[2]
        height = img.shape[3]

        # Compute valid scales
        scales = []
        cur_width = width
        cur_height = height
        cur_factor = 1
        while cur_width >= minsize and cur_height >= minsize:
            # ensure width and height are even
            w = cur_width
            h = cur_height
            scales.append((w, h, cur_factor))

            cur_factor *= factor
            cur_width = math.ceil(cur_width * factor)
            cur_height = math.ceil(cur_height * factor)

        # Get candidate boxesi ph
        candidate_boxes = torch.empty((0, 4), dtype=torch.int32, device=self.device)
        candidate_scores = torch.empty((0), device=self.device)
        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(
                img, size=(w, h), mode='bilinear')
            p_distribution, box_regs, _ = self.pnet(resize_img)

            candidate, scores, _ = self._generate_bboxes(
                p_distribution, box_regs, f, threshold)

            candidate_boxes = torch.cat([candidate_boxes, candidate])
            candidate_scores = torch.cat([candidate_scores, scores])

        # nms
        if candidate_boxes.shape[0] != 0:
            keep = func.nms(candidate_boxes.cpu().numpy(), candidate_scores.cpu().numpy(), 0.3)
            return candidate_boxes[keep]
        else:
            return candidate_boxes

    @_no_grad
    def stage_two(self, img, boxes, threshold):

        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        width = img.shape[2]
        height = img.shape[3]

        # get candidate faces
        candidate_faces = torch.empty((0, 3, 24, 24))

        for box in boxes:
            im = img[:, :, box[0]: box[2], box[1]: box[3]]
            im = torch.nn.functional.interpolate(
                im, size=(24, 24), mode='bilinear')
            candidate_faces = torch.cat([candidate_faces, im])

        # rnet forward pass
        p_distribution, box_regs, _ = self.rnet(candidate_faces)

        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]

        # compute offsets
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        weights = torch.stack([w, h, w, h])

        offsets = box_regs * weights
        candidate = boxes + offsets

        # nms
        keep = func.nms(candidate, scores, 0.3)
        return candidate[keep]

    @_no_grad
    def stage_three(self, img, boxes, threshold):
        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        # get candidate faces
        candidate_faces = torch.empty((0, 3, 48, 48))

        for box in boxes:
            im = img[:, :, box[0]: box[2], box[1]: box[3]]
            im = torch.nn.functional.interpolate(
                im, size=(48, 48), mode='bilinear')
            candidate_faces = torch.cat([candidate_faces, im])

        p_distribution, box_regs, landmarks = self.rnet(candidate_faces)
        candidate, scores, landmarks = self._filter_boxes(
            boxes, p_distribution, box_regs, threshold, landmarks)

        # nms
        keep = func.nms(candidate, scores, 0.3)
        return candidate[keep], landmarks[keep]
