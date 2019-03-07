import math
import cv2
import torch
import numpy as np
import time

import mtcnn.utils.functional as func


def _no_grad(func):

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


class BatchImageDetector(object):

    def __init__(self, pnet, rnet, onet, device='cpu'):

        self.device = torch.device(device)

        self.pnet = pnet.to(self.device)
        self.rnet = rnet.to(self.device)
        self.onet = onet.to(self.device)

        self.onet.eval()  # Onet has dropout layer.

    def _preprocess(self, imgs):

        # Convert image from rgb to bgr for Compatible with original caffe model.
        tmp = []
        for i, img in enumerate(imgs):
            tmp.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        imgs = np.stack(tmp)
        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = torch.FloatTensor(imgs).to(self.device)
        imgs = func.imnormalize(imgs)

        return imgs

    def detect(self, imgs, threshold=[0.6, 0.7, 0.9], factor=0.7, minsize=12, nms_threshold=[0.7, 0.7, 0.3]):

        imgs = self._preprocess(imgs)
        stage_one_boxes = self.stage_one(
            imgs, threshold[0], factor, minsize, nms_threshold[0])
        stage_two_boxes = self.stage_two(
            imgs, stage_one_boxes, threshold[1], nms_threshold[1])
        stage_three_boxes, landmarks = self.stage_three(
            imgs, stage_two_boxes, threshold[2], nms_threshold[2])

        return stage_three_boxes, landmarks

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a FloatTensor of shape [n, 2, h, w].
            offsets: a FloatTensor array of shape [n, 4, h, w].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            boxes: LongTensor with shape [x, 4].
            score: FloatTensor with shape [x].
            offses: FloatTensor with shape [x, 4]
            img_label: IntTensor with shape [x]
        """
        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # extract positive probability and resize it as [n, m] dim tensor.
        probs = probs[:, 1, :, :]

        # indices of boxes where there is probably a face
        mask = probs > threshold
        inds = mask.nonzero()

        if inds.shape[0] == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device), \
                torch.empty(0, dtype=torch.float32, device=self.device), \
                torch.empty(0, dtype=torch.float32, device=self.device), \
                torch.empty(0, dtype=torch.int32, device=self.device)

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[inds[:, 0], i, inds[:, 1], inds[:, 2]]
                              for i in range(4)]

        offsets = torch.stack([tx1, ty1, tx2, ty2], 1)
        score = probs[inds[:, 0], inds[:, 1], inds[:, 2]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = torch.stack([
            stride*inds[:, -1] + 1.0,
            stride*inds[:, -2] + 1.0,
            stride*inds[:, -1] + 1.0 + cell_size,
            (stride*inds[:, -2] + 1.0 + cell_size),
        ], 0).transpose(0, 1).float()

        bounding_boxes = torch.round(bounding_boxes / scale).int()
        return bounding_boxes, score, offsets, inds[:, 0].int()

    def _calibrate_box(self, bboxes, offsets):
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.

        Arguments:
            bboxes: a IntTensor of shape [n, 4].
            offsets: a IntTensor of shape [n, 4].

        Returns:
            a IntTensor of shape [n, 4].
        """
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = torch.unsqueeze(w, 1)
        h = torch.unsqueeze(h, 1)

        # this is what happening here:
        # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
        # x1_true = x1 + tx1*w
        # y1_true = y1 + ty1*h
        # x2_true = x2 + tx2*w
        # y2_true = y2 + ty2*h
        # below is just more compact form of this

        # are offsets always such that
        # x1 < x2 and y1 < y2 ?

        translation = torch.cat([w, h, w, h], 1).float() * offsets
        bboxes += torch.round(translation).int()
        return bboxes

    def _convert_to_square(self, bboxes):
        """Convert bounding boxes to a square form.

        Arguments:
            bboxes: a IntTensor of shape [n, 4].

        Returns:
            a IntTensor of shape [n, 4],
                squared bounding boxes.
        """

        square_bboxes = torch.zeros_like(bboxes, device=self.device, dtype=torch.float32)
        x1, y1, x2, y2 = [bboxes[:, i].float() for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = torch.max(h, w)
        square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
        square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0

        square_bboxes = torch.ceil(square_bboxes + 1).int()
        return square_bboxes

    def _refine_boxes(self, bboxes, w, h):

        bboxes = torch.max(torch.zeros_like(bboxes, device=self.device), bboxes)
        sizes = torch.IntTensor([[h, w, h, w]] * bboxes.shape[0]).to(self.device)
        bboxes = torch.min(bboxes, sizes)
        return bboxes

    def _calibrate_landmarks(self, bboxes, landmarks, align=False):
        """Compute the face landmarks coordinates
        
        Args:
            bboxes (torch.IntTensor): bounding boxes of shape [n, 4]
            landmarks (torch.floatTensor): landmarks regression output of shape [n, 10]
            align (bool, optional): Defaults to False. If "False", return the coordinates related to the origin image. Else, return the coordinates related to alinged faces.
        
        Returns:
            torch.IntTensor: face landmarks coordinates of shape [n, 10]
        """

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = torch.unsqueeze(w, 1)
        h = torch.unsqueeze(h, 1)

        translation = torch.cat([w]*5 + [h]* 5, 1).float() * landmarks
        if align:
            landmarks = torch.ceil(translation).int()
        else:
            landmarks = torch.stack([bboxes[:, 0]] * 5 + [bboxes[:, 1]] * 5, 1) + torch.round(translation).int()
        return landmarks

    @_no_grad
    def stage_one(self, imgs, threshold, factor, minsize, nms_threshold):
        """Stage one of mtcnn detection.
        
        Args:
            imgs (torch.FloatTensro): Output of "_preprocess" method.
            threshold (float): The minimum probability of reserve bounding boxes.
            factor (float): Image pyramid scaling ratio.
            minsize (int): The minimum size of reserve bounding boxes.
            nms_threshold (float): retain boxes that satisfy overlap <= thresh
        
        Returns:
            torch.IntTensor: All bounding boxes with image label output by stage one detection. [n, 5]
        """

        width = imgs.shape[-2]
        height = imgs.shape[-1]
        num_img = imgs.shape[0]

        # Compute valid scales
        scales = []
        cur_width = width
        cur_height = height
        cur_factor = 1
        while cur_width >= 12 and cur_height >= 12:
            if 12 / cur_factor >= minsize:  # Ignore boxes that smaller than minsize

                w = cur_width
                h = cur_height
                scales.append((w, h, cur_factor))

            cur_factor *= factor
            cur_width = math.ceil(cur_width * factor)
            cur_height = math.ceil(cur_height * factor)

        # Get candidate boxesi ph
        candidate_boxes = torch.empty(0, dtype=torch.int32, device=self.device)
        candidate_scores = torch.empty(0, device=self.device)
        candidate_offsets = torch.empty(
            0, dtype=torch.float32, device=self.device)
        all_img_labels = torch.empty(0, dtype=torch.int32, device=self.device)
        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(
                imgs, size=(w, h), mode='bilinear')
            p_distribution, box_regs, _ = self.pnet(resize_img)

            candidate, scores, offsets, img_labels = self._generate_bboxes(
                p_distribution, box_regs, f, threshold)

            candidate_boxes = torch.cat([candidate_boxes, candidate])
            candidate_scores = torch.cat([candidate_scores, scores])
            candidate_offsets = torch.cat([candidate_offsets, offsets])
            all_img_labels = torch.cat([all_img_labels, img_labels])

        
        if candidate_boxes.shape[0] != 0:
            candidate_boxes = self._calibrate_box(
                candidate_boxes, candidate_offsets)
            candidate_boxes = self._convert_to_square(candidate_boxes)
            candidate_boxes = self._refine_boxes(
                candidate_boxes, width, height)
            
            final_boxes = torch.empty(0, dtype=torch.int32, device=self.device)
            final_img_labels = torch.empty(0, dtype=torch.int32, device=self.device)
            for i in range(num_img):
                mask = all_img_labels == i
                keep = func.nms(candidate_boxes[mask].cpu().numpy(),
                            candidate_scores[mask].cpu().numpy(), nms_threshold)
                final_boxes = torch.cat([final_boxes, candidate_boxes[mask][keep]])
                final_img_labels = torch.cat([final_img_labels, all_img_labels[mask][keep]])

            return torch.cat([final_boxes, final_img_labels.unsqueeze(1 )], -1)
        else:
            return candidate_boxes


    @_no_grad
    def stage_two(self, imgs, boxes, threshold, nms_threshold):

        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        width = imgs.shape[2]
        height = imgs.shape[3]
        lablels = boxes[:, -1]
        boxes = boxes[:, :4]

        num_img = imgs.shape[0]

        # get candidate faces
        candidate_faces = list()

        for box, label in zip(boxes, lablels):
            im = imgs[label, :, box[1]: box[3], box[0]: box[2]].unsqueeze(0)
            im = torch.nn.functional.interpolate(
                im, size=(24, 24), mode='bilinear')
            candidate_faces.append(im)
        
        candidate_faces = torch.cat(candidate_faces, 0)

        # rnet forward pass
        p_distribution, box_regs, _ = self.rnet(candidate_faces)

        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]
        labels = lablels[mask]

        if boxes.shape[0] != 0:
            boxes = self._calibrate_box(boxes, box_regs)
            boxes = self._convert_to_square(boxes)
            boxes = self._refine_boxes(boxes, width, height)

            final_boxes = torch.empty(0, dtype=torch.int32, device=self.device)
            final_img_labels = torch.empty(0, dtype=torch.int32, device=self.device)
            for i in range(num_img):
                mask = labels == i
                keep = func.nms(boxes[mask].cpu().numpy(),
                            scores[mask].cpu().numpy(), nms_threshold)
                final_boxes = torch.cat([final_boxes, boxes[mask][keep]])
                final_img_labels = torch.cat([final_img_labels, labels[mask][keep]])

            return torch.cat([final_boxes, final_img_labels.unsqueeze(1 )], -1)

        else:

            return boxes


    @_no_grad
    def stage_three(self, imgs, boxes, threshold, nms_threshold):
        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes, torch.empty(0, device=self.device, dtype=torch.int32)

        width = imgs.shape[2]
        height = imgs.shape[3]

        labels = boxes[:, -1]
        boxes = boxes[:, :4]

        num_img = imgs.shape[0]

        # get candidate faces
        candidate_faces = list()

        for box, label in zip(boxes, labels):
            im = imgs[label, :, box[1]: box[3], box[0]: box[2]].unsqueeze(0)
            im = torch.nn.functional.interpolate(
                im, size=(48, 48), mode='bilinear')
            candidate_faces.append(im)
        
        candidate_faces = torch.cat(candidate_faces, 0)

        p_distribution, box_regs, landmarks = self.onet(candidate_faces)

        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]
        labels =labels[mask]

        if boxes.shape[0] != 0:

            # compute face landmark points
            landmarks = self._calibrate_landmarks(boxes, landmarks)
            landmarks = torch.stack([landmarks[:, :5], landmarks[:, 5:10]], 2)

            boxes = self._calibrate_box(boxes, box_regs)
            boxes = self._refine_boxes(boxes, width, height)

            final_boxes = torch.empty(0, dtype=torch.int32, device=self.device)
            final_img_labels = torch.empty(0, dtype=torch.int32, device=self.device)
            final_landmarks = torch.empty(0, dtype=torch.int32, device=self.device)
            for i in range(num_img):
                
                # nms
                mask = labels == i
                keep = func.nms(boxes[mask].cpu().numpy(),
                            scores[mask].cpu().numpy(), nms_threshold)
                final_boxes = torch.cat([final_boxes, boxes[mask][keep]])
                final_img_labels = torch.cat([final_img_labels, labels[mask][keep]])

                # compute face landmark points
                landm = landmarks  [mask][keep]
                final_landmarks = torch.cat([final_landmarks, landm])

            return torch.cat([final_boxes, final_img_labels.unsqueeze(1 )], -1), final_landmarks

        else:
            return boxes, landmarks