import cv2
import torch

class FaceDetector(object):

    def __init__(self, pnet, onet, rnet, device='cpu'):
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet

        self.device = device
        [net.to(device) for net in [pnet, rnet, onet]]

    def detect(self, img, threshold=[0.6, 0.7, 0.7], factor=0.7, minsize=12):

        img = self._preprocess(img)
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

        candidate_boxes = []
        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(img, size=(w, h), mode='bilinear')
            boxes = self._generate_boxes(w, h)
            labels, box_regs = self.pnet(resize_img)
            scores = labels[:, 1]
            mask = (scores >= threshold[0])
            boxes = boxes[mask]
            box_regs = box_regs[mask]
            final_boxes = boxes - (box_regs * 12)
            
            # Set value zero for negative value
            final_boxes[final_boxes<0] = 0
            mask_x2gtx1 = final_boxes[:, 2] > final_boxes[:, 0]
            mask_y2gtx1 = final_boxes[:, 3] > final_boxes[:, 1]
            mask_error = (mask_x2gtx1 + mask_y2gtx1) == 0
            candidate = final_boxes[mask_error]/f
            candidate = torch.ceil(candidate).int()
            candidate_boxes.append(candidate)

        candidate_boxes = torch.cat(candidate_boxes, 0)

        # The second stage

    def _preprocess(self, img):

        if isinstance(img, str):
            img = cv2.imread(img)
        
        img = torch.FloatTensor(img)
        img = img.transpose(2, 0, 1)
        img = (img - 127.5) * 0.0078125
        img = img.unsqueeze(0)

        return img

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

    def _nms(self, dets, scores, thresh, mode="Union"):
        """
        greedily select boxes with high confidence
        keep boxes overlap <= thresh
        rule out overlap > thresh
        
        Arguments:
            dets {torch.Tensor} -- [[x1, y1, x2, y2]]
            scores {torch.Tensor} -- score of each boxes
            thresh {float} -- retain overlap <= thresh
        
        Keyword Arguments:
            mode {str} -- [description] (default: {"Union"})
        
        Returns:
            torch.Tensor -- Remain boxes.
        """

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]


        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = torch.sort(scores)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if mode == "Union":
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == "Minimum":
                ovr = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return np.array(keep)
            