import cv2
import torch
import numpy as np

from mtcnn.utils.nms.cpu_nms import cpu_nms
from mtcnn.utils.nms.gpu_nms import gpu_nms

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr


def nms(dets, scores, thresh, device="cpu"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cpu':
        ret = cpu_nms(dets.astype(np.float32), scores.astype(np.float32), thresh)
    
    else:
        dets = np.concatenate([dets.astype(np.float32), scores.astype(np.float32).reshape(-1, 1)], 1)
        ret = gpu_nms(dets , thresh, device_id=device.index)

    return ret

    
def imnormalize(img):
    """
    Normalize pixel value from (0, 255) to (-1, 1) 
    """

    img = (img - 127.5) * 0.0078125
    return img

def iou_torch(box, boxes):
    """Compute IoU between detect box and gt boxes
    
    Args:
        box (torch.IntTensor): shape (4, )
        boxes (torch.IntTensor): shape (n, 4)
    
    Returns:
        torch.FloatTensor: [description]
    """

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = xx2 - xx1 + 1
    h = yy2 - yy1 + 1
    w = torch.max(torch.zeros_like(w), w)
    h = torch.max(torch.zeros_like(h), h)

    inter = w * h
    ovr = inter.float() / (box_area + area - inter).float()
    
    return ovr
