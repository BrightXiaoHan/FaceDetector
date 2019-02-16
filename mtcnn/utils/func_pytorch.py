import torch

def nms(dets, scores, thresh, mode="Union"):
    """
    Nms algorithm pytorch implementation 

    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    
    Arguments:
        dets {torch.LongTensor} -- [[x1, y1, x2, y2]]
        scores {torch.Tensor} -- score of each boxes
        thresh {float} -- retain overlap <= thresh
    
    Keyword Arguments:
        mode {str} -- Union or Minimum (default: {"Union"})
    
    Returns:
        torch.Tensor -- Remain boxes.
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    device = dets.device

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = torch.sort(scores)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        order = order[1:]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order])
        yy1 = torch.max(y1[i], y1[order])
        xx2 = torch.min(x2[i], x2[order])
        yy2 = torch.min(y2[i], y2[order])

        w = torch.max(torch.zeros(1, dtype=torch.int32, device=device), xx2 - xx1 + 1)
        h = torch.max(torch.zeros(1, dtype=torch.int32, device=device), yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter.float() / (areas[i] + areas[order] - inter).float()
        elif mode == "Minimum":
            ovr = inter.float() / torch.min(areas[i], areas[order]).float()

        order = order[ovr <= thresh]

    keep = torch.stack(keep)

    return keep

def imnormalize(img):
    """
    Normalize pixel value from (0, 255) to (-1, 1) 
    """

    img = (img - 127.5) * 0.0078125
    return img