import torch
import numpy as np
from mtcnn.utils.align_trans import get_reference_facial_points, warp_and_crop_face

refrence = get_reference_facial_points(default_square= True)

def align_multi(img, boxes, landmarks, crop_size=(112, 112)):
    """Align muti-faces in a image
    
    Args:
        img (np.ndarray or torch.Tensor): Image matrix returned by cv2.imread()
        boxes (np.ndarray or torch.IntTensor): Bounding boxes with shape [n, 4]
        landmarks (np.ndarray or torch.IntTensor): Facial landmarks points with shape [n, 5, 2] 
    """

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        
    faces = []
    for landmark in landmarks:
        warped_face = warp_and_crop_face(img, landmark, refrence, crop_size=crop_size)
        faces.append(warped_face)
    if len(faces) > 0:
        faces = np.stack(faces)
    return boxes, faces

def filter_side_face(boxes, landmarks):
    """Mask all side face judged through facial landmark points.
    
    Args:
        boxes (torch.IntTensor): Bounding boxes with shape [n, 4]
        landmarks (or torch.IntTensor): Facial landmarks points with shape [n, 5, 2]
    
    Returns:
        torch.Tensor: Tensor mask.
    """
    mid = (boxes[:, 2] + boxes[:, 0]).float() / 2
    mask =  (landmarks[:, 0, 0].float() - mid) * (landmarks[:, 1, 0].float() - mid) <= 0 

    return mask

    


