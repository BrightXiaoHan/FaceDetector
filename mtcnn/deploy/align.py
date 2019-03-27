import torch
from mtcnn.utils.align_trans import get_reference_facial_points, warp_and_crop_face

refrence = get_reference_facial_points(default_square= True)

def align_multi(img, boxes, landmarks, crop_size=(112, 112)):
    """Align muti-faces in a image
    
    Args:
        img (np.array or torch.Tensor): Image matrix returned by cv2.imread()
        boxes (torch.IntTensor): Bounding boxes with shape [n, 4]
        landmarks (torch.IntTensor): Fatial landmarks points with shape [n, 5, 2] 
    """

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        
    faces = []
    for landmark in landmarks:
        warped_face = warp_and_crop_face(img, landmark, refrence, crop_size=crop_size)
        faces.append(warped_face)
    return boxes, faces


    


