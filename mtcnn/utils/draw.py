"""
Some useful function for visualize bounding box and face landmarks.
"""
import cv2


def draw_boxes(img, boxes):
    """Draw bounding boxes on original image.

    Args:
        img (np.array): image matrix returned by cv2.imread
        boxes (list): Each item contrains a bounding box (x1, y1, w, h). (List like objects are all ok. "np.array" for example.)
    """
    for box in boxes:
        # Default draw red box on it.
        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0))
    
    return img


def draw_landmarks(img, landmarks):
    """Draw points on original image.
    
    Args:
        img (np.array): image matrix returned by cv2.imread
        landmarks (list): Each item contains a point coordinates (x, y). (List like objects are all ok. "np.array" for example.)
    """
    for point in landmarks:
        
        # Default draw blue point on it
        cv2.circle(img, tuple(point), 2, (0, 0, 255))

    
    return img
