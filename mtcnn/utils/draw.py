"""
Some useful function for visualize bounding box and face landmarks.
"""
import cv2


def draw_boxes(img, boxes, color=(255, 0, 0)):
    """Draw bounding boxes on original image.

    Args:
        img (np.array): image matrix returned by cv2.imread
        boxes (list): Each item contrains a bounding box (x1, y1, w, h). (List like objects are all ok. "np.array" for example.)
    """
    for box in boxes:
        # Default draw red box on it.
        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color)
    
    return img

def draw_boxes2(img, boxes, color=(255, 0, 0)):
    """Draw bounding boxes on original image.

    Args:
        img (np.array): image matrix returned by cv2.imread
        boxes (list): Each item contrains a bounding box (x1, y1, x2, y2). (List like objects are all ok. "np.array" for example.)
    """
    for box in boxes:
        # Default draw red box on it.
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)
    
    return img

def crop(img, boxes, landmarks=None):
    """Cut region from origin image
    
    Args:
        img (np.array): image matrix returned by cv2.imread
        boxes (list): Each item contrains a bounding box (x1, y1, x2, y2). (List like objects are all ok. "np.array" for example.)
    """
    if landmarks is not None:
        img = img.copy()
        batch_draw_landmarks(img, landmarks)

    img_list = []
    for box in boxes:
        i = img[box[1]:box[3], box[0]:box[2]]
        img_list.append(i)

    return img_list


def draw_landmarks(img, landmarks, color=(0, 0, 255)):
    """Draw points on original image.
    
    Args:
        img (np.array): image matrix returned by cv2.imread
        landmarks (list): Each item contains a point coordinates (x, y). (List like objects are all ok. "np.array" for example.)
    """
    for point in landmarks:
        
        # Default draw blue point on it
        cv2.circle(img, tuple(point), 2, color)

    
    return img

def batch_draw_landmarks(img, batch_landmarks, color=(0, 0, 255)):

    for landmarks in batch_landmarks:
        draw_landmarks(img, landmarks, color)
    
    return img
