import os
import sys
import cv2
import numpy as np
import argparse
import mtcnn


parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--video_path', type=str,
                    default=None, help="Read from video.")
parser.add_argument('--saved_path', type=str, default=None,
                    help="If set, Save as video. Or show it on screen.")

args = parser.parse_args()

pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

fourcc = cv2.VideoWriter_fourcc(*"XVID")

cap = cv2.VideoCapture("./output/108.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if args.saved_path is not None:
    out = cv2.VideoWriter(args.saved_path, fourcc, fps, size)

while True:

    res, image = cap.read()
    if not res:
        break

    boxes, landmarks = detector.detect(image)
    
    image = mtcnn.utils.draw.draw_boxes2(image, boxes)
    image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)

    if args.saved_path is None:
        cv2.imshow("asdfas", image)
        cv2.waitKey(1)
    else:
        out.write(image)

if args.saved_path is not None:
    out.release()
