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
parser.add_argument("--minsize", type=int, default=24,
                    help="Min size of faces you want to detect. Larger number will speed up detect method.")
parser.add_argument("--device", type=str, default='cpu',
                    help="Target device to process video.")
parser.add_argument("--model_dir", type=str, default="", help="There are pre-trained pnet, rnet, onet in this folder.")

args = parser.parse_args()

if args.model_dir == '':
    pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')
else:
    pnet, rnet, onet = mtcnn.get_net(args.model_dir)

detector = mtcnn.FaceDetector(pnet, rnet, onet, device=args.device)

fourcc = cv2.VideoWriter_fourcc(*"XVID")

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if args.saved_path is not None:
    out = cv2.VideoWriter(args.saved_path, fourcc, fps, size)

while True:

    res, image = cap.read()
    if not res:
        break

    boxes, landmarks = detector.detect(image, minsize=args.minsize)

    image = mtcnn.utils.draw.draw_boxes2(image, boxes)
    image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)

    if args.saved_path is None:
        cv2.imshow("asdfas", image)
        cv2.waitKey(1)
    else:
        out.write(image)

if args.saved_path is not None:
    out.release()
