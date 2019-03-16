import os
import sys
import cv2
import numpy as np
import argparse
import mtcnn


parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--video_path', type=str, help="Read from video.")
parser.add_argument('--output_folder', type=str, help="Save the tracking result.")
parser.add_argument('--saved_path', type=str, default=None,
                    help="If set, Save as video. Or show it on screen.")
parser.add_argument("--minsize", type=int, default=24,
                    help="Min size of faces you want to detect. Larger number will speed up detect method.")
parser.add_argument('--min_interval', type=int, default=3, help="See FaceTracker.")
parser.add_argument("--device", type=str, default='cpu',
                    help="Target device to process video.")

args = parser.parse_args()

pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')
detector = mtcnn.FaceDetector(pnet, rnet, onet, device=args.device)
tracker = mtcnn.FaceTracker(detector, min_interval=args.min_interval)
tracker.set_detect_params(minsize=args.minsize)

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

    boxes, landmarks = tracker.track(image)

    image = mtcnn.utils.draw.draw_boxes2(image, boxes)
    # image = mtcnn.utils.draw.draw_boxes2(image, tracker.boxes_cache)
    image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)

    if args.saved_path is None:
        cv2.imshow("asdfas", image)
        cv2.waitKey(1)
    else:
        out.write(image)

for k, v in tracker.get_cache().items():
    if len(v) < 5:
        continue
    saved_dir = os.path.join(args.output_folder, str(k))
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    for i, img in enumerate(v):
        cv2.imwrite(os.path.join(saved_dir, "%d.jpg" % i), img)


if args.saved_path is not None:
    out.release()
