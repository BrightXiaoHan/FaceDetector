import argparse

import mtcnn.train.gen_pnet_train as gptd
import mtcnn.train.gen_landmark as gl
from mtcnn.datasets import get_by_name


parser = argparse.ArgumentParser(
    description='Generate training data for pnet.')
parser.add_argument('-o', dest="output_folder", default="output/data_train", type=str, help="Folder to save training data for pnet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")
parser.add_argument("-l", type=str, dest="landmarks_dataset", default="CelebA",
                    help="Landmark localization dataset name.")
args = parser.parse_args()

landmarks_dataset = get_by_name(args.landmarks_dataset)
landmarks_meta = landmarks_dataset.get_train_meta()
gl.gen_landmark_data(landmarks_meta, 12, args.output_folder, argument=False, suffix='pnet')

detection_dataset = get_by_name(args.detection_dataset)
detection_meta = detection_dataset.get_train_meta()
gptd.generate_training_data_for_pnet(detection_meta, output_folder=args.output_folder)

