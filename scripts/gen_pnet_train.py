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
landmarks_eval_meta = landmarks_dataset.get_val_meta()

print("Start generate landmarks training data for pnet.")
gl.gen_landmark_data(landmarks_meta, 12, args.output_folder, argument=False, suffix='pnet')
print("Done")
print("Start generate landmarks eval data for pnet.")
gl.gen_landmark_data(landmarks_eval_meta, 12, args.output_folder, argument=False, suffix='pnet_eval')
print("Done")

detection_dataset = get_by_name(args.detection_dataset)
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
gptd.generate_training_data_for_pnet(detection_meta, output_folder=args.output_folder, suffix='pnet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
gptd.generate_training_data_for_pnet(detection_eval_meta, output_folder=args.output_folder, suffix='pnet_eval')
print("Done")
