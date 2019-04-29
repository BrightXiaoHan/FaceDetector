import argparse
import torch

import mtcnn.train.gen_onet_train as gotd
import mtcnn.train.gen_landmark as gl
from mtcnn.datasets import get_by_name
from mtcnn.network.mtcnn_pytorch import PNet, RNet


parser = argparse.ArgumentParser(
    description='Generate training data for onet.')
parser.add_argument('-pm', type=str, dest="pnet_model_file", help="Pre-trained pnet model file.")
parser.add_argument('-rm', type=str, dest="rnet_model_file", help="Pre-trained rnet model file.")
parser.add_argument('-o', dest="output_folder", default="output/data_train", type=str, help="Folder to save training data for onet.")
parser.add_argument("-d", dest="detection_dataset",type=str, default="WiderFace",
                    help="Face Detection dataset name.")
parser.add_argument("-l", type=str, dest="landmarks_dataset", default="CelebA",
                    help="Landmark localization dataset name.")
args = parser.parse_args()

landmarks_dataset = get_by_name(args.landmarks_dataset)
landmarks_meta = landmarks_dataset.get_train_meta()
landmarks_eval_meta = landmarks_dataset.get_val_meta()

print("Start generate landmarks training data for onet.")
gl.gen_landmark_data(landmarks_meta, 48, args.output_folder, argument=True, suffix='onet')
print("Done")
print("Start generate landmarks eval data for onet.")
gl.gen_landmark_data(landmarks_eval_meta, 48, args.output_folder, argument=True, suffix='onet_eval')
print("Done")

# load pre-trained pnet
print("Loading pre-trained pnet.")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pnet = PNet(device=device, is_train=False)
pnet.load(args.pnet_model_file)
rnet = RNet(device=device, is_train=False)
rnet.load(args.rnet_model_file)

detection_dataset = get_by_name(args.detection_dataset)
detection_meta = detection_dataset.get_train_meta()
detection_eval_meta = detection_dataset.get_val_meta()
print("Start generate classification and bounding box regression training data.")
gotd.generate_training_data_for_onet(pnet, rnet, detection_meta, args.output_folder, crop_size=48, suffix='onet')
print("Done")

print("Start generate classification and bounding box regression eval data.")
gotd.generate_training_data_for_onet(pnet, rnet, detection_eval_meta, args.output_folder, crop_size=48, suffix='onet_eval')
print("Done")
