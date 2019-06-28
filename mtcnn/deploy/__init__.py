import os
import torch
import numpy as np
import mtcnn.network.mtcnn_pytorch as mtcnn_pytorch

from .detect import FaceDetector

here = os.path.dirname(os.path.abspath(__file__))

def get_net(weight_folder=None):
    """
    Create pnet, rnet, onet for detector.
    """

    pnet = mtcnn_pytorch.PNet()
    rnet = mtcnn_pytorch.RNet()
    onet = mtcnn_pytorch.ONet()

    if weight_folder is not None:
        pnet.load(os.path.join(weight_folder, 'pnet'))
        rnet.load(os.path.join(weight_folder, 'rnet'))
        onet.load(os.path.join(weight_folder, 'onet'))

    return pnet, rnet, onet


def get_net_caffe(weight_folder):
    """
    Create pnet, rnet, onet for detector. And init weights with caffe model from original mtcnn repo.
    """
    weight_folder = os.path.join(here, "models")
    pnet, rnet, onet = get_net()
    pnet.load_caffe_model(
        np.load(os.path.join(weight_folder, 'pnet.npy'), allow_pickle=True)[()])
    rnet.load_caffe_model(
        np.load(os.path.join(weight_folder, 'rnet.npy'), allow_pickle=True)[()])
    onet.load_caffe_model(
        np.load(os.path.join(weight_folder, 'onet.npy'), allow_pickle=True)[()])

    return pnet, rnet, onet

def get_default_detector(device=None):
    """
    Get the default face detector with pnet, rnet, onet trained by original mtcnn author. 
    """
    pnet, rnet, onet = get_net_caffe(os.path.join(here, "models"))
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector = FaceDetector(pnet, rnet, onet, device)
    return detector
