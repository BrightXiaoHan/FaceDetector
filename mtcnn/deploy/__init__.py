import os
import numpy as np
import mtcnn.network.mtcnn_pytorch as mtcnn_pytorch


def get_net():
    """
    Create pnet, rnet, onet for detector.
    """

    pnet = mtcnn_pytorch.PNet()
    rnet = mtcnn_pytorch.RNet()
    onet = mtcnn_pytorch.ONet()

    return pnet, rnet, onet

def get_net_caffe(weight_folder):
    """
    Create pnet, rnet, onet for detector. And init weights with caffe model from original mtcnn repo.
    """
    pnet, rnet, onet = get_net()
    pnet.load_caffe_model(
    np.load(os.path.join(weight_folder, 'pnet.npy'))[()])
    rnet.load_caffe_model(
        np.load(os.path.join(weight_folder, 'rnet.npy'))[()])
    onet.load_caffe_model(
        np.load(os.path.join(weight_folder, 'onet.npy'))[()])

    return pnet, rnet, onet