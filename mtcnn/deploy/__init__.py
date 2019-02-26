import mtcnn.network.mtcnn_pytorch as mtcnn_pytorch


def get_net():
    """
    Create pnet, rnet, onet for detector.
    """

    pnet = mtcnn_pytorch.PNet()
    rnet = mtcnn_pytorch.RNet()
    onet = mtcnn_pytorch.ONet()

    return pnet, rnet, onet
