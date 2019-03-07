import os
import unittest
import torch

import numpy as np
import mtcnn.network.mtcnn_pytorch as mtcnn

here = os.path.dirname(__file__)

class TestMtcnnPytorch(unittest.TestCase):

    def test_pnet(self):
        pnet = mtcnn.PNet(is_train=True)
        data = torch.randn(100, 3, 12, 12)

        det, box, landmarks = pnet(data)
        self.assertEqual(list(det.shape), [100, 2, 1, 1])
        self.assertEqual(list(box.shape), [100, 4, 1, 1])
        self.assertEqual(list(landmarks.shape), [100, 10, 1, 1])

        pnet.get_loss(data, torch.randint(-1, 3, (100, ), dtype=torch.int64), torch.randn(100, 4), torch.randn(100, 10))

    def test_rnet(self):
        rnet = mtcnn.RNet(is_train=True)
        data = torch.randn(100, 3, 24, 24)

        det, box, landmarks = rnet(data)
        self.assertEqual(list(det.shape), [100, 2])
        self.assertEqual(list(box.shape), [100, 4])
        self.assertEqual(list(landmarks.shape), [100, 10])

        rnet.get_loss(data, torch.ones(100, dtype=torch.int64), torch.randn(100, 4), torch.randn(100, 10))

    def test_onet(self):
        onet = mtcnn.ONet(is_train=True)
        data = torch.randn(100, 3, 48, 48)

        det, box, landmarks = onet(data)
        self.assertEqual(list(det.shape), [100, 2])
        self.assertEqual(list(box.shape), [100, 4])
        self.assertEqual(list(landmarks.shape), [100, 10])

        onet.get_loss(data, torch.ones(100, dtype=torch.int64), torch.randn(100, 4), torch.randn(100, 10))

    def test_load_caffe_model(self):
        pnet = mtcnn.PNet()
        rnet = mtcnn.RNet()
        onet = mtcnn.ONet()
        weight_folder = os.path.join(here, '../output/converted')
        pnet.load_caffe_model(np.load(os.path.join(weight_folder, 'pnet.npy'))[()])
        rnet.load_caffe_model(np.load(os.path.join(weight_folder, 'rnet.npy'))[()])
        onet.load_caffe_model(np.load(os.path.join(weight_folder, 'onet.npy'))[()])
        
if __name__ == "__main__":
    unittest.main()