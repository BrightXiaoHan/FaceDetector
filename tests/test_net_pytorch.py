import unittest
import torch

import mtcnn.network.matcnn_pytorch as mtcnn

class TestMtcnnPytorch(unittest.TestCase):

    def test_pnet(self):
        pnet = mtcnn.PNet(is_train=True)
        data = torch.randn(100, 3, 12, 12)

        det, box, landmarks = pnet(data)
        self.assertEqual(list(det.shape), [100, 1, 1, 1])
        self.assertEqual(list(box.shape), [100, 4, 1, 1])
        self.assertEqual(list(landmarks.shape), [100, 10, 1, 1])

        pnet.get_loss(data, torch.ones(100), torch.randn(100, 4), torch.randn(100, 10))

    def test_rnet(self):
        rnet = mtcnn.RNet(is_train=True)
        data = torch.randn(100, 3, 24, 24)

        det, box, landmarks = rnet(data)
        self.assertEqual(list(det.shape), [100, 1])
        self.assertEqual(list(box.shape), [100, 4])
        self.assertEqual(list(landmarks.shape), [100, 10])

        rnet.get_loss(data, torch.ones(100), torch.randn(100, 4), torch.randn(100, 10))

    def test_onet(self):
        onet = mtcnn.ONet(is_train=True)
        data = torch.randn(100, 3, 48, 48)

        det, box, landmarks = onet(data)
        self.assertEqual(list(det.shape), [100, 1])
        self.assertEqual(list(box.shape), [100, 4])
        self.assertEqual(list(landmarks.shape), [100, 10])

        onet.get_loss(data, torch.ones(100), torch.randn(100, 4), torch.randn(100, 10))
        
if __name__ == "__main__":
    unittest.main()