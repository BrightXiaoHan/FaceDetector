import unittest
import torch

import mtcnn.network.matcnn_pytorch as mtcnn

class TestMtcnnPytorch(unittest.TestCase):

    def test_pnet(self):
        pnet = mtcnn.PNet()
        data = torch.randn(100, 3, 24, 24)

        det, box = pnet(data)
        self.assertEqual(list(det.shape), [100, 1, 7, 7])
        self.assertEqual(list(box.shape), [100, 4, 7, 7])

    def test_rnet(self):
        rnet = mtcnn.RNet()
        data = torch.randn(100, 3, 24, 24)

        det, box = rnet(data)
        self.assertEqual(list(det.shape), [100, 3])
        self.assertEqual(list(box.shape), [100, 4])

    def test_onet(self):
        onet = mtcnn.ONet()
        data = torch.randn(100, 3, 48, 48)

        det, box, landmarks = onet(data)
        self.assertEqual(list(det.shape), [100, 3])
        self.assertEqual(list(box.shape), [100, 4])
        self.assertEqual(list(landmarks.shape), [100, 10])
        
if __name__ == "__main__":
    unittest.main()