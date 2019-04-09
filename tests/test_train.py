import unittest
from mtcnn.train.train_net import Trainer

class TestTrain(unittest.TestCase):

    def test_train_pnet(self):
        trainer = Trainer('pnet', device='cuda:0')
        trainer.train(10, 256, './output/data_train')
    

if __name__ == "__main__":
    unittest.main()

    