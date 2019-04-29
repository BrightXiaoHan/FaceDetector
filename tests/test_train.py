import unittest
import torch
from mtcnn.train.train_net import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestTrain(unittest.TestCase):

    def test_train_pnet(self):

        trainer = Trainer('pnet', device=device, log_dir='./runs/test/', output_folder='./runs/test/')
        trainer.train(20, 256, './output/test')

    def test_train_rnet(self):
        trainer = Trainer('rnet', device=device, log_dir='./runs/test/', output_folder='./runs/test/')
        trainer.train(3, 256, './output/test')
    
    def test_train_onet(self):
        trainer = Trainer('onet', device=device, log_dir='./runs/test/', output_folder='./runs/test')
        trainer.train(3, 256, './output/test')

if __name__ == "__main__":
    unittest.main()

    