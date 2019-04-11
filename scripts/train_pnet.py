import argparse
from mtcnn.train.train_net import Trainer

parser = argparse.ArgumentParser(
    description='Generate training data for pnet.')
parser.add_argument('-e', dest='epoch', type=int)
parser.add_argument('-b', dest='batch_size', type=int)
parser.add_argument('-d', dest="data_train", default="output/data_train", type=str, help="Folder that save training data for pnet.")
parser.add_argument('-dv', dest="device", default='cpu', type=str, help="'gpu', 'cuda:0' and so on.")

args = parser.parse_args()

trainer = Trainer('pnet', device='cuda:0', log_dir='./runs/pnet/')
trainer.train(args.epoch, args.batch_size, args.data_train)