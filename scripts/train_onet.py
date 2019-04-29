import argparse
from mtcnn.train.train_net import Trainer

parser = argparse.ArgumentParser(
    description='Generate training data for onet.')
parser.add_argument('-e', dest='epoch', type=int)
parser.add_argument('-b', dest='batch_size', type=int)
parser.add_argument('-o', dest="output_filename", help="Path to save the model.")
parser.add_argument('-d', dest="data_train", default="output/data_train", type=str, help="Folder that save training data for onet.")
parser.add_argument('-dv', dest="device", default='cpu', type=str, help="'gpu', 'cuda:0' and so on.")
parser.add_argument('-r', dest="resume", default=False, type=bool, help="If resume from latest checkpoint.")


args = parser.parse_args()      

trainer = Trainer('onet', device=args.device, log_dir='./runs/onet/', resume=args.resume, output_folder='./runs/onet')
trainer.train(args.epoch, args.batch_size, args.data_train)
trainer.export_model(args.output_filename)