import torch
import progressbar

from tensorboardX import SummaryWriter
from mtcnn.network.mtcnn_pytorch import PNet, RNet, ONet
from mtcnn.train.data import get_data_loader

class Trainer(object):

    def __init__(self, net_stage, optimizer="SGD", device='cpu'):
        
        self.net_stage = net_stage
        self.device = device
        
        if net_stage == 'pnet':
            self.net = PNet(is_train=True, device=self.device)
        
        if optimizer is "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        else:
            raise AttributeError("Don't support optimizer named %s." % optimizer)

        self.writer = SummaryWriter()
        self.globle_step = 0
        self.epoch_num = 0

        

    def train(self, num_epoch, batch_size, data_folder):
        for i in range(num_epoch):
            loader = get_data_loader(data_folder, self.net_stage, batch_size)
            self._train_epoch(loader)
            

    def _train_epoch(self, dataloader):

        for batch in dataloader:
            loss = self._train_batch(batch)
            self.globle_step += 1
            self.writer.add_scalar("Loss", loss, self.globle_step)

    def _train_batch(self, batch):
        images = batch[0].to(self.device)

        labels = batch[1].to(self.device)
        boxes = batch[2].to(self.device)
        landmarks = batch[3].to(self.device)

        self.optimizer.zero_grad()
        loss = self.net.get_loss(images, labels, boxes, landmarks)
        loss.backward()
        self.optimizer.step()
        print(loss)
        return loss

    
    def eval(self, dataloader):
        pass

    

