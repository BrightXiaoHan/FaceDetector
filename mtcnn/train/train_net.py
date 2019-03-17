import torch

from tensorboardX import SummaryWriter
from mtcnn.network.mtcnn_pytorch import PNet, RNet, ONet

class Trainer(object):

    def __init__(self, net_stage, optimizer="SGD", device='cpu'):
        
        self.net_stage = net_stage
        
        if net_stage == 'pnet':
            self.net = PNet()
        
        if optimizer is "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        else:
            raise AttributeError("Don't support optimizer named %s." % optimizer)

        self.writer = SummaryWriter()
        self.globle_step = 0
        self.epoch_num = 0

        

    def train(self, num_epoch, batch_size, ):
        pass

    def _train_epoch(self, dataloader):
        for batch in dataloader:
            loss = self._train_batch(batch)
            self.globle_step += 1
            self.writer.add_scalar("Loss", loss, self.globle_step)



    def _train_batch(self, batch):
        images = batch[0]
        labels = batch[1]
        boxes = batch[2]
        landmarks = batch[3]

        self.optimizer.zero_grad()
        loss = self.net.get_loss(images, labels, boxes, landmarks)
        loss.backward()
        self.optimizer.step()

        return loss

    
    def eval(self, dataloader):
        pass

    

