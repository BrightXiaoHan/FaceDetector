import torch
import progressbar

from mtcnn.network.mtcnn_pytorch import PNet, RNet, ONet
from mtcnn.train.data import MtcnnDataset

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

        self.globle_step = 0
        self.epoch_num = 0

        
    def train(self, num_epoch, batch_size, data_folder):
        dataset = MtcnnDataset(data_folder, self.net_stage, batch_size)

        for i in range(num_epoch):
            data_iter = dataset.get_iter()
            self._train_epoch(data_iter)
            

    def _train_epoch(self, data_iter):

        for batch in data_iter:
            loss = self._train_batch(batch)
            self.globle_step += 1

    def _train_batch(self, batch):

        # assemble batch
        (pos_img, pos_reg), (part_img, part_reg), (neg_img, neg_reg), (landm_img, landm_reg) = batch

        # stack all images together
        images = torch.cat([pos_img, part_img, neg_img, landm_img]).to(self.device)

        # create labels for each image. 0 (neg), 1 (pos), 2 (part), 3(landmark)
        pos_label = torch.ones(pos_img.shape[0], dtype=torch.long)
        part_label = torch.ones(part_img.shape[0], dtype=torch.long) * 2
        neg_label = torch.zeros(neg_img.shape[0], dtype=torch.long)
        landm_label = torch.ones(landm_img.shape[0], dtype=torch.long) * 3

        labels = torch.cat([pos_label, part_label, neg_label, landm_label]).to(self.device)

        # stack boxes reg
        fake_landm_data_box_reg = torch.zeros((landm_img.shape[0], 4), dtype=torch.float)
        boxes_reg = torch.cat([pos_reg, part_reg, neg_reg, fake_landm_data_box_reg]).to(self.device)

        # stack landmarks reg
        fake_data_landm_reg = torch.zeros((pos_label.shape[0] + part_label.shape[0] + neg_label.shape[0], 10), dtype=torch.float)
        landmarks = torch.cat([fake_data_landm_reg, landm_reg]).to(self.device)

        # train step
        self.optimizer.zero_grad()
        loss = self.net.get_loss(images, labels, boxes_reg, landmarks)
        loss.backward()
        self.optimizer.step

        print(loss)
        return loss

    
    def eval(self, dataloader):
        pass

    

