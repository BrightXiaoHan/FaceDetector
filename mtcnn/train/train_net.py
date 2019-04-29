import os
import torch
import glob
import progressbar

from mtcnn.network.mtcnn_pytorch import PNet, RNet, ONet
from mtcnn.train.data import MtcnnDataset
from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(self, net_stage, optimizer="SGD", device='cpu', log_dir='./runs', output_folder='./runs', resume=False):
        
        self.net_stage = net_stage
        self.device = device
        self.output_folder = output_folder
        
        if net_stage == 'pnet':
            self.net = PNet(is_train=True, device=self.device)
        
        elif net_stage == 'rnet':
            self.net = RNet(is_train=True, device=self.device)
        
        elif net_stage == 'onet':
            self.net = ONet(is_train=True, device=self.device)
        
        if optimizer is "SGD":
            # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
            self.optimizer = torch.optim.Adam(self.net.parameters())
        else:
            raise AttributeError("Don't support optimizer named %s." % optimizer)

        self.globle_step = 1
        self.epoch_num = 1

        if resume:
            self.load_state_dict()

        self.writer = SummaryWriter(log_dir=log_dir, purge_step=self.epoch_num)

        
    def train(self, num_epoch, batch_size, data_folder):
        dataset = MtcnnDataset(data_folder, self.net_stage, batch_size, suffix=self.net_stage)
        eval_dataset = MtcnnDataset(data_folder, self.net_stage, batch_size, suffix=self.net_stage+'_eval')

        for i in range(num_epoch - self.epoch_num + 1):
            print("Training epoch %d ......" % self.epoch_num)
            data_iter, total_batch = dataset.get_iter()
            self._train_epoch(data_iter, total_batch)
            print("Training epoch %d done." % self.epoch_num)

            print("Evaluate on training data...")
            data_iter, total_batch = dataset.get_iter()
            result = self.eval(data_iter, total_batch)
            print("Epoch %d, " % self.epoch_num, "result on training set: acc %f, precision %f, recall %f, f1 %f, avg_cls_loss %f, avg_box_loss %f, avg_landmark_loss %f" % result)

            self.writer.add_scalars("training_set", {
                i: j for i, j in 
                zip(["acc", "precision", "recall", "f1", "avg_cls_loss", "avg_box_loss", "avg_landmark_loss"], result)
            }, global_step=self.epoch_num)

            print("Evaluate on eval data...")
            data_iter, total_batch = eval_dataset.get_iter()
            result = self.eval(data_iter, total_batch)

            self.writer.add_scalars("eval_set", {
                i: j for i, j in 
                zip(["acc", "precision", "recall", "f1", "avg_cls_loss", "avg_box_loss", "avg_landmark_loss"], result)
            }, global_step=self.epoch_num)
            print("Epoch %d, " % self.epoch_num, "result on eval set: acc %f, precision %f, recall %f, f1 %f, avg_cls_loss %f, avg_box_loss %f, avg_landmark_loss %f" % result)

            self.save_state_dict()

            self.epoch_num += 1

    def _train_epoch(self, data_iter, total_batch):
        
        bar = progressbar.ProgressBar(max_value=total_batch)

        for i, batch in enumerate(data_iter):
            bar.update(i)

            loss = self._train_batch(batch)
            self.writer.add_scalar('train/batch_loss', loss, global_step=self.epoch_num)
            self.globle_step += 1

        bar.update(total_batch)    

    def _train_batch(self, batch):

        # assemble batch
        images, labels, boxes_reg, landmarks = self._assemble_batch(batch)

        # train step
        self.optimizer.zero_grad()
        loss = self.net.get_loss(images, labels, boxes_reg, landmarks)
        loss.backward()
        self.optimizer.step()

        return loss


    def _assemble_batch(self, batch):
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

        return images, labels, boxes_reg, landmarks
    
    def eval(self, data_iter, total_batch):
        total = 0
        right = 0
        tp = 0  # True positive
        fp = 0  # False positive
        fn = 0  # False negative
        tn = 0  # True negative

        total_cls_loss = 0
        total_box_loss = 0
        total_landmark_loss = 0

        bar = progressbar.ProgressBar(max_value=total_batch)

        for i, batch in enumerate(data_iter):
            bar.update(i)

            # assemble batch
            images, gt_label, gt_boxes, gt_landmarks = self._assemble_batch(batch)
            
            # Forward pass
            with torch.no_grad():
                pred_label, pred_offset, pred_landmarks = self.net.forward(images)

            # Reshape the tensor
            pred_label = pred_label.view(-1, 2)
            pred_offset = pred_offset.view(-1, 4)
            pred_landmarks = pred_landmarks.view(-1, 10)

            # Compute the loss
            total_cls_loss += self.net.cls_loss(gt_label, pred_label)
            total_box_loss += self.net.box_loss(gt_label, gt_boxes, pred_offset)
            total_landmark_loss += self.net.landmark_loss(
                gt_label, gt_landmarks, pred_landmarks)

            # compute the classification acc
            pred_label = torch.argmax(pred_label, dim=1)

            mask = gt_label <= 1
            right += torch.sum(gt_label[mask] == pred_label[mask])
            total += gt_label[mask].shape[0]
        
            p_mask = gt_label == 1
            tp += torch.sum(gt_label[p_mask] == pred_label[p_mask])
            fp += torch.sum(gt_label[p_mask] != pred_label[p_mask])

            n_mask = gt_label == 0
            tn += torch.sum(gt_label[n_mask] == pred_label[n_mask])
            fn += torch.sum(gt_label[n_mask] != pred_label[n_mask])

        bar.update(total_batch)

        acc = right.float() / total
        precision = tp.float() / (tp + fp)
        recall = tp.float() / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        avg_cls_loss = total_cls_loss / i
        avg_box_loss = total_box_loss / i
        avg_landmark_loss = total_landmark_loss / i

        return acc, precision, recall, f1, avg_cls_loss, avg_box_loss, avg_landmark_loss    


    def save_state_dict(self):
        checkpoint_name = "checkpoint_epoch_%d" % self.epoch_num
        file_path = os.path.join(self.output_folder, checkpoint_name)

        state = {
            'epoch_num': self.epoch_num,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file_path)

    def export_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_state_dict(self):

        # Get the latest checkpoint in output_folder
        all_checkpoints = glob.glob(os.path.join(self.output_folder, 'checkpoint_epoch_*'))

        if len(all_checkpoints) > 1:
            epoch_nums = [int(i.split('_')[-1]) for i in all_checkpoints]
            max_index = epoch_nums.index(max(epoch_nums))
            latest_checkpoint = all_checkpoints[max_index]

            state = torch.load(latest_checkpoint)
            self.epoch_num = state['epoch_num'] + 1
            self.net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer']) 
