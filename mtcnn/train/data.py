import torch
import numpy as np
import mtcnn.train.gen_landmark as landm
import mtcnn.train.gen_pnet_train as pnet
import mtcnn.utils.functional as func

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_data_loader(output_folder, net_stage, framework, batch_size):
    transform = transforms.Compose([ToTensor(framework)])
    dataset = MtcnnDataset(output_folder, net_stage, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return dataloader

def collate_fn(sameples):
    images = torch.stack([i[0] for i in sameples])
    labels = torch.stack([i[1] for i in sameples])
    boxes = torch.stack([i[2] for i in sameples])
    landmarks = torch.stack([i[3] for i in sameples])    
    return images, labels, boxes, landmarks


class ToTensor(object):

    def __init__(self, framework):
        """Convert ndarrays in sample to Tensors.
        # TODO Temporary only supports pytorch

        Args:
            framework (str): tensorflow, pytorch, mtcnn is avaliable
        """
        if framework not in ['tensorflow', 'pytorch', 'mtcnn']:
            raise AttributeError(
                "Parameter 'framework must be one of 'tensorflow', 'pytorch' and 'mtcnn'.")
        self.framework = framework

        if self.framework == 'pytorch':
            import torch
            self.torch = torch

    def __call__(self, sample):
        sample[0] = sample[0].transpose((2, 0, 1))
        sample[0] = func.imnormalize(sample[0])

        if self.framework == 'pytorch':
            return list(map(self.torch.from_numpy, sample))


class MtcnnDataset(Dataset):
    """
    Dataset for training MTCNN.
    """

    def __init__(self, output_folder, net_stage, transform=None):
        """
        Put things together. The structure of 'output_folder' looks like this:

        output_folder/
        ├── landmarks (generate by 'gen_landmark_data' method.)
        │   ├── image_matrix.npy
        │   └── landmarks.npy
        ├── negative  (neg, part, pos generate by 'generate_training_data_for_pnet' method)
        │   ├── pnet_image_matrix.npy
        │   └── pnet_meta.csv
        ├── part
        │   ├── pnet_image_matrix.npy
        │   └── pnet_meta.csv
        └── positive
            ├── pnet_image_matrix.npy
            └── pnet_meta.csv
        net_stage is one of 'pnet', 'rnet' and 'onet'
        """
        self.transform = transform

        # get landmarks data
        landmark_image, landmark_meta = landm.get_landmark_data(
            output_folder, suffix='pnet/landmarks')
        landmark_meta = np.reshape(landmark_meta, (-1, 10))

        # get classification and box regression tasks data
        if net_stage == 'pnet':
            (pos_img, neg_img, part_img), (pos_meta, neg_meta,
                                           part_meta) = pnet.get_training_data_for_pnet(output_folder)
        elif net_stage == 'rnet':
            pass
        elif net_stage == 'onet':
            pass
        else:
            raise AttributeError(
                "Parameter 'net_stage' must be one of 'pnet', 'rnet' and 'onet' instead of %s." % net_stage)

        # assemble training data together
        self.images = np.concatenate(
            (pos_img, neg_img, part_img, landmark_image))

        # Give label 0 to neg, 1 to pos, 2 to part, -1 to landmark
        self.labels = np.concatenate((np.ones(pos_img.shape[0]), np.zeros(
            neg_img.shape[0]), np.ones(part_img.shape[0])*2, np.ones(landmark_image.shape[0]) * -2)).astype(int)

        # Ground thruth boxes coordinate
        self.gt_boxes = np.concatenate(
            (pos_meta, neg_meta, part_meta, np.zeros((landmark_meta.shape[0], 4))))

        # Ground thruth landmark
        self.gt_landm = np.concatenate([np.zeros((pos_img.shape[0], 10)), np.zeros(
            (neg_img.shape[0], 10)), np.zeros((part_img.shape[0], 10)), landmark_meta])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        sameple = [self.images[idx], np.array(self.labels[idx]),  # Prevent label degenerate into numbers
                   self.gt_boxes[idx], self.gt_landm[idx]]
        if self.transform:
            sameple = self.transform(sameple)
        return sameple
