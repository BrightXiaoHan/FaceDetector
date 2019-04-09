import cv2
import torch
import numpy as np
import mtcnn.train.gen_landmark as landm
import mtcnn.train.gen_pnet_train as pnet
import mtcnn.utils.functional as func

from torch.utils.data import Dataset, DataLoader

class ToTensor(object):

    def __call__(self, sample):
        sample[0] = cv2.imread(sample[0])
        # Convert image from rgb to bgr for Compatible with original caffe model.
        sample[0] = cv2.cvtColor(sample[0], cv2.COLOR_RGB2BGR)
        sample[0] = sample[0].transpose((2, 0, 1))
        sample[0] = func.imnormalize(sample[0])
        sample[0] = torch.tensor(sample[0], dtype=torch.float)

        sample[1] = torch.tensor(sample[1], dtype=torch.float)

        return sample

class ImageMetaDataset(Dataset):

    def __init__(self, image, meta=None):
        self.image = image
        self.meta = meta
        self.transform = ToTensor()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        if self.meta is None:
            # negative data has no bounding box regression labels. 
            sample = [self.image[index], np.zeros((4,))]

        else:
            sample = [self.image[index], self.meta[index]]
        
        return self.transform(sample)

class MtcnnDataset(object):
    """
    Dataset for training MTCNN.
    """

    def __init__(self, output_folder, net_stage, batch_size):
        """
        Put things together. The structure of 'output_folder' looks like this:

        output_folder/
        ├── landmarks (generate by 'gen_landmark_data' method.)
        │   ├── 1.jpg
        │   └── 2.jpg
        ├── negative  (neg, part, pos generate by 'generate_training_data_for_pnet' method)
        │   ├── 1.jpg
        │   └── 2.jpg
        ├── part
        │   ├── 1.jpg
        │   └── 2.jpg
        ├── positive
        ├──   ├── 1.jpg
        ├──   └── 2.jpg
        ├── pnet_negative_meta.csv
        ├── pnet_part_meta.csv
        └── pnet_positive_meta.csv

        net_stage is one of 'pnet', 'rnet' and 'onet'
        """

        self.batch_size = batch_size
        # get classification and regression tasks data
        if net_stage == 'pnet':
            # get landmarks data
            self.landmark_data = landm.get_landmark_data(
                output_folder, suffix='pnet')
            self.data = pnet.get_training_data_for_pnet(output_folder)
        elif net_stage == 'rnet':
            pass  # TODO
        elif net_stage == 'onet':
            pass  # TODO
        else:
            raise AttributeError(
                "Parameter 'net_stage' must be one of 'pnet', 'rnet' and 'onet' instead of %s." % net_stage)

        self.pos = ImageMetaDataset(self.data.pos, self.data.pos_reg)
        self.part = ImageMetaDataset(self.data.part, self.data.part_reg)
        self.neg = ImageMetaDataset(self.data.neg)
        self.landm = ImageMetaDataset(self.landmark_data.images, self.landmark_data.landmarks)

        pos_len = len(self.pos)
        part_len = len(self.part)
        neg_len = len(self.neg)
        landm_len = len(self.landm)

        total_len = pos_len + part_len + neg_len + landm_len

        self.pos_batch = int(batch_size * (pos_len / total_len))
        self.part_batch = int(batch_size * (part_len / total_len))
        self.neg_batch = int(batch_size * (neg_len / total_len))
        self.landm_batch = int(batch_size * (landm_len / total_len))

    def get_iter(self):
        pos_loader = DataLoader(self.pos, self.pos_batch, shuffle=True)
        part_loader = DataLoader(self.part, self.part_batch, shuffle=True)
        neg_loader = DataLoader(self.neg, self.neg_batch, shuffle=True)
        landm_loader = DataLoader(self.landm, self.landm_batch, shuffle=True)

        transform = ToTensor()

        def generator():

            for i in zip(pos_loader, part_loader, neg_loader, landm_loader):
                yield i

        return generator()
