import cv2
import os
import random
import pandas as pd
import torch
import numpy as np
import mtcnn.train.gen_landmark as landm
import mtcnn.train.gen_pnet_train as pnet
import mtcnn.train.gen_rnet_train as rnet
import mtcnn.utils.functional as func

from torch.utils.data import Dataset, DataLoader

class ClsBoxData(object):
    
    """
    Define a custom data structure training classification task and bounding box regression task.
    """

    def __init__(self, pos, part, neg, pos_reg, part_reg):

        self.pos = pos
        self.part = part
        self.neg = neg
        self.pos_reg = pos_reg
        self.part_reg = part_reg

def get_training_data(output_folder, suffix):
    """Get training data for classification and bounding box regression tasks

    Arguments:
        output_folder {str} -- Consistent with parameter 'output_folder' passed in method "generate_training_data_for...".
        suffix {str} -- Create a folder called $suffix in $output_folder.
    Returns:
        {PnetData} -- 'PnetData' object.
    """

    positive_dest = os.path.join(output_folder, suffix, 'positive')
    negative_dest = os.path.join(output_folder, suffix, 'negative')
    part_dest = os.path.join(output_folder, suffix, 'part')

    positive_meta_file = os.path.join(output_folder, suffix, 'positive_meta.csv')
    part_meta_file = os.path.join(output_folder, suffix, 'part_meta.csv')
    negative_meta_file = os.path.join(output_folder, suffix, 'negative_meta.csv')

    # load from disk to menmory
    positive_meta = pd.read_csv(positive_meta_file)
    pos = [os.path.join(part_dest, i) for i in positive_meta.iloc[:, 0]]
    pos_reg = np.array(positive_meta.iloc[:, 1:])

    part_meta = pd.read_csv(part_meta_file)
    part = [os.path.join(part_dest, i) for i in part_meta.iloc[:, 0]]
    part_reg = np.array(part_meta.iloc[:, 1:])

    negative_meta = pd.read_csv(negative_meta_file)
    neg = [os.path.join(negative_dest, i) for i in negative_meta.iloc[:, 0]]

    return ClsBoxData(pos, part, neg, pos_reg, part_reg)


class LandmarkData(object):
    """
    Custom data structure for storing facial landmark points training data.
    """
    def __init__(self, images, landmarks):
        self.images = images
        self.landmarks = landmarks
    
    def __len__(self):
        return len(self.images)


def get_landmark_data(output_folder, suffix=''):
    
    image_file_folder = os.path.join(output_folder, suffix, 'landmarks')
    meta_file = os.path.join(output_folder, suffix, 'landmarks_meta.csv')

    meta = pd.read_csv(meta_file)
    images = [os.path.join(image_file_folder, i) for i in meta.iloc[:, 0]]
    landmarks = np.array(meta.iloc[:, 1:]).astype(float)
    
    return LandmarkData(images, landmarks)

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

    def __init__(self, image, meta=None, max_len=-1):
        if max_len >0 and max_len < len(image):
            if meta is None:
                image = random.sample(image, max_len)
            else:
                image, meta = zip(*random.sample(list(zip(image, meta)), max_len))

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

    def __init__(self, output_folder, net_stage, batch_size, suffix):
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
            self.landmark_data = get_landmark_data(
                output_folder, suffix=suffix)
            self.data = get_training_data(output_folder, suffix=suffix)
        elif net_stage == 'rnet':
            self.landmark_data = get_landmark_data(output_folder, suffix=suffix)
            self.data = get_training_data(output_folder, suffix=suffix)
        elif net_stage == 'onet':
            self.landmark_data = get_landmark_data(output_folder, suffix=suffix)
            self.data = get_training_data(output_folder, suffix=suffix)
        else:
            raise AttributeError(
                "Parameter 'net_stage' must be one of 'pnet', 'rnet' and 'onet' instead of %s." % net_stage)

        # Ensure the ratio of four kinds of data (pos, part, landmark, neg) is 1:1:1:3. (Follow the original paper)
        min_len = int(min([len(self.data.pos), len(self.data.part), len(self.landmark_data.landmarks), len(self.data.neg) / 3]))

        self.pos = ImageMetaDataset(self.data.pos, self.data.pos_reg, max_len=min_len)
        self.part = ImageMetaDataset(self.data.part, self.data.part_reg, max_len=min_len)
        self.neg = ImageMetaDataset(self.data.neg, max_len=min_len * 3)
        self.landm = ImageMetaDataset(self.landmark_data.images, self.landmark_data.landmarks, max_len=min_len)

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

        total_batch = min([len(pos_loader), len(part_loader), len(neg_loader), len(landm_loader)])

        return generator(), total_batch
