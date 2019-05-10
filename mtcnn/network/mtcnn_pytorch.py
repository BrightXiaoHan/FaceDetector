import torch
import torch.nn as nn

from collections import OrderedDict


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class _Net(nn.Module):
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1, is_train=False, device='cpu'):
        super(_Net, self).__init__()

        self.is_train = is_train
        self.device = torch.device(device)

        self._init_net()

        if is_train:
            # loss function
            self.cls_factor = cls_factor
            self.box_factor = box_factor
            self.land_factor = landmark_factor
            self.loss_cls = nn.NLLLoss(reduction='none')
            self.loss_box = nn.MSELoss()
            self.loss_landmark = nn.MSELoss()

        # weight initiation with xavier
        self.apply(weights_init)

        # Move tensor to target device
        self.to(self.device)

        if not self.is_train:
            self.eval()

    def get_loss(self, x, gt_label, gt_boxes, gt_landmarks):
        """
        Get total loss.
        Arguments:
            x {Tensor} -- Input normalized images. (Note here: rnet, onet only support fix size images.)
            gt_label {Tensor} -- Ground truth label.
            gt_boxes {Tensor} -- Ground truth boxes coordinate.

        Returns:
            Tensor -- classification loss + box regression loss + landmark loss
        """
        if not self.is_train:
            raise AssertionError(
                "Method 'get_loss' is avaliable only when 'is_train' is True.")

        # Forward pass
        pred_label, pred_offset, pred_landmarks = self.forward(x)

        # Reshape the tensor
        pred_label = pred_label.view(-1, 2)
        pred_offset = pred_offset.view(-1, 4)
        pred_landmarks = pred_landmarks.view(-1, 10)

        # Compute the loss
        cls_loss = self.cls_loss(gt_label, pred_label)
        box_loss = self.box_loss(gt_label, gt_boxes, pred_offset)
        landmark_loss = self.landmark_loss(
            gt_label, gt_landmarks, pred_landmarks)

        return cls_loss + box_loss + landmark_loss

    def _init_net(self):
        raise NotImplementedError

    def cls_loss(self, gt_label, pred_label):
        """Classification loss 

        Args:
            gt_label (Tensor): Pobability distribution with shape (batch_size, 2)
            pred_label (Tensor): Ground truth lables with shape (batch_size)

        Returns:
            Tensor: Cross-Entropy loss multiply by cls_factor
        """

        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)

        # Online hard sample mining

        mask = torch.eq(gt_label, 0) | torch.eq(gt_label, 1)
        valid_gt_label = torch.masked_select(gt_label, mask)
        mask = torch.stack([mask] * 2, dim=1)
        valid_pred_label = torch.masked_select(pred_label, mask).reshape(-1, 2)

        # compute log-softmax
        valid_pred_label = torch.log(valid_pred_label)

        loss = self.loss_cls(valid_pred_label, valid_gt_label)

        pos_mask = torch.eq(valid_gt_label, 1)
        neg_mask = torch.eq(valid_gt_label, 0)

        neg_loss = loss.masked_select(neg_mask)
        pos_loss = loss.masked_select(pos_mask)

        if neg_loss.shape[0] > pos_loss.shape[0]:
            neg_loss, _ = neg_loss.topk(pos_loss.shape[0])
        loss = torch.cat([pos_loss, neg_loss])
        loss = torch.mean(loss)

        return loss * self.cls_factor

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        mask = torch.eq(gt_label, 1) | torch.eq(gt_label, 2)
        # broadcast mask
        mask = torch.stack([mask] * 4, dim=1)

        # only valid element can effect the loss
        valid_gt_offset = torch.masked_select(gt_offset, mask).reshape(-1, 4)
        valid_pred_offset = torch.masked_select(
            pred_offset, mask).reshape(-1, 4)
        return self.loss_box(valid_pred_offset, valid_gt_offset)*self.box_factor

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, 3)
        # broadcast mask
        mask = torch.stack([mask] * 10, dim=1)

        valid_gt_landmark = torch.masked_select(
            gt_landmark, mask).reshape(-1, 10)
        valid_pred_landmark = torch.masked_select(
            pred_landmark, mask).reshape(-1, 10)
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)*self.land_factor
    
    def load_caffe_model(self, weights):
        if self.is_train:
            raise AssertionError("This method is avaliable only when 'is_train' is false.")
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n], device="cpu")

    def load(self, model_file):
        state_dict = torch.load(model_file, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

    def to_script(self):
        raise NotImplementedError

class PNet(_Net):

    def __init__(self, **kwargs):
        # Hyper-parameter from original papaer
        param = [1, 0.5, 0.5]
        super(PNet, self).__init__(*param, **kwargs)

    def _init_net(self):

        # backend
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),
            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2d(16, 32, kernel_size=3, stride=1)),
            ('prelu3', nn.PReLU(32))
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv4-1', nn.Conv2d(32, 2, kernel_size=1, stride=1)),
            ('softmax', nn.Softmax(1))
        ]))
        # bounding box regresion
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv4-2', nn.Conv2d(32, 4, kernel_size=1, stride=1)),
        ]))

        if self.is_train:
            # landmark regression
            self.landmarks = nn.Sequential(OrderedDict([
                ('conv4-2', nn.Conv2d(32, 10, kernel_size=1, stride=1))
            ]))

    def forward(self, x):
        feature_map = self.body(x)
        label = self.cls(feature_map)
        offset = self.box_offset(feature_map)
        landmarks = self.landmarks(feature_map) if self.is_train else torch.empty(0, device=self.device)

        return label, offset, landmarks

    def to_script(self):
        data = torch.randn((100, 3, 12, 12), device=self.device)
        script_module = torch.jit.trace(self, data)
        return script_module


class RNet(_Net):

    def __init__(self, **kwargs):
        # Hyper-parameter from original papaer
        param = [1, 0.5, 0.5]
        super(RNet, self).__init__(*param, **kwargs)

    def _init_net(self):

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, kernel_size=3, stride=1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, kernel_size=2, stride=1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv5-1', nn.Linear(128, 2)),
            ('softmax', nn.Softmax(1))
        ]))
        # bounding box regression
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv5-2', nn.Linear(128, 4))
        ]))

        if self.is_train:
            # lanbmark localization
            self.landmarks = nn.Sequential(OrderedDict([
                ('conv5-3', nn.Linear(128, 10))
            ]))

    def forward(self, x):
        # backend
        x = self.body(x)

        # detection
        det = self.cls(x)
        box = self.box_offset(x)
        landmarks = self.landmarks(x) if self.is_train else torch.empty(0, device=self.device)

        return det, box, landmarks

    def to_script(self):
        data = torch.randn((100, 3, 24, 24), device=self.device)
        script_module = torch.jit.trace(self, data)
        return script_module


class ONet(_Net):

    def __init__(self, **kwargs):
        # Hyper-parameter from original papaer
        param = [1, 5, 50]
        super(ONet, self).__init__(*param, **kwargs)

    def _init_net(self):
        # backend
        
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, kernel_size=2, stride=1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv6-1', nn.Linear(256, 2)),
            ('softmax', nn.Softmax(1))
        ]))
        # bounding box regression
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv6-2', nn.Linear(256, 4))
        ])) 
        # lanbmark localization
        self.landmarks = nn.Sequential(OrderedDict([
            ('conv6-3', nn.Linear(256, 10))
        ])) 

    def forward(self, x):
        # backend
        x = self.body(x)

        # detection
        det = self.cls(x)

        # box regression
        box = self.box_offset(x)

        # landmarks regresion
        landmarks = self.landmarks(x)

        return det, box, landmarks

    def to_script(self):
        data = torch.randn((100, 3, 48, 48), device=self.device)
        script_module = torch.jit.trace(self, data)
        return script_module
