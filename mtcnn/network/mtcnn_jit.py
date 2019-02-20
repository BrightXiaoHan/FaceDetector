"""
Mtcnn net with the hybrid frontend introduce by pytorch documentation "https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html?highlight=jit".
"""

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


class _Net(torch.jit.ScriptModule):
    def __init__(self, device='cpu'):
        super(_Net, self).__init__()

        self.device = torch.device(device)

        self._init_net()

        # weight initiation with xavier
        self.apply(weights_init)

    
    def _init_net(self):
        raise NotImplementedError

    
    def load_caffe_model(self, weights):
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n], device=self.device)

    def to(self, device):
        for _, p in self.named_parameters():
            p.data = p.data.to(device)
        
        return self
    
    def eval(self):
        pass


class PNet(_Net):

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

    @torch.jit.script_method
    def forward(self, x):
        feature_map = self.body(x)
        label = self.cls(feature_map)
        offset = self.box_offset(feature_map)
        landmarks = None

        return label, offset, landmarks


class RNet(_Net):


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

    @torch.jit.script_method
    def forward(self, x):
        # backend
        x = self.body(x)

        # detection
        det = self.cls(x)
        box = self.box_offset(x)
        landmarks =  None

        return det, box, landmarks


class ONet(_Net):

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

    @torch.jit.script_method
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
