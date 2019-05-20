import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model
#from models.FPN_model import FeaturePyramid
#from models.resnet_features import resnet50_features
#from utils.utils import convolution, fully_connected, residual

import pdb


class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.use_fpn = config.use_fpn
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone

        if self.backbone_arch in dir(models):
            pretrained_model = getattr(models, self.backbone_arch)()
            #self.model = nn.Sequential(*list(pretrained_model.children())[:-1])
            self.model = pretrained_model
        elif self.backbone_arch in pretrained_model:
            self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)

        if self.backbone_arch == 'resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_dcl:
            self.classifier_swap = nn.Linear(2048, 2*self.num_classes)
            self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=False)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.model(x)

        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        return out


