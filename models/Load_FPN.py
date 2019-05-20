import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model
from models.FPN_model import FeaturePyramid
from models.Asoftmax_linear import AngleLinear
from models.resnet_features import resnet50_features
from utils.utils import convolution, fully_connected, residual

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        if config.use_fpn:
            _resnet = resnet50_features(pretrained=True)
            self.fpn = FeaturePyramid(_resnet)#FPN50()

        self.use_dcl = config.use_dcl
        self.use_fpn = config.use_fpn
        self.use_Asoftmax = config.use_Asoftmax
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)

        if self.backbone_arch == 'resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        self.conv_28_14 = convolution(3, 256, 512, stride=2)
        self.conv_56_14 = nn.Sequential(
            convolution(3, 256, 512, stride=2),
            convolution(3, 512, 1024, stride=2)
            )
        self.conv_14_14 = convolution(3, 256, 512, stride=1)

        self.emb = nn.Sequential(
                convolution(3, 2048, 2048, stride=1),
                convolution(3, 2048, 2048, stride=1),
                convolution(3, 2048, 2048, stride=1)
                )

        if self.use_dcl:
            self.classifier_swap = nn.Linear(2048, 2)
            self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(2048, self.num_classes)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, last_cont=None):
        fpn_feat = self.fpn(x)

        feat_56x56 = self.conv_56_14(fpn_feat[0])
        feat_28x28 = self.conv_28_14(fpn_feat[1])
        feat_14x14 = self.conv_14_14(fpn_feat[2])
        feat = torch.cat([feat_56x56, feat_28x28, feat_14x14], 1)
        emb_feat = self.emb(feat)


        if self.use_dcl:
            mask = self.Convmask(emb_feat)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(emb_feat)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out

