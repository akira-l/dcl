import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels


class resnet_swap_2loss_add(nn.Module):
    def __init__(self, num_classes, stage, resume):
        super(resnet_swap_2loss_add,self).__init__()
        arch = 'se_resnet50'
        ser50 = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)
        ser50.avg_pool = nn.AdaptiveAvgPool2d(1)
        ser50.last_linear = nn.Linear(ser50.last_linear.in_features, 2019)

        if resume is not None:
            state_dict = torch.load(resume)['state_dict']
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items() if not 'num_batches_tracked' in k}
            ser50.load_state_dict(state_dict)
            print("RESUME>>>>>>>>>>>>>>>>>>>>>")

        for i in range(5):
            setattr(self, 'layer{}'.format(i), getattr(ser50, 'layer{}'.format(i)))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, num_classes)
        self.classifier_swap = nn.Linear(2048, 2)

        self.dim = 1024
        self.relu = nn.ReLU(inplace=True)

        self.stage = stage
        if stage == 3:
            self.reduce_dim_conv = nn.Conv2d(2048, self.dim, 1, stride=1, padding=0, bias=False)
            self.size = 7 * 7
        elif stage == 2:
            self.reduce_dim_conv = nn.Conv2d(1024, self.dim, 1, stride=1, padding=0, bias=False)
            self.size = 14 * 14
        elif stage == 1:
            self.reduce_dim_conv = nn.Conv2d(512, self.dim, 1, stride=1, padding=0, bias=False)
            self.size = 28 * 28
        else:
            raise NotImplementedError("No such stage")
        self.align_conv = nn.Sequential(self.reduce_dim_conv, self.relu)

        self.Convmask = nn.Conv2d(2048, 2, 1, stride=1, padding=0, bias=False)
        self.avgpool2 = nn.AvgPool2d(2,stride=2)


        
    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = x5
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        cls_r = self.classifier(x)
        out.append(cls_r)

        if self.training:
            if self.stage == 1:
                F = x3
            elif self.stage == 2:
                F = x4
            elif self.stage == 3:
                F = x5
            else:
                raise RuntimeError("WHAT'S FUCK")
            features = self.align_conv(F).permute(0, 2, 3, 1).view(F.size(0), self.size, self.dim)
            features = torch.matmul(features, features.transpose(1,2))
            mask = torch.stack( [torch.eye(self.size) for _ in range(x.size(0))] ).byte()
            features[mask] = 0
            features = features / self.dim
            out.append(features)
            out.append(F)

        return out
