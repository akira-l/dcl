from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels


class seresnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet_swap_2loss_add,self).__init__()
        arch = 'se_resnet50'
        ser50 = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)
        ser50.avg_pool = nn.AdaptiveAvgPool2d(1)
        ser50.last_linear = nn.Linear(ser50.last_linear.in_features, 2019)
        #resume = './net_model/pretrained/checkpoint_epoch_017_prec1_65.885_pth.tar'
        #state_dict = torch.load(resume)['state_dict']
        #state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items() if not 'num_batches_tracked' in k}
        #ser50.load_state_dict(state_dict)

        for i in range(5):
            setattr(self, 'layer{}'.format(i), getattr(ser50, 'layer{}'.format(i)))

        self.dim = 1024

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = ser50.last_linear#nn.Linear(2048, num_classes)
        self.classifier_swap = nn.Linear(2048, 2*num_classes)
        #self.classifier_swap = nn.Linear(2048, 2)
        self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=False)
        self.avgpool2 = nn.AvgPool2d(2,stride=2)

    def forward(self, x):

        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = x5
        mask = self.Convmask(x)
        mask = self.avgpool2(mask)
        mask = F.tanh(mask)
        mask = mask.view(mask.size(0),-1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))
        out.append(self.classifier_swap(x))
        out.append(mask)

        return out
