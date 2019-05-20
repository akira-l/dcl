import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.resnet_features import resnet50_features
from models.resnet_utilities.layers import conv1x1, conv3x3


class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)

        # both based around resnet_feature_5
        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)




    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, x):

        # don't need resnet_feature_2 as it is too large
        # resnet feature shape:
        #     3 torch.Size([2, 512, 28, 28])
        #     4 torch.Size([2, 1024, 14, 14])
        #     5 torch.Size([2, 2048, 7, 7])

        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)

        pyramid_feature_4 = self.upsample_transform_1(
            torch.add(upsampled_feature_5, pyramid_feature_4)
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)

        pyramid_feature_3 = self.upsample_transform_2(
            torch.add(upsampled_feature_4, pyramid_feature_3)
        )
        '''
        (Pdb) fpn_feat[0].shape
        torch.Size([48, 256, 56, 56])
        (Pdb) fpn_feat[1].shape
        torch.Size([48, 256, 28, 28])
        (Pdb) fpn_feat[2].shape
        torch.Size([48, 256, 14, 14])
        (Pdb) fpn_feat[3].shape
        torch.Size([48, 256, 7, 7])
        (Pdb) fpn_feat[4].shape
        torch.Size([48, 256, 4, 4])

        '''

        return pyramid_feature_3, pyramid_feature_4, pyramid_feature_5, pyramid_feature_6, pyramid_feature_7



