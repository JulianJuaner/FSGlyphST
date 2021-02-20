import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import SELayer, ConvModule

def build_head(head_cfg, model_cfg):
    if head_cfg.type == 'SimpleUpsample':
        return SimpleUpsample(head_cfg, model_cfg)
    else:
        raise NotImplementedError

BatchNorm = nn.BatchNorm2d

class SimpleUpsample(nn.Module):
    """
        A simple upsample module for decoding.
    """
    def __init__(self, head_cfg, model_cfg):
        super(SimpleUpsample, self).__init__()
        self.upsample = int(math.log(head_cfg.stride, 2))
        self.in_channel = model_cfg.backbone.out_dim + model.cfg.embedding.embedding_dim
        self.out_channel = model_cfg/out_channels

        self.conv_model = []
        self.conv_model.append(ConvModule(in_channel, in_channel//8, 3, 1, 1))
        self.conv_model.append(ConvModule(in_channel//4, in_channel//8, 3, 1, 1))
        for i in range(self.upsample-2):
            self.conv_model.append(ConvModule(in_channel//8, in_channel//8, 3, 1, 1))

        self.conv_model.append(nn.Conv2d(in_channel//4, self.out_channel, 1, 1, 0))
            

    def forward(self, embed):
        x = self.conv_model[0](embed)
        for i in range(1, self.upsample):
            x = F.interpolate(x, scale_factor=2)
            x = self.conv_model[i](x)

        x = F.interpolate(x, scale_factor=2)
        x = self.conv_model[-1](x)
        return F.sigmoid(x)


