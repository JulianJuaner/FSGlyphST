import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import SELayer, ConvModule

def build_head(head_cfg, model_cfg):
    if head_cfg.type == 'SimpleUpsample':
        return SimpleUpsample(head_cfg, model_cfg)
    if head_cfg.type == 'SkipConnect':
        return SkipConnect(head_cfg, model_cfg)
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
        self.in_channel = model_cfg.backbone.out_dim[0] + model_cfg.embedding.embedding_dim
        self.out_channel = model_cfg.out_channels
        self.dropout = head_cfg.dropout

        self.conv_model = nn.ModuleList()
        self.conv_model.append(ConvModule(self.in_channel, self.in_channel//4, 3, 1, 1))
        self.conv_model.append(ConvModule(self.in_channel//4, self.in_channel//4, 3, 1, 1))
        
        for i in range(self.upsample-1):
            self.conv_model.append(ConvModule(self.in_channel//4, self.in_channel//8, 3, 1, 1))
            self.conv_model.append(ConvModule(self.in_channel//8, self.in_channel//4, 3, 1, 1))

        self.conv_model.append(nn.Conv2d(self.in_channel//4, self.in_channel//4, 3, 1, 1))
        self.conv_model.append(nn.Conv2d(self.in_channel//4, self.out_channel, 1, 1, 0))
        self.dropout = nn.Dropout2d(p=self.dropout)

    def forward(self, embed):
        embed = embed[-1]
        x = self.conv_model[0](embed)
        x = self.conv_model[1](x)
        for i in range(1, self.upsample):
            x = F.interpolate(x, scale_factor=2)
            # residual.
            x = x + self.conv_model[2*i+1](self.conv_model[2*i](x))

        x = F.interpolate(x, scale_factor=2)
        x = self.conv_model[-2](x)
        x = self.dropout(x)
        x = self.conv_model[-1](x)
        return F.sigmoid(x)

class SkipConnect(nn.Module):
    """
        A simple upsample module for decoding.
    """
    def __init__(self, head_cfg, model_cfg):
        super(SkipConnect, self).__init__()
        self.upsample = int(math.log(head_cfg.stride, 2))
        self.in_channel = model_cfg.backbone.out_dim
        self.in_channel[0] +=  model_cfg.embedding.embedding_dim
        self.inter_channel = 128
        self.out_channel = model_cfg.out_channels
        self.conv_model = nn.ModuleList()
        self.dropout = head_cfg.dropout

        self.conv_model.append(ConvModule(self.in_channel[0], self.inter_channel, 3, 1, 1))
        self.conv_model.append(ConvModule(self.inter_channel, self.inter_channel, 3, 1, 1))

        for i in range(1, len(self.in_channel)):
            self.conv_model.append(ConvModule(self.inter_channel+self.in_channel[i], self.inter_channel, 3, 1, 1))
            self.conv_model.append(ConvModule(self.inter_channel, self.inter_channel, 3, 1, 1))

        for i in range(self.upsample-len(self.in_channel)+1):
            self.conv_model.append(ConvModule(self.inter_channel, self.inter_channel, 3, 1, 1))
            self.conv_model.append(ConvModule(self.inter_channel, self.inter_channel, 3, 1, 1))

        self.conv_model.append(nn.Conv2d(self.inter_channel, self.out_channel, 1, 1, 0))
        self.dropout = nn.Dropout2d(p=self.dropout)
    def forward(self, embed):
        
        x = self.conv_model[0](embed[-1])
        x = self.conv_model[1](x)
        
        for i in range(1, len(self.in_channel)):
            x = torch.cat((F.interpolate(x, scale_factor=2), embed[-i-1]), 1)
            # residual.
            x = self.conv_model[2*i+1](self.conv_model[2*i](x))
        # print(self.upsample-len(self.in_channel))
        for i in range(len(self.in_channel), self.upsample+1):
            
            x = F.interpolate(x, scale_factor=2)
            # residual.
            x = self.conv_model[2*i+1](self.conv_model[2*i](x))

        x = self.dropout(x)
        x = self.conv_model[-1](x)
        return F.sigmoid(x)