import torch.nn as nn
import torchvision
import os
import torch
import torch.nn.functional as F
from sourcecode.configs.make_cfg import merge_dict
from torch.nn import BatchNorm2d, SyncBatchNorm
from sourcecode.model.utils import build_resnet_block, conv3x3, ConvModule

def build_norm_layer(norm_cfg, **kwargs):
    from torch.nn import BatchNorm2d, SyncBatchNorm

    norm_layer_dict = {
        'bn_2d': BatchNorm2d,
        'sync_bn': SyncBatchNorm,
    }

    def norm(num_features):
        if norm_cfg is not None:
            _norm_cfg = norm_cfg.copy()
            norm_type = _norm_cfg.pop('type')
            return norm_layer_dict[norm_type](num_features, **_norm_cfg)
        else:
            return norm_layer_dict['bn_2d'](num_features)

    return norm

ResNet18_cfg = {
    'type': 'ResNet18',
    'layers': [2, 2, 2, 2],
    'strides': [1, 2, 2, 2],
    'dilations': [1, 1, 1, 1],
    # If wanna SeResNet, set block as 'SEBasicBlock'
    'block': 'BasicBlock'
}

ResNet34_cfg = {
    'type': 'ResNet34',
    'layers': [3, 4, 6, 3],
    'strides': [1, 2, 2, 2],
    'dilations': [1, 1, 1, 1],
    'block': 'BasicBlock'
}

ResNet50_cfg = {
    'type': 'ResNet50',
    'layers': [3, 4, 6, 3],
    'strides': [1, 2, 2, 2],
    'dilations': [1, 1, 1, 1],
    # If wanna SeResNet, set block as 'SEBottleNeck'
    'block': 'BottleNeck'
}

ResNet101_cfg = {
    'type': 'ResNet101',
    'layers': [3, 4, 23, 3],
    'strides': [1, 2, 2, 2],
    'dilations': [1, 1, 1, 1],
    'block': 'BottleNeck'
}

def build_backbone(backbone_cfg, full_cfg):
    if backbone_cfg.type == "ResNet18":
        net = ResNet18(in_channels=full_cfg.in_channels)
        print(backbone_cfg.weights)
        if os.path.exists(backbone_cfg.weights):
            print('loading pretrain backbone')
            net.load_state_dict(torch.load(backbone_cfg.weights), strict=False)
        return net 
    else:
        raise NotImplementedError

class ResNet(nn.Module):

    def __init__(self,
                 in_channels,
                 block,
                 layers,
                 strides,
                 dilations,
                 num_filter: int = 64,
                 norm_layer=None,
                 deep_stem=False,
                 relu_inplace=True,
                 **kwargs):
        super().__init__()
        self.deep_stem = deep_stem
        self.input_channel = in_channels
        block = build_resnet_block(block)
        self.out_channels = []

        if not self.deep_stem:
            self.inplanes = num_filter
            self.conv1 = nn.Conv2d(
                self.input_channel,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.bn1 = build_norm_layer(norm_layer)(self.inplanes)
        else:
            self.inplanes = num_filter * 2
            self.conv1 = conv3x3(self.input_channel, num_filter, stride=2)
            self.bn1 = build_norm_layer(norm_layer)(num_filter)
            self.conv2 = conv3x3(num_filter, num_filter)
            self.bn2 = build_norm_layer(norm_layer)(num_filter)
            self.conv3 = conv3x3(num_filter, self.inplanes)
            self.bn3 = build_norm_layer(norm_layer)(self.inplanes)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            num_filter,
            layers[0],
            stride=strides[0],
            dilation=dilations[0],
            norm_layer=norm_layer,
            relu_inplace=relu_inplace)
        self.out_channels.append(num_filter * block.expansion)
        self.layer2 = self._make_layer(
            block,
            num_filter * 2,
            layers[1],
            stride=strides[1],
            dilation=dilations[1],
            norm_layer=norm_layer,
            relu_inplace=relu_inplace)
        self.out_channels.append(num_filter * 2 * block.expansion)
        self.layer3 = self._make_layer(
            block,
            num_filter * 4,
            layers[2],
            stride=strides[2],
            dilation=dilations[2],
            norm_layer=norm_layer,
            relu_inplace=relu_inplace)
        self.out_channels.append(num_filter * 4 * block.expansion)
        self.layer4 = self._make_layer(
            block,
            num_filter * 8,
            layers[3],
            stride=strides[3],
            dilation=dilations[3],
            norm_layer=norm_layer,
            relu_inplace=relu_inplace)
        self.out_channels.append(num_filter * 8 * block.expansion)
        self.layers = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SyncBatchNorm) or isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.last_transform = nn.Conv2d(self.out_channels[-1], self.out_channels[-1], 1, 1, 0)

    def get_layers(self):
        return self.layers

    def get_out_channels(self):
        return self.out_channels

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_layer=None,
                    multi_grid=False,
                    multi_dilation=None,
                    relu_inplace=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvModule(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm_layer=norm_layer,
                activation=None)

        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        stride,
                        atrous=1,
                        downsample=downsample,
                        norm_layer=norm_layer,
                        relu_inplace=relu_inplace))
            elif dilation == 4:
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        stride,
                        atrous=2,
                        downsample=downsample,
                        norm_layer=norm_layer,
                        relu_inplace=relu_inplace))
            else:
                raise RuntimeError(
                    '=> unknown dilation size: {}'.format(dilation))
        else:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    atrous=multi_dilation[0],
                    downsample=downsample,
                    norm_layer=norm_layer,
                    relu_inplace=relu_inplace))
        self.inplanes = planes * block.expansion

        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        atrous=multi_dilation[i % div],
                        norm_layer=norm_layer,
                        relu_inplace=relu_inplace))
        else:
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        atrous=dilation,
                        norm_layer=norm_layer,
                        relu_inplace=relu_inplace))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        self.layers = []
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_stem:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)
        global_res = self.last_transform(F.adaptive_avg_pool2d(x, 1))
        return self.layers, global_res



class ResNet18(ResNet):
    """
    Default output stride is 32.
    If using dilated conv to set output stride as 16, then
        strides: [1, 2, 2, 1]
        dilations: [1, 1, 1, 2]
    if using dilated conv to set output stride as 8, then
        strides: [1, 2, 1, 1]
        dilations: [1, 1, 2, 4]
    """

    def __init__(self, **kwargs):
        kwargs = merge_dict(ResNet18_cfg, kwargs)
        super().__init__(**kwargs)


class ResNet34(ResNet):

    def __init__(self, **kwargs):
        kwargs = merge_dict(ResNet34_cfg, kwargs)
        super().__init__(**kwargs)


class ResNet50(ResNet):

    def __init__(self, **kwargs):
        kwargs = merge_dict(ResNet50_cfg, kwargs)
        super().__init__(**kwargs)


class ResNet101(ResNet):

    def __init__(self, **kwargs):
        kwargs = merge_dict(ResNet101_cfg, kwargs)
        super().__init__(**kwargs)
