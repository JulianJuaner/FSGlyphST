import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.utils import build_norm, ConvModule

def build_discrim(model_cfg, full_cfg):
    if model_cfg.type == "VGG":
        return myVGG(model_cfg, full_cfg)
    if model_cfg.type == "VGG_deep_shallow":
        return deepShallowVGG(model_cfg, full_cfg)
    if model_cfg.type == "VGG_deep_shallow_patch":
        return deepShallowVGGPatch(model_cfg, full_cfg)
    else:
        return discrim_zi2zi(model_cfg, full_cfg)

class myVGG(nn.Module):
    '''
        VGG Discriminator.
    '''
    def __init__(self, opts, cfg):
        super(myVGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # print(vgg16)
        self.vgg16_enc = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_enc.parameters():
            param.requires_grad = False

    def forward(self, data):
        vgg_output = self.vgg16_enc(data)
        return [vgg_output]
    
class deepShallowVGG(nn.Module):
    '''
        VGG Discriminator, with multiple scales.
    '''
    def __init__(self, opts, cfg):
        super(deepShallowVGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_shallow = torch.nn.Sequential(*list(vgg16.children())[0][:12])
        self.vgg16_deep = torch.nn.Sequential(*list(vgg16.children())[0][12:26])
        for param in vgg16.parameters():
            param.requires_grad = False

    def forward(self, data):
        feat = self.vgg16_shallow(data)
        vgg16_shallow = F.adaptive_avg_pool2d(feat, (7,7))
        vgg16_deep = self.vgg16_deep(feat)
        return [vgg16_shallow, vgg16_deep]

class deepShallowVGGPatch(nn.Module):
    '''
        VGG Discriminator, with multiple scales, add a patchwise discriminator.
    '''
    def __init__(self, opts, cfg):
        super(deepShallowVGGPatch, self).__init__()
        self.conv_1 = ConvModule(6, 32, 3, 2, 1)  # 256 -> 128
        self.conv_2 = ConvModule(32, 64, 5, 2, 2) # 128 -> 64
        self.conv_3 = ConvModule(64, 64, 5, 2, 2) # 64  -> 32
        self.conv_4 = ConvModule(64, 32, 3, 2, 1) # 32  -> 16
        self.conv_5 = ConvModule(32, 32, 3, 2, 1) # 16  -> 8
        self.conv_6 = ConvModule(32, 32, 3, 2, 1) # 8  -> 4
        self.patch_fc = nn.Conv2d(32, 1, 3, 2, 1) # 4  -> 2
        self.category_fc = nn.Linear(32*4*4, cfg.embedding.embedding_num)

    def forward(self, data):
        # discriminator path
        b,c,h,w = data.shape
        x = self.conv_1(data)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        patch_size_discrim = self.patch_fc(x)
        
        return F.sigmoid(patch_size_discrim)

class discrim_zi2zi(nn.Module):
    '''
        Discriminator for zi2zi.
    '''
    def __init__(self, opts, cfg):
        super(discrim_zi2zi, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.conv_1 = ConvModule(self.cfg.in_channels + self.cfg.out_channels, 32, 3, 2, 1)  # 256 -> 128
        self.conv_2 = ConvModule(32, 64, 5, 2, 2) # 128 -> 64
        self.conv_3 = ConvModule(64, 64, 5, 2, 2) # 64  -> 32
        self.conv_4 = ConvModule(64, 32, 3, 2, 1) # 32  -> 16
        self.patch_fc = nn.Conv2d(32, 1, 3, 2, 1) # 16  -> 8
        self.category_fc = nn.Linear(32*16*16, cfg.embedding.embedding_num)

    def forward(self, concat_real_fake):
        b,c,h,w = concat_real_fake.shape
        x = self.conv_1(concat_real_fake)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        patch_size_discrim = self.patch_fc(x)
        category_discrim = self.category_fc(x.view(b,-1))
        # print(category_discrim.shape)
        return F.sigmoid(patch_size_discrim), category_discrim

if __name__ == "__main__":
    VGG = deepShallowVGG(0, 0).cuda()
    rand_input = torch.FloatTensor(np.random.rand(16,3,256,256)).cuda()
    out1, out2 = VGG(rand_input)
    print(out1.shape, out2.shape)