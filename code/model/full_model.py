import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from code.model.head import build_head
from code.model.utils import build_norm, ConvModule
from code.model.backbone import build_backbone
from code.model.embedding import build_embedding

from code.configs.make_cfg import Struct

def build_model(model_cfg, full_cfg):
    if model_cfg.type == "Z2Z":
        return Z2Z(model_cfg, full_cfg)
    else:
        raise NotImplementedError

def build_discriminator(discriminator_cfg, model_cfg):
    if model_cfg.type == "zi2zi":
        return discrim_zi2zi(discriminator_cfg, model_cfg)
    else:
        raise NotImplementedError

class discrim_zi2zi(nn.Module):
    '''
        Discriminator for zi2zi.
    '''
     def __init__(self, opts, cfg):
        super(discrim_zi2zi, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.conv_1 = ConvModule(2, 32, 3, 2, 1)  # 256 -> 128
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
        category_discrim = self.category_fc(x).view(b,1,1,-1)
        return F.sigmoid(patch_size_discrim), F.softmax(category_discrim, 2)

class Z2Z(nn.Module):
    '''
        Implementation of the Zi2Zi original module by pytorch.
    '''
    def __init__(self, opts, cfg):
        super(Z2Z, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.encoder = build_backbone(opts.backbone, opts)
        self.decoder = build_head(opts.decoder, opts)
        self.discriminator = build_discriminator(opts.discriminator, opts)
        self.embedding = build_embedding(opts.embedding, opts)
        self.in_channels = 1
        self.out_channels = 1
        self.loss_func_dict = dict()

        for item in opts.loss:
            item = Struct(**item)
            self.loss_func_dict[item.type] = float(item.weight)
            
    def forward(self, data):
        input_img = data['input_img']
        cat_id = data['cat_id']
        fake_id = data['fake_id']
        target_img = data['target_img']

        feat_list, global_embed = self.backbone(input_img)
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand_as(feat_list[-1])
        batch_embed = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(batch_embed)
        real_patch_disrm, real_cate_discim = self.discriminator(torch.cat((input_img, target_img), 1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((fake_img, target_img), 1))

        _, real_embed = self.backbone(target_img)
        _, cycle_fake_embed = self.backbone(fake_img)

        const_loss = 





        




