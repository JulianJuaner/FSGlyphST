import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.head import build_head
from sourcecode.model.utils import build_norm, ConvModule
from sourcecode.model.backbone import build_backbone
from sourcecode.model.embedding import build_embedding

from sourcecode.configs.make_cfg import Struct

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
        category_discrim = self.category_fc(x.view(b,-1))
        # print(category_discrim.shape)
        return F.sigmoid(patch_size_discrim), category_discrim

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
        self.embedding = build_embedding(opts.embedding, opts)
        self.generator = nn.ModuleList()
        self.generator.append(self.encoder)
        self.generator.append(self.decoder)
        self.generator.append(self.embedding)
        self.discriminator = discrim_zi2zi(opts.discriminator, opts)
        self.in_channels = 1
        self.out_channels = 1
        self.loss_func_dict = dict()
    
        self.CE = nn.CrossEntropyLoss() # category loss, cheat loss
        self.L1Distance = nn.L1Loss() # L1 distance, constant loss
        self.BCE = nn.BCELoss()
        #self.gen_optimizer = optim.Adam(model.generator.parameters(), lr=cfg.TRAIN.optimizer.lr, weight_decay=0.0005)
        #self.dis_optimizer = optim.Adam(model.discriminator.parameters(), lr=cfg.TRAIN.optimizer.lr, weight_decay=0.0005)
    
        for item in opts.loss:
            item = Struct(**item)
            self.loss_func_dict[item.type] = float(item.weight)
        
    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        batch_embed = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(batch_embed)
        real_patch_disrm, real_cate_discim = self.discriminator(torch.cat((input_img, target_img), 1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img.detach()), 1))

        _, real_embed = self.encoder(target_img)
        _, cycle_fake_embed = self.encoder(fake_img)

        # loss function computation.
        
        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Distance(real_embed, cycle_fake_embed),2)
        real_category_loss  = self.CE(real_cate_discim, cat_id.view(-1))
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        category_loss       = self.loss_func_dict['Category']*(real_category_loss + fake_category_loss)
        l1_loss             = self.loss_func_dict['L1']*self.L1Distance(fake_img, target_img)
        cheat_loss          = self.loss_func_dict['Cheat']*self.BCE(fake_patch_disrm, torch.ones_like(fake_patch_disrm))
        discrim_loss        = self.BCE(real_patch_disrm, torch.ones_like(real_patch_disrm)) + \
                              self.BCE(fake_patch_disrm, torch.zeros_like(fake_patch_disrm))

        loss_g = cheat_loss + l1_loss + self.loss_func_dict['Category']* fake_category_loss + const_loss # + tv_loss
        loss_d = discrim_loss + category_loss/2

        loss_dict = dict()
        loss_dict['cheat_loss'] = cheat_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['fake_category_loss'] = fake_category_loss.item()
        loss_dict['real_category_loss'] = real_category_loss.item()
        loss_dict['discrim_loss'] = discrim_loss.item()
        loss_dict['const_loss'] = const_loss.item()
        loss_dict['loss_g'] = loss_g.item()
        loss_dict['loss_d'] = loss_d.item()

        return loss_d, loss_g, fake_img, loss_dict






        




