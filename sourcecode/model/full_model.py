import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.head import build_head
from sourcecode.model.utils import build_norm, ConvModule
from sourcecode.model.backbone import build_backbone
from sourcecode.model.embedding import build_embedding
from sourcecode.model.discriminator import build_discrim, deepShallowVGG
from torch import optim
from sourcecode.configs.make_cfg import Struct

def build_model(model_cfg, full_cfg):
    if model_cfg.type == "Z2Z":
        return Z2Z(model_cfg, full_cfg)
    elif model_cfg.type == "Z2Z_Distance":
        return Z2Z_Distance(model_cfg, full_cfg)
    elif model_cfg.type == "Z2Z_Type":
        return Z2Z_Type(model_cfg, full_cfg)
    elif model_cfg.type == "Z2Z_Type_Distance":
        return Z2Z_Type_Distance(model_cfg, full_cfg)
    elif model_cfg.type == "Z2Z_Glyph_Distance":
        return Z2Z_Glyph_Distance(model_cfg, full_cfg)
    elif model_cfg.type == "VGG_discrim":
        return VGG_discrim(model_cfg, full_cfg)
    elif model_cfg.type == "VGG_Embed":
        return VGG_Embed(model_cfg, full_cfg)
    elif model_cfg.type == "VGG_discrim_full":
        return VGG_discrim_full(model_cfg, full_cfg)
    elif model_cfg.type == "VGG_discrim_new_comp":
        return VGG_discrim_new_comp(model_cfg, full_cfg)
    else:
        raise NotImplementedError

def build_discriminator(discriminator_cfg, model_cfg):
    if model_cfg.type == "zi2zi":
        return discrim_zi2zi(discriminator_cfg, model_cfg)
    else:
        raise NotImplementedError

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
        self.discriminator = build_discrim(opts.discriminator, opts)
        self.in_channels = self.opts.in_channels
        self.out_channels = self.opts.out_channels
        self.loss_func_dict = dict()
    
        self.CE = nn.CrossEntropyLoss() # category loss, cheat loss
        self.L1Distance = nn.L1Loss() # L1 distance, constant loss
        self.BCE = nn.BCELoss()

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.gen_lr_factor, betas=(0.5, 0.999))
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.dis_lr_factor, betas=(0.5, 0.999))
    
        for item in opts.loss:
            item = Struct(**item)
            self.loss_func_dict[item.type] = item.weight

    def L1Loss(self, pred, target, data):
        return self.L1Distance(pred, target)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def evaluate(self, data):
        input_img = data['imgs'].cuda()
        cat_id = data['cat_id'].cuda()
        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)
        return fake_img

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)

        self.set_requires_grad(self.discriminator, True)
        self.dis_optimizer.zero_grad()
        
        real_patch_disrm, real_cate_discim = self.discriminator(torch.cat((input_img, target_img), 1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1).detach())

        real_category_loss  = self.CE(real_cate_discim, cat_id.view(-1))
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        category_loss       = self.loss_func_dict['Category']*(real_category_loss + fake_category_loss)
        discrim_loss        = self.BCE(real_patch_disrm, torch.ones_like(real_patch_disrm)) + \
                              self.BCE(fake_patch_disrm, torch.zeros_like(fake_patch_disrm))

        if self.opts.fake_emb:
            shuf_cat_id = data['shuf_cat_id'].cuda()
            shuf_cat_embed = self.embedding(shuf_cat_id)
            
            shuf_cat_embed = shuf_cat_embed.expand(-1, -1, 8, 8)
            shuf_input_img = data['shuf_input_img'].cuda()
            shuf_feat_list, _ = self.encoder(shuf_input_img)
            shuf_input_embed = shuf_feat_list[-1]
            shuf_feat_list[-1] = torch.cat((shuf_feat_list[-1], shuf_cat_embed), 1)
            shuf_res = self.decoder(shuf_feat_list)
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1).detach())
            cycle_shuf_embed, _ = self.encoder(shuf_res)

            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))
            d_loss_no_target        = self.BCE(shuf_patch_disrm, torch.zeros_like(shuf_patch_disrm))
            category_loss  += self.loss_func_dict['Category']*(real_category_loss + fake_category_loss + no_target_category_loss)
            discrim_loss +=  d_loss_no_target
            loss_d = discrim_loss + category_loss/3.0
            
        else:
            loss_d = discrim_loss + category_loss/2


        loss_d.backward()
        self.dis_optimizer.step()
        # G update.
        self.set_requires_grad(self.discriminator, False)
        self.gen_optimizer.zero_grad()        # set G's gradients to zero
        
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1))
        cycle_fake_embed, _ = self.encoder(fake_img)
        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        cheat_loss          = self.loss_func_dict['Cheat']*self.BCE(fake_patch_disrm, torch.ones_like(fake_patch_disrm))

        if self.opts.fake_emb:
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1))
            cheat_loss += self.loss_func_dict['Cheat']*self.BCE(shuf_patch_disrm, torch.ones_like(shuf_patch_disrm))
            no_target_const_loss    = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(shuf_input_embed, cycle_shuf_embed[-1], data),2)
            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))

            loss_g = cheat_loss/2 + l1_loss + (self.loss_func_dict['Category']* fake_category_loss + \
                no_target_category_loss)/2 + (const_loss + no_target_const_loss)/2 # + tv_loss

        else:
            loss_g = cheat_loss + l1_loss + self.loss_func_dict['Category']* fake_category_loss  + const_loss
            
        loss_g.backward()                   # calculate graidents for G
        self.gen_optimizer.step()   



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

class Z2Z_Distance(Z2Z):
    '''
        Implementation of the Zi2Zi original module by pytorch.
        Modification: add weighted loss for the L1 loss
    '''
    def __init__(self, opts, cfg):
        super(Z2Z_Distance, self).__init__(opts, cfg)
        self.L1Distance = nn.L1Loss(reduction='none')

    def L1Loss(self, pred, target, data):
        chn = pred.shape[1]
        loss_map =  self.L1Distance(pred, target)

        if chn == self.out_channels and not isinstance(data['distance_map'], list):
            distance_map = data['distance_map'].cuda()
            return torch.mean(distance_map*loss_map)
        else:
            return torch.mean(loss_map)

class Z2Z_Type(Z2Z):
    '''
        Implementation of the Zi2Zi original module by pytorch.
        Modification: type loss after backbone.
    '''
    def __init__(self, opts, cfg):
        super(Z2Z_Type, self).__init__(opts, cfg)
        self.font_size = opts.font_size
        self.fc_out_size = 40960
        # print(self.fc_out_size, self.font_size)
        self.type_fc = nn.Linear(self.fc_out_size, opts.font_size)

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        b,c,h,w = input_img.shape
        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        pred_type_id = self.type_fc(feat_list[-1].view(b,-1))
        fake_img = self.decoder(feat_list)

        self.set_requires_grad(self.discriminator, True)
        self.dis_optimizer.zero_grad()
        
        real_patch_disrm, real_cate_discim = self.discriminator(torch.cat((input_img, target_img), 1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1).detach())

        real_category_loss  = self.CE(real_cate_discim, cat_id.view(-1))
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        category_loss       = self.loss_func_dict['Category']*(real_category_loss + fake_category_loss)
        discrim_loss        = self.BCE(real_patch_disrm, torch.ones_like(real_patch_disrm)) + \
                              self.BCE(fake_patch_disrm, torch.zeros_like(fake_patch_disrm))

        if self.opts.fake_emb:
            shuf_cat_id = data['shuf_cat_id'].cuda()
            shuf_cat_embed = self.embedding(shuf_cat_id)
            
            shuf_cat_embed = shuf_cat_embed.expand(-1, -1, 8, 8)
            shuf_input_img = data['shuf_input_img'].cuda()
            shuf_feat_list, _ = self.encoder(shuf_input_img)
            shuf_input_embed = shuf_feat_list[-1]
            shuf_feat_list[-1] = torch.cat((shuf_feat_list[-1], shuf_cat_embed), 1)
            shuf_res = self.decoder(shuf_feat_list)
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1).detach())
            cycle_shuf_embed, _ = self.encoder(shuf_res)

            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))
            d_loss_no_target        = self.BCE(shuf_patch_disrm, torch.zeros_like(shuf_patch_disrm))
            category_loss  += self.loss_func_dict['Category']*(real_category_loss + fake_category_loss + no_target_category_loss)
            discrim_loss +=  d_loss_no_target
            loss_d = discrim_loss + category_loss/3.0
            
        else:
            loss_d = discrim_loss + category_loss/2


        loss_d.backward()
        self.dis_optimizer.step()
        # G update.
        self.set_requires_grad(self.discriminator, False)
        self.gen_optimizer.zero_grad()        # set G's gradients to zero
        
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1))
        cycle_fake_embed, _ = self.encoder(fake_img)
        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        cheat_loss          = self.loss_func_dict['Cheat']*self.BCE(fake_patch_disrm, torch.ones_like(fake_patch_disrm))
        type_loss           = self.loss_func_dict['Type']*self.CE(pred_type_id, type_id.view(-1))

        if self.opts.fake_emb:
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1))
            cheat_loss += self.loss_func_dict['Cheat']*self.BCE(shuf_patch_disrm, torch.ones_like(shuf_patch_disrm))
            no_target_const_loss    = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(shuf_input_embed, cycle_shuf_embed[-1], data),2)
            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))

            loss_g = type_loss + cheat_loss/2 + l1_loss + (self.loss_func_dict['Category']* fake_category_loss + \
                no_target_category_loss)/2 + (const_loss + no_target_const_loss)/2 # + tv_loss

        else:
            loss_g = type_loss + cheat_loss + l1_loss + self.loss_func_dict['Category']* fake_category_loss  + const_loss
            
        loss_g.backward()                   # calculate graidents for G
        self.gen_optimizer.step()   



        loss_dict = dict()
        loss_dict['cheat_loss'] = cheat_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['fake_category_loss'] = fake_category_loss.item()
        loss_dict['real_category_loss'] = real_category_loss.item()
        loss_dict['discrim_loss'] = discrim_loss.item()
        loss_dict['const_loss'] = const_loss.item()
        loss_dict['type_loss'] = type_loss.item()
        loss_dict['loss_g'] = loss_g.item()
        loss_dict['loss_d'] = loss_d.item()
        
        return loss_d, loss_g, fake_img, loss_dict

class Z2Z_Type_Distance(Z2Z_Type):
    '''
        Implementation of the Zi2Zi original module by pytorch.
        Modification: add weighted loss for the L1 loss
    '''
    def __init__(self, opts, cfg):
        super(Z2Z_Type_Distance, self).__init__(opts, cfg)
        self.L1Distance = nn.L1Loss(reduction='none')

    def L1Loss(self, pred, target, data):
        chn = pred.shape[1]
        loss_map =  self.L1Distance(pred, target)

        if chn == self.out_channels:
            distance_map = data['distance_map'].cuda()
            return torch.mean(distance_map*loss_map)
        else:
            return torch.mean(loss_map)

class Z2Z_Glyph_Distance(Z2Z_Distance):
    '''
        Add Glyph One-Hot Vector.
    '''
    def __init__(self, opts, cfg):
        super(Z2Z_Glyph_Distance, self).__init__(opts, cfg)
        self.glyph_conv = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.Conv2d(256, opts.glyph_embedding_num, 1, 1, 0),
        )
        self.generator.append(self.glyph_conv)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.gen_lr_factor, betas=(0.5, 0.999))
        

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        target_glyph = data['glyph_vector'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(real_embed, 1)))
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)

        self.set_requires_grad(self.discriminator, True)
        self.dis_optimizer.zero_grad()
        real_patch_disrm, real_cate_discim = self.discriminator(torch.cat((input_img, target_img), 1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1).detach())

        real_category_loss  = self.CE(real_cate_discim, cat_id.view(-1))
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        category_loss       = self.loss_func_dict['Category']*(real_category_loss + fake_category_loss)
        discrim_loss        = self.BCE(real_patch_disrm, torch.ones_like(real_patch_disrm)) + \
                              self.BCE(fake_patch_disrm, torch.zeros_like(fake_patch_disrm))

        if self.opts.fake_emb:
            shuf_cat_id = data['shuf_cat_id'].cuda()
            shuf_cat_embed = self.embedding(shuf_cat_id)
            
            shuf_cat_embed = shuf_cat_embed.expand(-1, -1, 8, 8)
            shuf_input_img = data['shuf_input_img'].cuda()
            shuf_feat_list, _ = self.encoder(shuf_input_img)
            shuf_input_embed = shuf_feat_list[-1]
            shuf_feat_list[-1] = torch.cat((shuf_feat_list[-1], shuf_cat_embed), 1)
            shuf_res = self.decoder(shuf_feat_list)
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1).detach())
            cycle_shuf_embed, _ = self.encoder(shuf_res)

            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))
            d_loss_no_target        = self.BCE(shuf_patch_disrm, torch.zeros_like(shuf_patch_disrm))
            category_loss  += self.loss_func_dict['Category']*(real_category_loss + fake_category_loss + no_target_category_loss)
            discrim_loss +=  d_loss_no_target
            loss_d = discrim_loss + category_loss/3.0
            
        else:
            loss_d = discrim_loss + category_loss/2


        loss_d.backward()
        self.dis_optimizer.step()
        # G update.
        self.set_requires_grad(self.discriminator, False)
        self.gen_optimizer.zero_grad()        # set G's gradients to zero
        glyph_loss = self.loss_func_dict['glyph']*2*self.BCE(glyph_embed.view(-1), target_glyph.view(-1))
        fake_patch_disrm, fake_cate_discim = self.discriminator(torch.cat((input_img, fake_img), 1))
        cycle_fake_embed, _ = self.encoder(fake_img)
        fake_glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(cycle_fake_embed[-1], 1)))
        glyph_loss += self.loss_func_dict['glyph']*self.BCE(fake_glyph_embed.view(-1), target_glyph.view(-1))
        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        fake_category_loss  = self.CE(fake_cate_discim, cat_id.view(-1))
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        cheat_loss          = self.loss_func_dict['Cheat']*self.BCE(fake_patch_disrm, torch.ones_like(fake_patch_disrm))

        if self.opts.fake_emb:
            shuf_patch_disrm, shuf_cate_discim = self.discriminator(torch.cat((shuf_input_img, shuf_res), 1))
            cheat_loss += self.loss_func_dict['Cheat']*self.BCE(shuf_patch_disrm, torch.ones_like(shuf_patch_disrm))
            no_target_const_loss    = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(shuf_input_embed, cycle_shuf_embed[-1], data),2)
            no_target_category_loss = self.loss_func_dict['Category']*self.CE(shuf_cate_discim, shuf_cat_id.view(-1))

            loss_g = glyph_loss + cheat_loss/2 + l1_loss + (self.loss_func_dict['Category']* fake_category_loss + \
                no_target_category_loss)/2 + (const_loss + no_target_const_loss)/2 # + tv_loss

        else:
            loss_g = glyph_loss + cheat_loss + l1_loss + self.loss_func_dict['Category']* fake_category_loss  + const_loss
            
        loss_g.backward()                   # calculate graidents for G
        self.gen_optimizer.step()   



        loss_dict = dict()
        loss_dict['glyph_loss'] = glyph_loss.item()
        loss_dict['cheat_loss'] = cheat_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['fake_category_loss'] = fake_category_loss.item()
        loss_dict['real_category_loss'] = real_category_loss.item()
        loss_dict['discrim_loss'] = discrim_loss.item()
        loss_dict['const_loss'] = const_loss.item()
        
        loss_dict['loss_g'] = loss_g.item()
        loss_dict['loss_d'] = loss_d.item()
        
        return loss_d, loss_g, fake_img, loss_dict

class VGG_discrim(Z2Z_Glyph_Distance):
    def __init__(self, opts, cfg):
        super(VGG_discrim, self).__init__(opts, cfg)
        self.L2loss = nn.MSELoss()

    def set_grad(self):
        self.set_requires_grad(self.discriminator, False)

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        target_glyph = data['glyph_vector'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(real_embed, 1)))
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)
        cycle_fake_embed, _ = self.encoder(fake_img)
        
        input_vgg_embed = self.discriminator(fake_img)
        target_vgg_embed = self.discriminator(target_img)
        self.set_grad()
        self.gen_optimizer.zero_grad()

        loss_dict = dict()

        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        perc_loss = 0
        for i in range(len(input_vgg_embed)):
            loss_i = self.L2loss(input_vgg_embed[i], target_vgg_embed[i])*self.loss_func_dict['VGG'][i]
            perc_loss += loss_i
            loss_dict['perc_loss_{}'.format(i)] = loss_i.item()

        glyph_loss = self.loss_func_dict['glyph']*2*self.BCE(glyph_embed.view(-1), target_glyph.view(-1))
        fake_glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(cycle_fake_embed[-1], 1)))
        glyph_loss += self.loss_func_dict['glyph']*self.BCE(fake_glyph_embed.view(-1), target_glyph.view(-1))
        glyph_const = self.loss_func_dict['glyph']*self.L2loss(fake_glyph_embed.view(-1), glyph_embed.view(-1))*3

        loss_g = glyph_loss  + l1_loss + const_loss + perc_loss + glyph_const
        loss_g.backward()
        self.gen_optimizer.step() 

        loss_dict['glyph_loss'] = glyph_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['glyph_const'] = glyph_const.item()
        loss_dict['loss_g'] = loss_g.item()
        
        return None, loss_g, fake_img, loss_dict

class VGG_discrim_new_comp(VGG_discrim):
    def __init__(self, opts, cfg):
        super(VGG_discrim_new_comp, self).__init__(opts, cfg)
        self.glyph_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2, 1),# 8->4
            nn.Conv2d(256, 256, 3, 2, 1),# 4->2
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 2, 1),# 2->1
            nn.Conv2d(256, 256, 1, 1, 0),# 1->1
            nn.Conv2d(256, opts.glyph_embedding_num, 1, 1, 0),
        )

        self.four_cor_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 2, 1),# 8->4
            nn.Conv2d(256, 256, 3, 2, 1),# 4->2
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1, 1, 0),# 2
            
        )
        self.center_index_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),# 8->4
            nn.Conv2d(128, 128, 1, 1, 0),# 4->2
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0),# 2
            nn.Conv2d(128, 10, 1, 1, 0),
        )
        self.generator.append(self.glyph_conv)
        self.generator.append(self.four_cor_head)
        self.generator.append(self.center_index_conv)
        self.CE = nn.CrossEntropyLoss()
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.gen_lr_factor, betas=(0.5, 0.999))
        
    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        target_glyph = data['glyph_vector'].cuda()
        b,c,h,w = input_img.shape
        
        gt_four_corner_index = target_glyph[:, 0, :5]
        gt_glyph_embed = target_glyph[:, :, 5:].contiguous()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]

        four_corner_index = self.four_cor_head(real_embed)
        glyph_embed = self.glyph_conv(real_embed)
        center_index = self.center_index_conv(F.adaptive_avg_pool2d(four_corner_index, 1))
        four_corner_index = self.center_index_conv(four_corner_index)

        four_corner_index = torch.cat((four_corner_index.view(b, 10, -1),
                                       center_index.view(b, 10, -1)), -1)

        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)
        cycle_fake_embed, _ = self.encoder(fake_img)
        
        input_vgg_embed = self.discriminator(fake_img)
        target_vgg_embed = self.discriminator(target_img)
        self.set_grad()
        self.gen_optimizer.zero_grad()

        loss_dict = dict()

        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        perc_loss = 0
        for i in range(len(input_vgg_embed)):
            loss_i = self.L2loss(input_vgg_embed[i], target_vgg_embed[i])*self.loss_func_dict['VGG'][i]
            perc_loss += loss_i
            loss_dict['perc_loss_{}'.format(i)] = loss_i.item()
        # print(glyph_embed.shape, gt_glyph_embed.shape)
        glyph_loss = self.loss_func_dict['glyph']*2*(self.CE(four_corner_index, gt_four_corner_index.long()))
        glyph_loss += self.loss_func_dict['glyph']*2*(self.L2loss(glyph_embed.view(-1), gt_glyph_embed.view(-1)))

        fake_four_corner_index = self.four_cor_head(cycle_fake_embed[-1])
        fake_glyph_embed = glyph_embed = self.glyph_conv(cycle_fake_embed[-1])
        fake_center_index = self.center_index_conv(F.adaptive_avg_pool2d(fake_four_corner_index, 1))
        fake_four_corner_index = self.center_index_conv(fake_four_corner_index)
        fake_four_corner_index = torch.cat((fake_four_corner_index.view(b, 10, -1),
                                       fake_center_index.view(b, 10, -1)), -1)

        # print(fake_glyph_embed.shape, fake_center_index.shape, fake_four_corner_index.shape)
        glyph_loss += self.loss_func_dict['glyph']*1*(self.CE(fake_four_corner_index, gt_four_corner_index.long()))
        glyph_loss += self.loss_func_dict['glyph']*1*(self.L2loss(fake_glyph_embed.view(-1), gt_glyph_embed.view(-1)))
        
        glyph_const = self.loss_func_dict['glyph']*self.L2loss(fake_glyph_embed.view(-1), glyph_embed.view(-1))*3
        glyph_const += self.loss_func_dict['glyph']*self.L2loss(fake_four_corner_index.view(-1), four_corner_index.view(-1))*3

        loss_g = glyph_loss  + l1_loss + const_loss + perc_loss + glyph_const
        loss_g.backward()
        self.gen_optimizer.step() 

        loss_dict['glyph_loss'] = glyph_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['glyph_const'] = glyph_const.item()
        loss_dict['loss_g'] = loss_g.item()
        
        return None, loss_g, fake_img, loss_dict
        
class VGG_discrim_full(VGG_discrim):
    def __init__(self, opts, cfg):
        super(VGG_discrim_full, self).__init__(opts, cfg)
        self.L2loss = nn.MSELoss()
        self.discriminator = build_discrim(opts.discriminator, opts)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.dis_lr_factor, betas=(0.5, 0.999))
        self.vgg_embed = deepShallowVGG(opts.discriminator, opts)

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        same_target_img = data['same_cat_targets'].cuda()
        target_glyph = data['glyph_vector'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(real_embed, 1)))
        cat_embed = self.embedding(cat_id)
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)
        cycle_fake_embed, _ = self.encoder(fake_img)

        self.set_requires_grad(self.discriminator, True)
        self.dis_optimizer.zero_grad()
        
        real_patch_disrm = self.discriminator(torch.cat((same_target_img, target_img), 1))
        fake_patch_disrm = self.discriminator(torch.cat((same_target_img, fake_img), 1).detach())

        discrim_loss        = self.BCE(real_patch_disrm, torch.ones_like(real_patch_disrm)) + \
                              self.BCE(fake_patch_disrm, torch.zeros_like(fake_patch_disrm))

        discrim_loss.backward()
        self.dis_optimizer.step()
        loss_dict = dict()
        self.set_requires_grad(self.discriminator, False)
        perc_loss = 0
        input_vgg_embed = self.vgg_embed(fake_img.detach())
        target_vgg_embed = self.vgg_embed(target_img)
        perc_loss = 0
    
        self.set_requires_grad(self.vgg_embed, False)
        self.gen_optimizer.zero_grad()

        fake_patch_disrm = self.discriminator(torch.cat((input_img, fake_img), 1))
        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        cheat_loss          = self.loss_func_dict['Cheat']*self.BCE(fake_patch_disrm, torch.ones_like(fake_patch_disrm))

        for i in range(len(input_vgg_embed)):
            loss_i = self.L2loss(input_vgg_embed[i], target_vgg_embed[i])*self.loss_func_dict['VGG'][i]
            perc_loss += loss_i
            loss_dict['perc_loss_{}'.format(i)] = loss_i.item()
        glyph_loss = 2*self.BCE(glyph_embed.view(-1), target_glyph.view(-1))
        fake_glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(cycle_fake_embed[-1], 1)))
        glyph_loss += self.BCE(fake_glyph_embed.view(-1), target_glyph.view(-1))
        glyph_const = self.L2loss(fake_glyph_embed.view(-1), glyph_embed.view(-1))*3

        loss_g = glyph_loss + l1_loss + const_loss + perc_loss + glyph_const + cheat_loss
        loss_g.backward()
        self.gen_optimizer.step()

        loss_dict['loss_d'] = discrim_loss.item()
        loss_dict['cheat_loss'] = cheat_loss.item()
        loss_dict['glyph_loss'] = glyph_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['glyph_const'] = glyph_const.item()
        loss_dict['loss_g'] = loss_g.item()
        
        return discrim_loss, loss_g, fake_img, loss_dict

class VGG_Embed(VGG_discrim):
    def __init__(self, opts, cfg):
        super(VGG_Embed, self).__init__(opts, cfg)
        self.L2loss = nn.MSELoss()
        self.generator = nn.ModuleList()
        self.generator.append(self.encoder)
        self.generator.append(self.decoder)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.TRAIN.optimizer.lr*cfg.TRAIN.gen_lr_factor, betas=(0.5, 0.999))

    def set_grad(self):
        self.set_requires_grad(self.discriminator, False)

    def forward(self, data):
        cat_id = data['cat_id'].cuda()
        type_id = data['type_id'].cuda()
        input_img = data['imgs'].cuda()
        target_img = data['targets'].cuda()
        target_glyph = data['glyph_vector'].cuda()

        feat_list, global_embed = self.encoder(input_img)
        real_embed = feat_list[-1]
        glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(real_embed, 1)))
        cat_embed = self.embedding(cat_id).detach()
        cat_embed = cat_embed.expand(-1, -1, 8, 8)
        feat_list[-1] = torch.cat((feat_list[-1], cat_embed), 1)
        fake_img = self.decoder(feat_list)
        cycle_fake_embed, _ = self.encoder(fake_img)
        
        input_vgg_embed = self.discriminator(fake_img)
        target_vgg_embed = self.discriminator(target_img)
        self.set_grad()
        self.gen_optimizer.zero_grad()

        loss_dict = dict()

        const_loss          = self.loss_func_dict['Constant']*torch.pow(self.L1Loss(real_embed, cycle_fake_embed[-1], data),2)
        l1_loss             = self.loss_func_dict['L1']*self.L1Loss(fake_img, target_img, data)
        perc_loss = 0
        for i in range(len(input_vgg_embed)):
            loss_i = self.L2loss(input_vgg_embed[i], target_vgg_embed[i])*self.loss_func_dict['VGG'][i]
            perc_loss += loss_i
            loss_dict['perc_loss_{}'.format(i)] = loss_i.item()

        glyph_loss = 2*self.BCE(glyph_embed.view(-1), target_glyph.view(-1))
        fake_glyph_embed = F.sigmoid(self.glyph_conv(F.adaptive_avg_pool2d(cycle_fake_embed[-1], 1)))
        glyph_loss += self.BCE(fake_glyph_embed.view(-1), target_glyph.view(-1))
        glyph_const = self.L2loss(fake_glyph_embed.view(-1), glyph_embed.view(-1))*3

        loss_g = glyph_loss  + l1_loss + const_loss + perc_loss + glyph_const
        loss_g.backward()
        self.gen_optimizer.step() 

        loss_dict['glyph_loss'] = glyph_loss.item()
        loss_dict['l1_loss'] = l1_loss.item()
        loss_dict['glyph_const'] = glyph_const.item()
        loss_dict['loss_g'] = loss_g.item()
        
        return None, loss_g, fake_img, loss_dict