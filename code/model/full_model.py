import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from code.model.head import build_head
from code.model.backbone import build_backbone
from code.model.embedding import build_embedding

from code.configs.make_cfg import Struct

def build_model(model_cfg, full_cfg):
    if model_cfg.type == "Z2Z":
        return Z2Z(model_cfg, full_cfg)

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
        self.discriminator = self.build_discriminator
        self.embedding = build_embedding(opts.embedding, opts)
        self.in_channels = 1
        self.out_channels = 1
        self.loss_func_dict = dict()

        for item in opts.loss:
            item = Struct(**item)
            self.loss_func_dict[item.type] = float(item.weight)

    def build_discriminator(self):
        pass
    def __forward__(self, data):
        pass

        




