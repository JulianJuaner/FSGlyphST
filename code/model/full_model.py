import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

def build_model(model_cfg, full_cfg):
    if model_cfg.type == "Z2Z":
        return Z2Z(model_cfg, full_cfg):

class Z2Z(nn.Module):
    '''
        Implementation of the Zi2Zi original module by pytorch.
    '''
    def __init__(self, opts, cfg):
        super(Z2Z, self).__init__()
        self.opts = opts
        self.cfg = cfg
        

