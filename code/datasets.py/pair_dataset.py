import os
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

'''
    Paired dataset.
'''
class PairedDataset(Dataset):
    def __init__(self, args):
        self.args = args