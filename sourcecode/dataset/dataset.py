import sys
import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sourcecode.dataset.transform import custom_transform

def build_dataset(dataset_cfg, full_cfg):
    if dataset_cfg.type == 'FontImgDataset':
        return FontImgDataset(dataset_cfg, full_cfg)
    else:
        raise NotImplementedError


class FontImgDataset(Dataset):
    """
    Dataset for zi2zi translation (r2v).
    """

    def __init__(self, opts, full_cfg):
        """
        Args:
            data_name: dataset name
            data_path: list of data path
            root: the root path, it will be used as prefix for each image path and json path.
            **kwargs:
        """
        self.opts = opts
        self.root = self.opts.root
        self.mode = self.opts.mode
        self.data_path = opts.data_path
        self.data_list = self._read_data_list(self.data_path)
        self.full_cfg = full_cfg
        self.mean = full_cfg.DATA.mean
        self.std = full_cfg.DATA.std
        self.value_scale = full_cfg.DATA.value_scale

    def _read_data_list(self, data_path):
        data_list = []
        for data_file in data_path:
            with open(data_file, 'r') as r:
                for line in r.readlines():
                    line = line.strip().split(' ')
                    if len(line) == 0:
                        continue
                    type_id = int(line[0])
                    font_id = int(line[1])
                    data_path = line[2]
                    data_list.append((type_id, font_id, data_path))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        type_id, font_id, img_path = self.data_list[item]
        if self.root is not None:
            img_path = '{}/{}'.format(self.root, img_path)
            target_path = '{}/{}/{}'.format(self.root, self.full_cfg.DATA.origin_type, img_path[-9:])
        image = cv2.imread(img_path, 0)
        target_image = cv2.imread(target_path, 0)

        if image is None:
            logger.error("{} doesn't exist.".format(img_path))
            raise FileExistsError(img_path)
        
        for func in self.full_cfg.DATA.preprocessor:
            image = custom_transform(func, image)

        if self.mode == 'training':
            for func in self.full_cfg.TRAIN.data_argumentation:
                image = custom_transform(func, image)

        image = torch.from_numpy(image[np.newaxis, ...]).float()/255
        target_image = torch.from_numpy(target_image[np.newaxis, ...]).float()/255

        return {
            'imgs':
            image.float(),
            'targets':
            target_image.float(),
            'cat_id':
            torch.LongTensor([font_id]),
            'type_id':
            torch.LongTensor([type_id]),
            'img_path':
            img_path
        }

class TorchNormalize:

    def __init__(self, mean, std, **kwargs):
        self.mean = mean if type(mean) is list else [mean]
        self.std = std if type(std) is list else [std]

    def __call__(self, img, **kwargs):
        img = transforms.Normalize(self.mean, self.std)(img)
        return img

