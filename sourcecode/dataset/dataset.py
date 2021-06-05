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
    elif dataset_cfg.type == 'FewShotFontImgDataset':
        return FewShotFontImgDataset(dataset_cfg, full_cfg)
    elif dataset_cfg.type == 'DoubleDistanceDataset':
        return DoubleDistanceDataset(dataset_cfg, full_cfg)
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
        self.embedding_num = full_cfg.MODEL.embedding.embedding_num
        self.data_path = opts.data_path
        self.data_cat_ids = []
        for _ in range(30):
            self.data_cat_ids.append([])
        self.data_list = self._read_data_list(self.data_path)
        self.full_cfg = full_cfg
        self.mean = full_cfg.DATA.mean
        self.std = full_cfg.DATA.std
        self.norm = TorchNormalize(self.mean, self.std)
        self.value_scale = full_cfg.DATA.value_scale
        self.shuf_prob = 0.5
        self.glyph_value = self._gen_glyph_vector(self.opts.glyph_path)
        if hasattr(opts, 'shuf_prob'):
            print("setting shuffle probability")
            self.shuf_prob = opts.shuf_prob

    def _gen_glyph_vector(self, data_path):
        data_list = dict()
        with open(data_path, 'r') as r:
            for line in r.readlines():
                line = line.strip().split(' ')
                if len(line) == 0:
                    continue
                vector = [int(i) for i in line]
                data_list[str(vector[0])] = vector[1:]
        return data_list

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
                    self.data_cat_ids[font_id].append(data_path[-9:])
                    if font_id < self.embedding_num:
                        data_list.append((type_id, font_id, data_path))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        type_id, font_id, img_path = self.data_list[item]
        same_type_idx = self.data_cat_ids[font_id][random.randint(0, len(self.data_cat_ids[font_id])-1)]
        if self.root is not None:
            target_path = '{}/{}'.format(self.root, img_path)
            new_target_path = target_path.replace(target_path[-9:], same_type_idx)
            # print(new_target_path, same_type_idx)
            img_path = "{}/{}/{}".format(self.root, self.full_cfg.DATA.origin_type, img_path[-9:])

        image = 255 - cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,-1]
        target_image = 255 - cv2.imread(target_path, cv2.IMREAD_UNCHANGED)[:,:,-1]
        new_target_image = 255 - cv2.imread(new_target_path, cv2.IMREAD_UNCHANGED)[:,:,-1]
        if image is None:
            logger.error("{} doesn't exist.".format(img_path))
            raise FileExistsError(img_path)
        
        for func in self.full_cfg.DATA.preprocessor:
            image = custom_transform(func, image)

        if 'training' in self.mode:
            for func in self.full_cfg.TRAIN.data_argumentation:
                image = custom_transform(func, image)

        distance_map = "s"
        # generate distance map.
        if 'distance' in self.mode:
            if 'double_distance' in self.mode:
                distance_map_front = cv2.distanceTransform(255 - target_image, cv2.DIST_L1, 5)
                distance_map_back = cv2.distanceTransform(target_image, cv2.DIST_L1, 5)/5
                distance_map = torch.from_numpy(((1 + (distance_map_back + distance_map_front)/30))[np.newaxis, ...]).float()
            else:
                distance_map = cv2.distanceTransform(255 - target_image, cv2.DIST_L1, 5)
                distance_map = 10*distance_map.astype(np.float32) + \
                    cv2.GaussianBlur(255-target_image,(9,9),0).astype(np.float32)
                distance_map = torch.from_numpy(((1 + distance_map/255)**2)[np.newaxis, ...]).float()
            

        if self.full_cfg.MODEL.in_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
            new_target_image = cv2.cvtColor(new_target_image, cv2.COLOR_GRAY2BGR)
            image = self.norm(torch.from_numpy(image.transpose(2,0,1)).float())/self.value_scale
            target_image = self.norm(torch.from_numpy(target_image.transpose(2,0,1)).float())/self.value_scale
            new_target_image = self.norm(torch.from_numpy(new_target_image.transpose(2,0,1)).float())/self.value_scale

        else:
            image = torch.from_numpy(image[np.newaxis, ...]).float()/255
            target_image = torch.from_numpy(target_image[np.newaxis, ...]).float()/255
            new_target_image = torch.from_numpy(new_target_image[np.newaxis, ...]).float()/255

        # adjustment of the shuffle probability.
        if self.shuf_prob > random.random():
            shuf_cat_id = torch.LongTensor([random.randint(0, self.embedding_num-1)])
        else:
            shuf_cat_id = torch.LongTensor([font_id])
        glyph_vector = self.glyph_value[str(type_id)]
        glyph_vector = torch.FloatTensor([glyph_vector])

        #print(glyph_vector.shape)
        return {
            'glyph_vector':
            glyph_vector,
            'imgs':
            image.float(),
            'shuf_input_img':
            image.float(),
            'targets':
            target_image.float(),
            'same_cat_targets':
            new_target_image.float(),
            'cat_id':
            torch.LongTensor([font_id]),
            'shuf_cat_id':
            shuf_cat_id,
            'type_id':
            torch.LongTensor([type_id]),
            'img_path':
            img_path,
            'distance_map':
            distance_map
        }

class TorchNormalize:

    def __init__(self, mean, std, **kwargs):
        self.mean = mean if type(mean) is list else [mean]
        self.std = std if type(std) is list else [std]

    def __call__(self, img, **kwargs):
        img = transforms.Normalize(self.mean, self.std)(img)
        return img

# Dataset for the few-shot setting.
class FewShotFontImgDataset(FontImgDataset):
    def __init__(self, opts, full_cfg):
        super(FewShotFontImgDataset, self).__init__(opts, full_cfg)
        self.data_list = self._read_data_list(self.data_path)

    def _read_data_list(self, data_path):
        balance_factor = self.opts.balance_factor
        few_shot_factor = self.opts.few_shot_factor
        few_shot_cats = self.opts.few_shot_cats
        all_cats = self.opts.all_cats
        counter = 0
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
                    
                    self.data_cat_ids[font_id].append(data_path[-9:])
                    if font_id < all_cats:
                        if font_id < few_shot_cats:
                            counter += 1
                            if counter%few_shot_factor == 0:
                                for i in range(balance_factor):
                                    data_list.append((type_id, font_id, data_path))
                        else:
                            counter = 0
                            data_list.append((type_id, font_id, data_path))
        return data_list