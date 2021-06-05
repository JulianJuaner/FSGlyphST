from sourcecode.dataset import build_dataset
from sourcecode.configs import make_config, Options
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from piq import FID, IS
import argparse
import numpy as np
import os
import cv2
import torch

# feature based evaluation.
class TestDataset(Dataset):
    def __init__(self, opts, full_cfg):
        self.root = os.path.join(full_cfg.FOLDER, "inference_vis")
        self.data_list = sorted(os.listdir(self.root))[:7029]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.data_list[item])
        image = cv2.imread(img_path, 1)
        #print(image.shape)
        image = torch.from_numpy(image.transpose(2,0,1)).float()/255
        return {
            'images':
            image.float(),
            }

class GTDataset(Dataset):
    def __init__(self, opts, full_cfg):
        self.root = os.path.join(".", "./data/test_GT")
        self.data_list = sorted(os.listdir(self.root))[:7029]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.data_list[item])
        image = cv2.imread(img_path, 1)
        image = torch.from_numpy(image.transpose(2,0,1)).float()/255
        return {
            'images':
            image.float(),
            }

def evaluate(cfg):
    # metric = Metrics()
    test_dataset = TestDataset(cfg.DATA.test_data, cfg)
    GT_dataset = GTDataset(cfg.DATA.test_data, cfg)
    test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
    )
    gt_loader = DataLoader(
            GT_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
    )

    is_metric = IS(num_splits=1)
    print("computing IS Score...")
    first_feats = is_metric.compute_feats(gt_loader)
    second_feats = is_metric.compute_feats(test_loader)
    _is = is_metric(first_feats, second_feats)
    print("IS score:", _is)

    fid_metric = FID()
    print("computing FID Score...")
    fid = fid_metric(first_feats, second_feats)
    print("FID score:", fid)


    
if "__main__" in __name__:
    # initialize exp configs.
    parser = argparse.ArgumentParser()
    OptionInit = Options(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    folder_name = opt.exp
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))

    # modification: update cfg for different root settings.
    from sourcecode.configs.profile_configs import PROFILE_ROOTS, PROFILE_PRETRAIN

    # replace pretrain.
    if hasattr(exp_cfg.MODEL, 'backbone') and hasattr(exp_cfg.MODEL.backbone, 'weights'):
        if opt.profile in PROFILE_PRETRAIN.keys():
            exp_cfg.MODEL.backbone.weights = exp_cfg.MODEL.backbone.weights.replace(
                PROFILE_PRETRAIN['default'], PROFILE_PRETRAIN[opt.profile])
    
    # replace root.
    exp_cfg.DATA.train_data.root = exp_cfg.DATA.train_data.root.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    exp_cfg.DATA.eval_data.root = exp_cfg.DATA.eval_data.root.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    exp_cfg.DATA.test_data.root = exp_cfg.DATA.test_data.root.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
                
    exp_cfg.DATA.train_data.glyph_path = exp_cfg.DATA.train_data.glyph_path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    exp_cfg.DATA.eval_data.glyph_path = exp_cfg.DATA.eval_data.glyph_path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    exp_cfg.DATA.test_data.glyph_path = exp_cfg.DATA.test_data.glyph_path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    # replace data list root.
    for i, path in enumerate(exp_cfg.DATA.train_data.data_path):
        exp_cfg.DATA.train_data.data_path[i] = path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    for i, path in enumerate(exp_cfg.DATA.eval_data.data_path):
        exp_cfg.DATA.eval_data.data_path[i] = path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    for i, path in enumerate(exp_cfg.DATA.test_data.data_path):
        exp_cfg.DATA.test_data.data_path[i] = path.replace(
                PROFILE_ROOTS['default'], PROFILE_ROOTS[opt.profile])
    print(exp_cfg)
    # inference model.
    evaluate(exp_cfg)