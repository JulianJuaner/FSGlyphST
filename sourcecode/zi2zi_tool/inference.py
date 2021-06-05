from sourcecode.model import build_model
from sourcecode.dataset import build_dataset
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate, compute_metric
from sourcecode.utils.metrics import Metrics
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import torch
import math
import os
import cv2


def inference(cfg):
    test_dataset = build_dataset(cfg.DATA.eval_data, cfg)

    test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    
    model = build_model(cfg.MODEL, cfg).train()
    model.cuda()
    metric = Metrics()

    if cfg.EVAL.epoch != -1:
        print("loading from ckpt {}".format(cfg.EVAL.epoch))
        start = cfg.EVAL.epoch
        model.load_state_dict(torch.load('{}ckpt{}.pth'.format(
                            cfg.FOLDER,
                            cfg.EVAL.epoch)), strict=False)

    test_iter = iter(test_loader)
    os.makedirs('{}inference_vis'.format(cfg.FOLDER), exist_ok=True)


    model.eval()
    os.makedirs('{}{}'.format(cfg.FOLDER, cfg.EVAL.epoch), exist_ok=True)
    epoch = cfg.EVAL.epoch
    out_log = os.path.join(cfg.FOLDER, 'out.csv')
    with open(out_log, 'wb') as log_file:
    
        for test_step in range(len(test_iter)):
            test_data = next(test_iter)
            res = model.evaluate(test_data)
            target = test_data["targets"]
            ssim, lpips, pix_hit, pix_total, pix_acc = metric.update_dict(res.detach(), target.cuda())
            print(str(ssim) + "," + str(lpips) + "," + str(pix_acc))
            log_file.write((str(ssim) + "," + str(lpips) + "," + str(pix_acc) + '\n').encode())
            if test_step % 1 == 0:
                vis_image = res.cpu().detach().numpy()[0,0,:,:]
                target = test_data["targets"].cpu().detach().numpy()[0,0,:,:]
                cv2.imwrite('{}inference_vis/{}vis.png'.format(cfg.FOLDER, test_step), 255*(vis_image))
                # cv2.imwrite('{}inference_vis/{}vis_target.png'.format(cfg.FOLDER, test_step), 255*(target))
        
    evaluation_summary = metric.summary()
    for key in evaluation_summary.keys():
        print(key,  '%.5f'%(evaluation_summary[key]), end =" ")
    print('end of evaluation.')

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
    inference(exp_cfg)