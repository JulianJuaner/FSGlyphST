from sourcecode.model import build_model
from sourcecode.dataset import build_dataset
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate, compute_metric
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

def train(cfg):
    train_dataset = build_dataset(cfg.DATA.train_data, cfg)
    eval_dataset = build_dataset(cfg.DATA.eval_data, cfg)
    model = build_model(cfg.MODEL, cfg).train()
    model = model.cuda()
    start = 0
    if cfg.TRAIN.resume != -1:
        print("loading from ckpt {}".format(cfg.TRAIN.resume))
        start = cfg.TRAIN.resume
        model.load_state_dict(torch.load('{}ckpt{}.pth'.format(
                            cfg.FOLDER,
                            cfg.TRAIN.resume)), strict=False)

    
    train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.batch_size_per_gpu, 
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

    eval_loader = DataLoader(
            eval_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    
    # loss_func = nn.CrossEntropyLoss().cuda()
    num_epoch = cfg.TRAIN.max_epoch
    os.makedirs('{}train_vis'.format(cfg.FOLDER), exist_ok=True)

    train_iter = iter(train_loader)
    if not model.embedding.embed_init:
        print("loading pretrained embeddings (if has)...")
        pbar = tqdm(total=len(train_iter))
        for step in range(len(train_iter)):
            pbar.update(1)
            data = next(train_iter)
            model.embedding.update_initialized(data)
        print("normalizing pretrained embeddings (if has)...")
        model.embedding.normalize_initial_feat()

    for epoch in range(start, num_epoch+1):
        model.train()
        train_iter = iter(train_loader)
        adjust_learning_rate(model.gen_optimizer, epoch, cfg)
        adjust_learning_rate(model.dis_optimizer, epoch, cfg)
        for step in range(len(train_iter)):
            #adjust_learning_rate(enc_optimizer, iteration, cfg, cfg.TRAIN.enc_lr_factor)
            #adjust_learning_rate(dec_optimizer, iteration, cfg, cfg.TRAIN.dec_lr_factor)
            
            if epoch % cfg.TRAIN.eval_freq == 0 and epoch != start and step == 0:
                # evaluation.
                if epoch % cfg.TRAIN.ckpt_freq == 0:
                    print(epoch, 'saving model')
                    torch.save(model.state_dict(), '{}ckpt{}.pth'.format(
                            cfg.FOLDER,
                            epoch))

                model.eval()
                eval_iter = iter(eval_loader)
                os.makedirs('{}{}'.format(cfg.FOLDER, epoch), exist_ok=True)
                for eval_step in range(len(eval_iter)):
                    eval_data = next(eval_iter)
                    res = model.evaluate(eval_data)
                    vis_image = None
                    if eval_step % 25 == 0:
                        vis_image = res.cpu().detach().numpy()[0,0,:,:]
                        target = eval_data["targets"].cpu().detach().numpy()[0,0,:,:]
                        cv2.imwrite('{}{}/{}vis.png'.format(cfg.FOLDER, epoch, eval_step), 255*np.hstack((vis_image, target)))
                print('end of evaluation.')
                model.train()

            data = next(train_iter)

            loss_d, loss_g, fake_img, loss_dict = model(data)
            if step%100 == 0:
                vis_image = fake_img.cpu().detach().numpy()[0,0,:,:]
                target = data["targets"].cpu().detach().numpy()[0,0,:,:]
                cv2.imwrite('{}train_vis/{}train_vis.png'.format(cfg.FOLDER, epoch*len(train_iter)+step), 255*np.hstack((vis_image, target)))



            elif step % cfg.TRAIN.print_freq == 0 and step != 0:
                    
                print('epoch', epoch,
                        'iter', step, '/', len(train_iter),
                        'lr: %.5f'%(model.gen_optimizer.param_groups[0]['lr']),
                         end =" ")
                for key in loss_dict.keys():
                    print(key,  '%.5f'%(loss_dict[key]), end =" ")
                print("")


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
    # train model.
    train(exp_cfg)