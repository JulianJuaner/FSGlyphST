from sourcecode.model import build_model
from sourcecode.dataset import build_dataset
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate, compute_metric
from torch import optim
from torch.utils.data import DataLoader
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
    
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=cfg.TRAIN.optimizer.lr, weight_decay=0.0005)
    dis_optimizer = optim.Adam(model.discriminator.parameters(), lr=cfg.TRAIN.optimizer.lr, weight_decay=0.0005)
    loss_func = nn.CrossEntropyLoss().cuda()
    num_epoch = cfg.TRAIN.max_epoch

    for epoch in range(num_epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            #adjust_learning_rate(enc_optimizer, iteration, cfg, cfg.TRAIN.enc_lr_factor)
            #adjust_learning_rate(dec_optimizer, iteration, cfg, cfg.TRAIN.dec_lr_factor)

            data = next(train_iter)

            loss_d, loss_g, fake_img, loss_dict = model(data)

            model.generator.zero_grad()
            loss_g.backward(retain_graph=True)
            gen_optimizer.step()
            model.generator.zero_grad()
            model.discriminator.zero_grad()
            loss_d.backward()
            dis_optimizer.step()

            if epoch % cfg.TRAIN.eval_freq == 0 and epoch != 0 and step == 0:
                # evaluation.
                if epoch % cfg.TRAIN.ckpt_freq == 0:
                    print(epoch, 'saving model')
                    torch.save(model.state_dict(), '{}ckpt{}.pth'.format(
                            cfg.FOLDER,
                            epoch))

                model.eval()
                eval_iter = iter(eval_loader)
                for eval_step in range(len(eval_iter)):
                    eval_data = next(eval_iter)
                print('start evaluation.')


            elif step % cfg.TRAIN.print_freq == 0 and step != 0:
                    
                print('epoch', epoch,
                        'iter', step, '/', len(train_iter),
                        'DL: %.4f'%(loss_d.item()),
                        'GL: %.4f'%(loss_g.item()),
                        'const_l: %.4f'%loss_dict['const_loss'],
                        'discrim_l: %.4f'%loss_dict['discrim_loss'],
                        'real_cate: %.4f'%loss_dict['real_category_loss'],
                        'fake_cate: %.4f'%loss_dict['fake_category_loss'],
                        'l1_l: %.4f'%loss_dict['l1_loss'],
                        'cheat_l: %.4f'%loss_dict['cheat_loss'],
                        'lr: %.5f'%(gen_optimizer.param_groups[0]['lr']),
                         )

        


if "__main__" in __name__:
    # initialize exp configs.
    parser = argparse.ArgumentParser()
    OptionInit = Options(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    folder_name = opt.exp
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))
    train(exp_cfg)