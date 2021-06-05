import torch
import os

def adjust_learning_rate(optimizer, epoch, cfg):
    if epoch%cfg.TRAIN.lr_scheduler.decay_epoch == 0 and epoch!=0:
        for param_group in optimizer.param_groups:  
            param_group['lr'] = param_group['lr']*cfg.TRAIN.lr_scheduler.decay_rate
    

def compute_metric(pred, target, acc_sum=None, pixel_sum=None):
    ignore_index = pred.shape[1]
    _, preds = torch.max(pred, dim=1)
    if acc_sum != None and pixel_sum!= None:
        acc_sum = acc_sum
        pixel_sum = pixel_sum
    else:
        acc_sum = torch.zeros((pred.shape[1]+1)).cuda()
        pixel_sum = torch.zeros((pred.shape[1]+1)).cuda()

    for i in range(pred.shape[1]):
        valid = (target == i).long()
        acc_sum[i] += torch.sum(valid * (preds == i).long())
        pixel_sum[i] += torch.sum(valid)

    acc_sum[-1] = sum(acc_sum[:-1])
    pixel_sum[-1] = sum(pixel_sum[:-1])

    return acc_sum, pixel_sum

def gen_key_value_pair():
    folder = "./data/PNGfont_cn/fzbs"
    res = dict()
    imgs = os.listdir(folder)
    for i in range(len(imgs)):
        res[imgs[i][:5]] = i
    return res

if __name__ == "__main__":
    import random
    import cv2
    res = gen_key_value_pair()
    ft = open('data_train.txt', 'w')
    fv = open('data_val.txt', 'w')
    folder_list = os.listdir("./data/PNGfont_cn")
    i = 0
    for folder in folder_list[:30]:
        print(folder)
        data_list = os.listdir(os.path.join("./data/PNGfont_cn", folder))
        for data in data_list:
            type_id = res[data[:5]]
            font_id = i
            data_path = os.path.join("./data/PNGfont_cn", folder, data)
            # img = cv2.imread(data_path, 0)
            if random.random()<0.1:
                fv.write('{} {} {}\n'.format(type_id, font_id, data_path))
            else:
                ft.write('{} {} {}\n'.format(type_id, font_id, data_path))
        i+=1
    ft.close()
    fv.close()
