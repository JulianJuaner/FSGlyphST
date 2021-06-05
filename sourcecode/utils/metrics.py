import torch
import torch.nn as nn
from piq import FID, ssim, SSIMLoss, LPIPS
import cv2
import numpy as np

class Metrics():
    """
        Bag of evaluation metrics.
    """
    def __init__(self):
        self.res = dict()
        self.count = 1e-8
        # self.Hausdorff = Hausdorff()
        self.PixAcc = PixAcc()
        self.LPIPS = _LPIPS()
        self.Chamfer = Chamfer()
        self.SSIM = SSIM()
        self.res["SSIM"] = 0
        self.res["LPIPS"] = 0
        self.res["Chamfer"] = 0
        # self.res["Hausdorff"] = 0
        self.res["PixAcc_hit"] = 0
        self.res["PixAcc_total"] = 1e-8

    def update_dict(self, pred, target):
        b,c,w,h = pred.shape
        self.count += b
        pred = pred.detach()
        target = target.detach()
        ssim_score = self.SSIM.forward(pred, target)
        self.res["SSIM"] += ssim_score
        lpips_score = self.LPIPS.forward(pred, target)
        self.res["LPIPS"] += lpips_score
        #chamfer_score = self.Chamfer.forward(pred, target)
        #self.res["Chamfer"] += chamfer_score
        #hausdorff_dist = self.Hausdorff.forward(pred, target)
        #self.res["Hausdorff"] += hausdorff_dist
        pix_batch_hit, pix_batch_total = self.PixAcc.forward(pred, target)
        self.res["PixAcc_hit"] += pix_batch_hit
        self.res["PixAcc_total"] += pix_batch_total
        return (ssim_score.item(), lpips_score.item(), pix_batch_hit.item(), pix_batch_total, pix_batch_hit.item()/pix_batch_total)

    def summary(self):
        self.res["SSIM"] /= self.count
        self.res["LPIPS"] /= self.count
        self.res["Chamfer"] /= self.count
        # self.res["Hausdorff"] /= self.count
        self.res["PixAcc"] = self.res["PixAcc_hit"]/self.res["PixAcc_total"]
        #del self.res["PixAcc_hit"]
        #del self.res["PixAcc_total"]
        return self.res

class IS():
    """
        Implementation of the Inception Score.
    """
    pass

class Hausdorff():
    """
        Implementation of the Hausdorff Distance.
    """
    pass
    

class _FID():
    """
        Implementation of the Frechet Inception Distance.
    """
    def __init__(self):
        self.fid_metric = FID()
    def forward(self, p1, p2):
        first_feats = self.fid_metric.compute_feats(p1)
        second_feats = self.fid_metric.compute_feats(p2)
        fid = self.fid_metric(first_feats, second_feats)
        return fid

class PixAcc():
    """
        Implementation of the Pixel Accuracy.
    """
    def __init__(self):
        super(PixAcc, self).__init__()
        self.threshold = 0.5
    def forward(self, p1, p2):
        preds, _ = torch.max((p1>=self.threshold), dim=1)
        #print(preds.shape, p1.shape, p1)
        gt, _ = torch.max((p2>=self.threshold), dim=1)
        # valid = (preds >= self.threshold).long()
        acc_sum = torch.sum((preds == gt).long())
        pixel_sum = 256*256
        return acc_sum.float(), pixel_sum

class _LPIPS():
    """
        Implementation of the Learned Perceptual Image Patch Similarity (AlexNet).
    """
    def __init__(self):
        self.loss_fn_alex = LPIPS()
    def forward(self, p1, p2):
        loss = self.loss_fn_alex(p1, p2)
        return loss

class Chamfer():
    """
        Implementation of the Chamfer Distance.
    """
    def __init__(self):
        from chamfer_distance import ChamferDistance
        self.chamfer_dist = ChamferDistance().cuda()
    def forward(self, p1, p2):
        dist1, dist2, idx1, idx2 = self.chamfer_dist(p1[0],p2[0])
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss

class SSIM():
    """
        Implementation of the Structural Similarity Index.
    """
    def __init__(self):
        self.loss = SSIMLoss(data_range=1.)
    def forward(self, pred, target):
        ssim_index = ssim(pred, target, data_range=1.)
        # ssim_loss = self.loss(pred, target)
        return ssim_index

# Multi-Scale Gradient Magnitude Similarity.
class MSGMS(torch.nn.Module):
    def __init__(self):
        super(MSGMS, self).__init__()
        
        self.weight_h = torch.FloatTensor([[[[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]]],
                                    [[[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]]],
                                    [[[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]]]])
        self.weight_v = torch.FloatTensor([[[[1/3, 1/3, 1/3], [0,0,0], [-1/3, -1/3, -1/3]]],
                                    [[[1/3, 1/3, 1/3], [0,0,0], [-1/3, -1/3, -1/3]]],
                                    [[[1/3, 1/3, 1/3], [0,0,0], [-1/3, -1/3, -1/3]]]])

        self.conv_h = nn.Conv2d(3,3,3,1,1,groups=3, bias=False)
        self.conv_v = nn.Conv2d(3,3,3,1,1,groups=3, bias=False)

        self.conv_h.weight = nn.Parameter(self.weight_h)
        self.conv_v.weight = nn.Parameter(self.weight_v)

        self.c = 0.0026

        for params in self.parameters():
            params.requires_grad = False
        

    def forward(self, real, distorted):
        vis_result = []
        loss_result = []
        for i in range(len(real)):
            m_r = torch.pow(torch.pow(self.conv_h(real[i]), 2) + torch.pow(self.conv_v(real[i]), 2), 0.5)
            m_d = torch.pow(torch.pow(self.conv_h(distorted[i]), 2) + torch.pow(self.conv_v(distorted[i]), 2), 0.5)
            loss = 1 - torch.div(2*m_r*m_d + self.c, m_r*m_r + m_d*m_d + self.c)
            if i == len(real) - 1:
                vis_result = loss
            loss_result.append(loss.mean())
            
        return vis_result, loss_result

if __name__ == '__main__':
    real = cv2.imread('sth', 1)/255
    fake = cv2.imread('sth', 1)/255
    image_real = torch.FloatTensor(real).permute(2,0,1).unsqueeze(0)
    image_fake = torch.FloatTensor(fake).permute(2,0,1).unsqueeze(0)
    MSGMS_loss = MSGMS()
    vis, loss = MSGMS_loss([image_real], [image_fake])
    image_loss = np.array(vis[0].permute(1,2,0))
    cv2.imwrite('res.png',image_loss*255)
    