import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

def build_embedding(embedding_cfg, model_cfg):
    if embedding_cfg.type == "SimpleEmbedding":
        return SimpleEmbedding(embedding_cfg, model_cfg)
    elif embedding_cfg.type == "VGGEmbedding":
        return VGGEmbedding(embedding_cfg, model_cfg)
    else:
        raise NotImplementedError

class Zi2ZiEmbedding(nn.Module):
    def __init__(self, opts, cfg):
        super(Zi2ZiEmbedding, self).__init__()
        self.embed_init = True
    def update_initialized(self, data):
        pass
    def normalize_initial_feat(self):
        pass

class SimpleEmbedding(Zi2ZiEmbedding):
    def __init__(self, opts, cfg):
        super(SimpleEmbedding, self).__init__(opts, cfg)
        self.embedding_num = opts.embedding_num
        self.embedding_dim = opts.embedding_dim
        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim).cuda()
    
    def forward(self, idx):
        return self.embedding(idx).view(-1, self.embedding_dim, 1, 1)

class VGGEmbedding(Zi2ZiEmbedding):
    def __init__(self, opts, cfg):
        super(VGGEmbedding, self).__init__(opts, cfg)
        self.embed_init = False
        self.count = [0 for i in range(opts.embedding_num)]
        self.embedding_num = opts.embedding_num
        self.embedding_dim = opts.embedding_dim
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_deep = torch.nn.Sequential(*list(vgg16.children())[0][0:26]).eval()
        for param in self.vgg16_deep.parameters():
            param.requires_grad = False
        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim).cuda()
        self.embedding.weight = torch.nn.Parameter(
            torch.zeros((self.embedding_num, self.embedding_dim)).cuda()
            )
            
    def update_initialized(self, data):
        cat_id = data['cat_id'].cuda()
        input_img = data['imgs'].cuda()
        feat = self.vgg16_deep(input_img)
        b, c, h, w = feat.shape
        vgg_feat = F.adaptive_avg_pool2d(feat, (1)).view(b, c)
        for i in range(b):
            self.embedding.weight[int(cat_id[i])] += torch.nn.Parameter(vgg_feat[int(cat_id[i])])
            self.count[int(cat_id[i])] += 1

    def normalize_initial_feat(self):
        for i in range(self.embedding_num):
            self.embedding.weight[i]/=self.count[i]
        del self.vgg16_deep
        
    def forward(self, idx):
        return self.embedding(idx).view(-1, self.embedding_dim, 1, 1)

if __name__ == "__main__":
    import numpy as np
    embed = nn.Embedding(20, 512).cuda()
    embed.weight = torch.nn.Parameter(
            torch.zeros(20, 512).cuda()
            )
    rand_input = torch.FloatTensor(np.random.rand(16,3,256,256)).cuda()
    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    for param in vgg16.parameters():
        param.requires_grad = False
    vgg16_deep = torch.nn.Sequential(*list(vgg16.children())[0][0:26])
    feat = vgg16_deep(rand_input)
    vgg_feat = F.adaptive_avg_pool2d(feat, (1)).view(16, 512)
    for i in range(len(vgg_feat)):
        embed.weight[i] += torch.nn.Parameter(
            vgg_feat[i]
            )
    print(vgg_feat[0][:10])
    print(embed.weight[0][:10])
    # print(embed(torch.LongTensor([0,1,2]).cuda()))