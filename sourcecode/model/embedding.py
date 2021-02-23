import torch.nn as nn

def build_embedding(embedding_cfg, model_cfg):
    if embedding_cfg.type == "SimpleEmbedding":
        return SimpleEmbedding(embedding_cfg, model_cfg)
    else:
        raise NotImplementedError
    
class SimpleEmbedding(nn.Module):
    def __init__(self, opts, cfg):
        super(SimpleEmbedding, self).__init__()
        self.embedding_num = opts.embedding_num
        self.embedding_dim = opts.embedding_dim
        self.embedding = nn.Embedding(self.embedding_num, self.embedding_dim)
        
    def forward(self, idx):
        return self.embedding(idx).view(-1, self.embedding_dim, 1, 1)