
def build_embedding(embedding_cfg, model_cfg):
    if model_cfg.type == "SimpleEmbedding":
        return resnet18(embedding_cfg, model_cfg)
    else:
        raise NotImplementedError