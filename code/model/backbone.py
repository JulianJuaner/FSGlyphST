
def build_backbone(backbone_cfg, full_cfg):
    if model_cfg.type == "resnet18":
        return resnet18(backbone_cfg, full_cfg)
    else:
        raise NotImplementedError