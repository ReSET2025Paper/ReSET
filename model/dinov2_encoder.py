import torch
from utils import freeze_model

def dinov2_vitb14_reg(pretrained=True):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    if pretrained:
        freeze_model(model)
        model.eval()
    return model