# model architecture, extracted from the kaggle notebook cell 4
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models


class GEM(nn.Module):
    # generalized mean pooling (inspiration from the winner's solution)
    def __init__(self, p=3, eps=1e-6):
        super(GEM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x ** self.p
        x = f.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x ** (1.0 / self.p)
        return x


def build_model(device):
    # building efficientnet-b3 on GEM pooling
    model = models.efficientnet_b3(weights='DEFAULT')
    model.avgpool = GEM(p=3)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(device)
    return model
