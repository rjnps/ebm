import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from r3m import load_r3m
import torch.nn as nn


class R3MEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.r3m_model = load_r3m("resnet50")  # resnet18, resnet34
        self.r3m_model.eval()
        self.r3m_model.to(device)

    def forward(self, image):
        # input shape -> [B*H, c, h, w]
        with torch.no_grad():
            embedding = self.r3m_model(image*255.0)
        return embedding


class ResNetEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        raise NotImplementedError
