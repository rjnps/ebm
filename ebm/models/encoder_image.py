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

    def forward(self, images):
        # input shape -> [B*H, c, h, w]
        # preprocess the input images
        images = T.Resize(224)(images)
        with torch.no_grad():
            # R3M expects inputs in the range of 0 - 255
            embedding = self.r3m_model(images*255.0) # [B*H, 2048]
        return embedding


class ResNetEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        raise NotImplementedError


class FineTuneEncoderImage(nn.Module):
    def __init__(self, in_fts=2048, out_fts=512):
        super().__init__()
        self.finetuner = nn.Sequential(
            nn.Linear(in_fts, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_fts)
        )

    def forward(self, x):
        return self.finetuner(x)
