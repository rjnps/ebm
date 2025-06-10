import torch
import torch.nn as nn


class FineTuneEncoderLanguage(nn.Module):
    def __init__(self, in_fts=728, out_fts=256):
        super().__init__()
        self.finetuner = nn.Sequential(
            nn.Linear(in_fts, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_fts)
        )

    def forward(self, x):
        return self.finetuner(x)