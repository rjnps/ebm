import torch
import torch.nn as nn


class DMPHead(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        drop_out = nn.Dropout(dropout) if self.training or dropout > 0.0 else nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_size, 512),
            drop_out,
            nn.GELU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        return self.mlp_head(x)