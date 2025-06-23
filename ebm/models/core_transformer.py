import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ebm.models.base_policy import BasePolicy
from einops import rearrange, repeat


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, head_out_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attn_scale = head_out_size ** (-0.5)

        self.qkv = nn.Linear(dim, num_heads * head_out_size * 3, bias=False)
        self.output_layer = nn.Sequential(nn.Linear(num_heads * head_out_size, dim),
                                          nn.Dropout(dropout))

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # each of q, k, v has this dim -> [batch, num_head, num_vectors_in_a_batch, vector_size]

        attn = (q @ k.transpose(-2, -1)) * self.attn_scale
        # attn size -> (B, H, N, N)

        # Causal mask implementation
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2: # mask along (B, N)
                # make the size of mask compatible with attention size
                # mask[:, None, None, :] -> (B, 1, 1, N)
                # padding mask -> for variable length sequences, not allowing
                # padded values to affect attention calculation
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N):
                # same causal mask for all the batches
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif len(mask.shape) == 3:
                # each batch with a different causal mask
                attn = attn.masked_fill(~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf"))
            else:
                raise Exception("mask shape is not correct for attention")

        attn = attn.softmax(dim=-1)
        self.attn_weights = attn
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        out = self.output_layer(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim_inp, dim_cond, num_heads, head_out_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attn_scale = head_out_size ** (-0.5)

        self.q_proj = nn.Linear(dim_inp, num_heads*head_out_size, bias=False)
        self.k_proj = nn.Linear(dim_cond, num_heads*head_out_size, bias=False)
        self.v_proj = nn.Linear(dim_cond, num_heads*head_out_size, bias=False)

        self.output_layer = nn.Sequential(nn.Linear(num_heads*head_out_size, dim_inp),
                                          nn.Dropout(dropout))

    def forward(self, x, c_,  mask=None):
        #assert len(c.shape) == 2, "Expected a 2D tensor for conditioning, got {} dim tensor".format(len(c.shape))
        c = c_[:, 0, :]
        B, N1, C_inp = x.shape
        B, C_cond = c.shape

        q = self.q_proj(x).reshape(B, N1, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(c).reshape(B, self.num_heads, -1).unsqueeze(1).permute(0, 2, 1, 3)
        v = self.v_proj(c).reshape(B, self.num_heads, -1).unsqueeze(1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.attn_scale

        # attn shape -> (B, H, N, 1)

        attn = attn.softmax(dim=-1)
        self.attn_weights = attn

        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        out = self.output_layer(out)
        return out

        # TODO: Masked Cross Attention


def drop_path(x,
              drop_prob=0.0,
              training=False,
              scale_by_keep=True):
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape -> (B, 1, 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # Random tensor, Bernaulli with probability keep_prob -> either 1 or 0 as output
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)  # Rescaling preserves the expected value
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class TransformerFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # GELU used in GPT-3 and BERT
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2)*2)

        inv_freq = 1.0 / (
                self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor


class CoreTransformer(nn.Module):
    def __init__(self,
                 input_size,
                 input_size_cond,
                 num_layers,
                 num_heads,
                 head_output_size,
                 mlp_hidden_size,
                 dropout,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        SelfAttention(input_size,
                                      num_heads=num_heads,
                                      head_out_size=head_output_size,
                                      dropout=dropout),
                        Norm(input_size),
                        CrossAttention(input_size,
                                       input_size_cond,
                                       num_heads=num_heads,
                                       head_out_size=head_output_size,
                                       dropout=dropout
                                       ),
                        Norm(input_size),
                        TransformerFeedForward(input_size,
                                               mlp_hidden_size,
                                               dropout),
                    ]
                )
            )
            self.attention_output[_] = None

        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)

        # Padding mask for varying sequence or vector length
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)

    def forward(self, x, c, mask):
        for layer_idx, (attn_norm,
                        self_attn,
                        s_attn_norm,
                        cross_attn,
                        c_attn_norm,
                        ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(self_attn(attn_norm(x), mask))
                x = x + drop_path(cross_attn(s_attn_norm(x), c, None))
            elif self.mask is not None:
                x = x + drop_path(self_attn(attn_norm(x), self.mask))
                x = x + drop_path(cross_attn(s_attn_norm(x), c, None))
            else:  # no masking
                x = x + drop_path(self_attn(attn_norm(x), None))
                x = x + drop_path(cross_attn(s_attn_norm(x), c, None))

            x = x + self.drop_path(ff(c_attn_norm(x)))

        return x

    @property
    def device(self):
        return next(self.parameters()).device


class LatentNet(nn.Module):
    def __init__(self,
                 num_layers=4,
                 input_size=512,
                 input_size_cond=512,
                 num_heads=8,
                 head_output_size=64,
                 mlp_hidden_size=728,
                 dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        SelfAttention(input_size,
                                      num_heads=num_heads,
                                      head_out_size=head_output_size,
                                      dropout=dropout),
                        Norm(input_size),
                        CrossAttention(input_size,
                                       input_size_cond,
                                       num_heads=num_heads,
                                       head_out_size=head_output_size,
                                       dropout=dropout
                                       ),
                        Norm(input_size),
                        TransformerFeedForward(input_size,
                                               mlp_hidden_size,
                                               dropout),
                    ]
                )
            )

        self.mlp_head = nn.Sequential(
            nn.Linear(input_size, 728),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(728, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, input_size)
        )

    def forward(self, x, c):
        for layer_idx, (attn_norm,
                        self_attn,
                        s_attn_norm,
                        cross_attn,
                        c_attn_norm,
                        ff) in enumerate(self.layers):

            x = x + drop_path(self_attn(attn_norm(x), None))
            x = x + drop_path(cross_attn(s_attn_norm(x), c, None))
            x = x + self.drop_path(ff(c_attn_norm(x)))
        out = self.mlp_head(x)
        return out


class AggregateFeatures(nn.Module):
    def __init__(self, policy_cfg):
        super().__init__()
        self.policy_cfg = policy_cfg
        if policy_cfg.agg_feat == "MaxPool":
            self.layer = nn.MaxPool1d(policy_cfg.horizon)
        elif policy_cfg.agg_feat == "AvgPool":
            self.layer = nn.AvgPool1d(kernel_size=policy_cfg.horizon)
        elif policy_cfg.agg_feat == "Linear":
            self.layer = nn.Linear(policy_cfg.horizon, 1)
        elif policy_cfg.agg_feat == "LastLayer":
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        # x -> [B, H, D]
        if self.policy_cfg.agg_feat == "LastLayer":
            return x[:, -1]
        else:
            x = x.transpose(1, 2)
            out = self.layer(x)
            out = out.squeeze(-1)
            return out


class MLPHead(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        drop_out = nn.Dropout(dropout) if self.training or dropout > 0.0 else nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_size, 128),
            drop_out,
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.mlp_head(x)


class LatentEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
