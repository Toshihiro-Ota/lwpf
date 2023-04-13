import torch
import torch.nn as nn
from .layers import trunc_normal_
from .vision_transformer import checkpoint_filter_fn
import lagrangian_units as lu

from .vision_transformer import Attention
class LagrangianAttention(Attention):
    def __init__(self, *args, lag_drop=None, **kwargs):
        super().__init__(*args, **kwargs)
        print(lag_drop)
        self.func_d = lu.get_lagunit(lag_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn, mask = self.func_d(attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn = mask * attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

from .vision_transformer import Block as _Block
class Block(_Block):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., lag_drop=None, **argv):
        super().__init__(dim, num_heads, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, **argv)
        self.attn = LagrangianAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, lag_drop=lag_drop)

from .vision_transformer import VisionTransformer as _VisionTransformer
class VisionTransformer(_VisionTransformer):
    def __init__(self, *args, block_fn=Block, drop_rate=0., **kwargs):
        if isinstance(drop_rate, str):
            drop_opts = drop_rate.split(',')
            drop_rate = float(drop_opts[0])
            lag_drop = drop_opts[1]
            block_fn = lambda *args, **kwargs: Block(*args, lag_drop=lag_drop, **kwargs)
        super().__init__(*args, block_fn=block_fn, drop_rate=drop_rate, **kwargs)
