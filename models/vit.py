import math
from functools import partial, reduce
from itertools import chain
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, to_2tuple
from timm.models import VisionTransformer
from torch.utils.checkpoint import checkpoint


__all__ = [
    'vit_small',
    'vit_base',
]


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if not return_attention:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            return attn  # return attention

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attention=False):
        y = self.attn(self.norm1(x), return_attention=return_attention)
        if return_attention:
            return y  # y is the attention map
        x = x + self.drop_path1(self.ls1(y))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def build_2d_sincos_position_embedding(grid_size, embed_dim, num_prefix_tokens, temperature=10000.):
    h, w = grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    if num_prefix_tokens == 0:
        pos_embed = nn.Parameter(pos_emb)
    else:
        # add cls tokens to positional embedding
        pe_token = torch.zeros([1, num_prefix_tokens, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))

    pos_embed.requires_grad = False
    return pos_embed


class ViT(VisionTransformer):

    def __init__(self, return_patches=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.checkpointing_every = 1
        self.depth = kwargs['depth']
        self.patch_size = kwargs['patch_size']
        self.return_patches = return_patches

        # Use fixed 2D sin-cos position embedding
        self.pos_embed = build_2d_sincos_position_embedding(grid_size=self.patch_embed.grid_size,
                                                            embed_dim=self.embed_dim,
                                                            num_prefix_tokens=0 if self.no_embed_class
                                                            else self.num_prefix_tokens)
        # Freeze patch projection
        self.patch_embed.proj.weight.requires_grad = False
        self.patch_embed.proj.bias.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True, every: int = 1):
        self.grad_checkpointing = enable
        self.checkpointing_every = every

    def forward_features(self, x, mix_ratio: float = 0., is_teacher: bool = False):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, every=self.checkpointing_every)
        else:
            x = self.blocks(x)

        x = self.norm(x)

        if self.return_patches and self.training:
            return x

        if self.global_pool == 'avg':
            return x[:, self.num_prefix_tokens:].mean(dim=1)
        else:
            return x[:, 0]

    def forward(self, x, **kwargs):
        x = self.forward_features(x, **kwargs)
        return x

    def get_selfattention(self, x, block_idx: int = -1):
        assert self.global_pool == 'token'

        if block_idx < 0:
            block_idx = self.depth + block_idx

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < block_idx:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True,
        use_reentrant=False,
):
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x,
                       use_reentrant=use_reentrant,
                       preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def vit_small(patch_size=16, qkv_bias=True, use_fc_norm=False, **kwargs):
    model = ViT(
        img_size=224, patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6), fc_norm=use_fc_norm,
        block_fn=Block, **kwargs
    )
    return model


def vit_base(patch_size=16, qkv_bias=True, use_fc_norm=False, **kwargs):
    model = ViT(
        img_size=224, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6), fc_norm=use_fc_norm,
        block_fn=Block, **kwargs
    )
    return model


def vit_small_LS(patch_size=16, **kwargs):
    model = ViT(
        img_size=224, patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=1., block_fn=Block, **kwargs
    )
    return model


def vit_base_LS(patch_size=16, **kwargs):
    model = ViT(
        img_size=224, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=1., block_fn=Block, **kwargs
    )
    return model
