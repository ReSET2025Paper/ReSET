from __future__ import annotations

from math import pi
from typing import Literal, Union, Callable

import math
import torch
from einops import rearrange, repeat
from torch import nn, einsum, broadcast_tensors, Tensor
from torch.amp import autocast
from torch.nn import Module

def exists(val) -> bool:
    return val is not None


def default(val, d) -> Tensor:
    return val if exists(val) else d


def broadcat(tensors, dim=-1) -> Tensor:
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def rotate_half(x) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[
        -1], f"Feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat([t_left, t_transformed, t_right], dim=-1)
    return out.type(dtype)


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)

class RotaryEmbedding(Module):
    def __init__(
            self,
            dim,
            custom_freqs: Union[Tensor, None] = None,
            freqs_for: Literal["lang", "pixel", "constant"] = "lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
            interpolate_factor=1.0,
            theta_rescale_factor=1.0,
            seq_before_head_dim=False,
            cache_if_possible=True,
            cache_max_seq_len=8192
    ) -> None:
        super(RotaryEmbedding, self).__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "spacetime":
            time_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        if freqs_for == "spacetime":
            self.time_freqs = nn.Parameter(time_freqs, requires_grad=learned_freq)
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer("cached_freqs", torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.register_buffer("cached_freqs_seq_len", torch.tensor(0), persistent=False)

        self.learned_freq = learned_freq

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        self.use_xpos = use_xpos
        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = xpos_scale_base

            self.register_buffer("scale", scale, persistent=False)
            self.register_buffer("cached_scales", torch.zeros(cache_max_seq_len, dim), persistent=False)
            self.register_buffer("cached_scales_seq_len", torch.tensor(0), persistent=False)

            self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, freqs, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(
            scale), ("You must use '.rotate_queries_and_keys' method instead and pass in both queries and keys "
                     "for length extrapolatable rotary embeddings")

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")
        return apply_rotary_emb(seq_freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_and_keys(self, q, k, freqs, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(seq_freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(seq_freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: Union[int, None] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len) and (offset + seq_len) <= self.cache_max_seq_len

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales_seq_len.item():
            return self.cached_scales[offset: (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = repeat(scale, "n d -> n (d r)", r=2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)
        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            # Only allow pixel freqs for last two dimensions
            use_pixel = (self.freqs_for == "pixel" or self.freqs_for == "spacetime") and ind >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            if self.freqs_for == "spacetime" and not use_pixel:
                seq_freqs = self.forward(pos, self.time_freqs, seq_len=dim)
            else:
                seq_freqs = self.forward(pos, self.freqs, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(seq_freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, freqs: Tensor, seq_len=None, offset=0):
        should_cache = self.cache_if_possible and not self.learned_freq and exists(
            seq_len) and self.freqs_for != "pixel" and (offset + seq_len) <= self.cache_max_seq_len

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs_seq_len.item():
            return self.cached_freqs[offset: (offset + seq_len)].detach()
        else:
            freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)

            if should_cache and offset == 0:
                self.cached_freqs[:seq_len] = freqs.detach()
                self.cached_freqs_seq_len.copy_(seq_len)
            return freqs


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].cuda()


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)

    def scaled_dot_product_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    
class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x
    
class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = True
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        # added
        self.time_embed = nn.Parameter(torch.randn(1, 500, 1, model_dim))  # [1, T, 1, D] learnable time encodings
        # added 
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        # print('model dimension shape:', x.shape)
        #changed 
        x = self.pos_enc(x)
        x = x + self.time_embed[:, :x.shape[1]]
        # changed
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal)
        x = self.out(x)
        return x  # (B, T, E)
    
class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.out(x)
        return x  # (B, T, E)
    
def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, H, W, C = videos.shape
    videos = videos[:, :, :H - (H % size), :W - (W % size), :]
    x = rearrange(videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c", hp=size, wp=size, hn=hn)
    return x[:, :, :h_out, :w_out]


######################################## Diffusion policy #############################################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module
