# File: unet_resnetVAE.py
# Mô tả: Kiến trúc U-Net linh hoạt, hỗ trợ cả hai chế độ:
# 1. Sinh ảnh (deterministic=False): Sử dụng latent space ngẫu nhiên (mu, logvar).
# 2. Tái tạo ảnh (deterministic=True): Sử dụng latent space tất định (chỉ mu).
# Hỗ trợ `precomputed_z` để inference dễ dàng hơn.

import sys
sys.path.append('./code/torchcfm/models/unet/')

import math
from abc import abstractmethod
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nn import (
    checkpoint, conv_nd, linear, normalization,
    timestep_embedding, zero_module, avg_pool_nd
)

# --- Các lớp và hàm phụ trợ (Attention, ResBlock, etc.) ---
# --- Phần này giữ nguyên so với mã nguồn gốc của dự án ---

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels, self.out_channels, self.use_conv, self.dims = channels, out_channels or channels, use_conv, dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels, self.out_channels, self.use_conv, self.dims = channels, out_channels or channels, use_conv, dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels, self.emb_channels, self.dropout, self.out_channels = channels, emb_channels, dropout, out_channels or channels
        self.use_conv, self.use_checkpoint, self.use_scale_shift_norm = use_conv, use_checkpoint, use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up: self.h_upd, self.x_upd = Upsample(channels, False, dims), Upsample(channels, False, dims)
        elif down: self.h_upd, self.x_upd = Downsample(channels, False, dims), Downsample(channels, False, dims)
        else: self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels: self.skip_connection = nn.Identity()
        elif use_conv: self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else: self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb): return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    def _forward(self, x, emb):
        if self.updown:
            h = self.in_layers[:-1](x); h = self.h_upd(h); x = self.x_upd(x); h = self.in_layers[-1](h)
        else: h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1); h = self.out_layers[0](h) * (1 + scale) + shift; h = self.out_layers[1:](h)
        else: h = h + emb_out; h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels, self.use_checkpoint = channels, use_checkpoint
        self.num_heads = channels // num_head_channels if num_head_channels != -1 else num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x): return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    def _forward(self, x):
        b, c, *spatial = x.shape; x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)); h = self.attention(qkv); h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n): super().__init__(); self.n_heads = n
    def forward(self, qkv):
        bs, width, length = qkv.shape; ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch)); w = th.einsum("bct,bcs->bts", q * scale, k * scale)
        w = th.softmax(w.float(), dim=-1).type(w.dtype); a = th.einsum("bts,bcs->bct", w, v)
        return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):
    def __init__(self, n): super().__init__(); self.n_heads = n
    def forward(self, qkv):
        bs, width, length = qkv.shape; ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1); scale = 1 / math.sqrt(math.sqrt(ch))
        w = th.einsum("bct,bcs->bts", (q*scale).view(bs*self.n_heads,ch,length), (k*scale).view(bs*self.n_heads,ch,length))
        w = th.softmax(w.float(), dim=-1).type(w.dtype); a = th.einsum("bts,bcs->bct", w, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

# --- Lớp UNetModel chính (ĐÃ ĐƯỢC CẬP NHẬT) ---

class UNetModel(nn.Module):
    def __init__(
        self, image_size, in_channels, model_channels, out_channels,
        num_res_blocks, attention_resolutions, dropout=0,
        channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2,
        num_classes=None, num_latents=None, use_checkpoint=False,
        use_fp16=False, num_heads=1, num_head_channels=-1,
        num_heads_upsample=-1, use_scale_shift_norm=False,
        resblock_updown=False, use_new_attention_order=False,
        training=True, deterministic=False
    ):
        super().__init__()
        if num_heads_upsample == -1: num_heads_upsample = num_heads
        
        # Lưu các thuộc tính
        self.training, self.image_size, self.in_channels, self.model_channels, self.out_channels = training, image_size, in_channels, model_channels, out_channels
        self.num_res_blocks, self.attention_resolutions, self.dropout, self.channel_mult = num_res_blocks, attention_resolutions, dropout, channel_mult
        self.num_classes, self.num_latents, self.use_checkpoint = num_classes, num_latents, use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.deterministic = deterministic

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))

        # Latent conditioning layers
        if self.num_latents is not None:
            mu_dim = math.floor(time_embed_dim / 2)
            if self.deterministic:
                self.latent_encodings = nn.Linear(self.num_latents, mu_dim)
            else:
                self.latent_encodings = nn.Linear(self.num_latents, mu_dim * 2)
            self.latent_mlp = nn.Linear(mu_dim, time_embed_dim)

        # Kiến trúc U-Net (input, middle, output blocks)
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        input_block_chans = [ch]; ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult*model_channels), dims=dims, use_checkpoint=use_checkpoint)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions: layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers)); input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=ch)))
                ds *= 2; input_block_chans.append(ch)
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint)
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims, use_checkpoint=use_checkpoint)]
                ch = int(model_channels * mult)
                if ds in attention_resolutions: layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=ch)); ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)))

    def forward(self, t, x, y=None, precomputed_z=None):
        timesteps = t.repeat(x.shape[0]) if t.dim() == 0 else t
        hs = []; emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        mu, logvar = None, None # Khởi tạo
        if self.num_latents is not None:
            if precomputed_z is not None:
                z = precomputed_z
            else:
                latent_params = self.latent_encodings(y)
                if self.deterministic:
                    mu, logvar = latent_params, None
                    z = mu
                else:
                    mu, logvar = th.chunk(latent_params, 2, dim=1)
                    if self.training:
                        z = mu + th.randn_like(mu) * th.exp(0.5 * logvar)
                    else: # Dùng mu khi eval/inference để ổn định
                        z = mu
            
            proj = self.latent_mlp(z)
            emb = emb + proj

        h = x.type(self.dtype)
        for module in self.input_blocks: h = module(h, emb); hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks: h = th.cat([hs.pop(), h], dim=1); h = module(h, emb)
        h = h.type(x.dtype)
        
        return (self.out(h), mu, logvar) if self.num_latents is not None else self.out(h)

# --- Lớp Wrapper ---
class UNetModelWrapper(UNetModel):
    def __init__(
        self, dim, num_channels, num_res_blocks, channel_mult=None,
        attention_resolutions="16", deterministic=False, **kwargs
    ):
        image_size = dim[-1]
        if channel_mult is None:
            if image_size >= 256: channel_mult = (1, 1, 2, 2, 4, 4)
            else: channel_mult = (1, 2, 2, 2)
        
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
        super().__init__(
            image_size=image_size, in_channels=dim[0], model_channels=num_channels,
            out_channels=dim[0], num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds), channel_mult=channel_mult,
            deterministic=deterministic, **kwargs
        )
        
    def forward(self, t, x, y=None, *args, **kwargs):
        return super().forward(t, x, y=y, *args, **kwargs)