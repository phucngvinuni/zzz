# File: unet_latent.py
# Mô tả: U-Net được thiết kế để hoạt động trong Latent Space.
#        Kiến trúc này không có logic điều kiện latent, nó chỉ học
#        một ánh xạ từ latent nhiễu đến latent sạch.

import sys
sys.path.append('./code/torchcfm/models/unet/')

import math
from abc import abstractmethod
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# --- Import đầy đủ các thành phần cần thiết từ nn.py ---
from nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
    zero_module,
)

# --- Các lớp và hàm phụ trợ (Attention, ResBlock, etc.) ---
# --- Phần này giữ nguyên so với các file U-Net khác trong dự án ---

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
        self.channels, self.out_channels, self.use_conv = channels, out_channels or channels, use_conv
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv: x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels, self.out_channels, self.use_conv = channels, out_channels or channels, use_conv
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
    def forward(self, x):
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, dims=2, use_checkpoint=False):
        super().__init__()
        self.channels, self.emb_channels, self.dropout, self.out_channels = channels, emb_channels, dropout, out_channels or channels
        self.use_conv, self.use_checkpoint = use_conv, use_checkpoint
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels: self.skip_connection = nn.Identity()
        elif use_conv: self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else: self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb): return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels, self.use_checkpoint = channels, use_checkpoint
        self.num_heads = channels // num_head_channels if num_head_channels != -1 else num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x): return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    def _forward(self, x):
        b, c, *spatial = x.shape; x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)); h = self.attention(qkv); h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    def __init__(self, n): super().__init__(); self.n_heads = n
    def forward(self, qkv):
        bs, width, length = qkv.shape; ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1); scale = 1 / math.sqrt(math.sqrt(ch))
        w = th.einsum("bct,bcs->bts", (q*scale).view(bs*self.n_heads,ch,length), (k*scale).view(bs*self.n_heads,ch,length))
        w = th.softmax(w.float(), dim=-1).type(w.dtype); a = th.einsum("bts,bcs->bct", w, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

# --- Lớp UNetModel chính ---
class UNetModel(nn.Module):
    def __init__(
        self, image_size, in_channels, model_channels, out_channels,
        num_res_blocks, attention_resolutions, dropout=0,
        channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2,
        num_heads=1, num_head_channels=-1, use_checkpoint=False,
        num_heads_upsample=-1,
    ):
        super().__init__()
        if num_heads_upsample == -1: num_heads_upsample = num_heads
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), dims=dims, use_checkpoint=use_checkpoint)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2
        
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
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1))
        )

    def forward(self, t, x, y=None):
        timesteps = t.repeat(x.shape[0]) if t.dim() == 0 and x.dim() > 1 else t
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([hs.pop(), h], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        
        return self.out(h)

# --- Lớp Wrapper ---
class UNetModelWrapper(UNetModel):
    def __init__(self, dim, num_channels, num_res_blocks, channel_mult=None, attention_resolutions="16", **kwargs):
        image_size = dim[-1]
        if channel_mult is None:
            if image_size >= 32: channel_mult = (1, 2, 4, 8)
            else: channel_mult = (1, 2, 2, 2)
        
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
        super().__init__(
            image_size=image_size,
            in_channels=dim[0],
            model_channels=num_channels,
            out_channels=dim[0],
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            **kwargs
        )