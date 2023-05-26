from math import pi
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from torch import Tensor, einsum
from torch.nn import functional as F
from .utils import exists, default, prob_mask_like

"""
Norms
"""

class LayerNorm(nn.Module):
    def __init__(self, features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm

"""
Attention Helper Blocks
"""

class InsertNullTokens(nn.Module):
    def __init__(self, head_features: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.tokens = nn.Parameter(torch.randn(2, head_features))

    def forward(
        self, k: Tensor, v: Tensor, *, mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        b = k.shape[0]
        nk, nv = repeat_many(
            self.tokens.unbind(dim=-2), "d -> b h 1 d", h=self.num_heads, b=b
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        mask = F.pad(mask, pad=(1, 0), value=True) if exists(mask) else None
        return k, v, mask


def FeedForward1d(channels: int, multiplier: int = 2):
    mid_channels = int(channels * multiplier)
    return nn.Sequential(
        LayerNorm1d(channels=channels, bias=False),
        Conv1d(
            in_channels=channels, out_channels=mid_channels, kernel_size=1, bias=False
        ),
        nn.GELU(),
        LayerNorm1d(channels=mid_channels, bias=False),
        Conv1d(
            in_channels=mid_channels, out_channels=channels, kernel_size=1, bias=False
        ),
    )

def attention_mask(
    sim: Tensor,
    mask: Tensor,
) -> Tensor:
    mask = rearrange(mask, "b j -> b 1 1 j")
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        use_null_tokens: bool = True,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_null_tokens = use_null_tokens
        mid_features = head_features * num_heads
        out_features = out_features if exists(out_features) else features

        self.insert_null_tokens = InsertNullTokens(
            head_features=head_features, num_heads=num_heads
        )
        self.to_out = nn.Sequential(
            nn.Linear(in_features=mid_features, out_features=out_features, bias=False),
            LayerNorm(features=out_features, bias=False),
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:

        # Split heads, scale queries
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        q = q * self.scale

        # Insert null tokens
        if self.use_null_tokens:
            k, v, mask = self.insert_null_tokens(k, v, mask=mask)

        # Compute similarity matrix with bias and mask
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = sim + attention_bias if exists(attention_bias) else sim
        sim = attention_mask(sim, mask) if exists(mask) else sim

        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("... n j, ... j d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int = 64,
        num_heads: int = 8,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        mid_features = head_features * num_heads

        self.norm = LayerNorm(features, bias=False)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_null_tokens=False,
            out_features=out_features,
        )

    def forward(self, x: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(x), chunks=2, dim=-1))
        x = self.attention(q, k, v) if mask is None else self.attention(q, k, v, mask=mask.squeeze(1))
        return x

"""
Transformer Blocks
"""

class TransformerBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 8,
        head_features: int = 32,
        multiplier: int = 2,
    ):
        super().__init__()
        self.attention = EinopsToAndFrom(
            "b c l",
            "b l c",
            Attention(
                features=channels, num_heads=num_heads, head_features=head_features
            ),
        )
        self.feed_forward = FeedForward1d(channels=channels, multiplier=multiplier)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x if x_mask is None else self.attention(x, mask=x_mask.squeeze(1)) + x
        x = self.feed_forward(x) + x
        return x

"""
Time Embeddings
"""

class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

"""
Convolutional Helper Blocks
"""

def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)

def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)

def scale_and_shift(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (scale + 1) + shift

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        )

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None,
        inj_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        
        x = self.groupnorm(x)
        if exists(scale_shift):
            x = scale_and_shift(x, scale=scale_shift[0], shift=scale_shift[1])
        
        if exists(inj_embeddings):
            x = inj_embeddings * x
        x = self.activation(x)
        return self.project(x)


"""
UNet Helper Functions and Blocks
"""

def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor*kernel_multiplier + 1,
        stride=factor,
        padding=factor*(kernel_multiplier//2),
    )

def Upsample1d(
    in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest"),
            nn.ReflectionPad1d(1),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
        )

class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dilation: int = 1,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None,
    ) -> None:
        super().__init__()

        self.to_cond_embedding = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=int(time_embed_dim or 0)+int(classes_embed_dim or 0),
                          out_features=out_channels*2
                ),
            ) if exists(time_embed_dim) or exists(classes_embed_dim) else None
        )

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            dilation=dilation,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels, out_channels=out_channels, num_groups=num_groups
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embed: Optional[Tensor] = None, 
                class_embed: Optional[Tensor] = None, 
                inj_embeddings: Optional[Tensor] = None) -> Tensor:

        # Compute scale and shift from conditional embedding (time_embed + class_embed)
        scale_shift = None
        if exists(self.to_cond_embedding) and (exists(time_embed) or exists(class_embed)):
            cond_emb = tuple(filter(exists, (time_embed, class_embed)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.to_cond_embedding(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x)
        h = self.block2(h, scale_shift=scale_shift, 
                        inj_embeddings=inj_embeddings)

        return h + self.to_out(x)

"""
UNet Blocks
"""

class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None,
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads) and exists(attention_features)
        )

        self.use_attention = use_attention

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_embed_dim
        )

        if use_attention:
            assert exists(attention_heads) and exists(attention_features)
            self.attention = EinopsToAndFrom(
                "b c l",
                "b l c",
                Attention(
                    features=channels,
                    num_heads=attention_heads,
                    head_features=attention_features,
                ),
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_embed_dim
        )

    def forward(self, x: Tensor, t: Optional[Tensor] = None, 
                c: Optional[Tensor] = None, 
                inj_embeddings: Optional[Tensor] = None) -> Tensor:
        x = self.pre_block(x, t, c, inj_embeddings)
        if self.use_attention:
            x = self.attention(x)
        x = self.post_block(x, t, c, inj_embeddings)
        return x

class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        extract_channels: int = 0,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_embed_dim: Optional[int]  = None, 
        classes_embed_dim: Optional[int] = None,
        use_injected_channels: Optional[bool] = False,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_attention = use_attention
        self.use_extract = extract_channels > 0

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels * 2 if use_injected_channels and i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_embed_dim
                )
                for i in range(num_layers)
            ]
        )

        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
            )

    def forward(
        self, x: Tensor, 
        t: Optional[Tensor] = None, 
        c: Optional[Tensor] = None, 
        inj_embeddings: Optional[Tensor] = None,
        inj_channels: Optional[Tensor] = None) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        if inj_channels is not None and inj_channels.shape[-1] == x.shape[-1]:
            x = torch.cat([x, inj_channels], dim=1)

        skips = []
        for block in self.blocks:
            x = block(x, t, c, inj_embeddings)
            skips += [x] if self.use_skip else []

        if self.use_attention:
            x = self.transformer(x)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return (x, skips) if self.use_skip else x

class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads)
            and exists(attention_features)
            and exists(attention_multiplier)
        )

        self.use_pre_upsample = use_pre_upsample
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_embed_dim
                )
                for _ in range(num_layers)
            ]
        )

        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        skips: Optional[List[Tensor]] = None,
        t: Optional[Tensor] = None,
        c: Optional[Tensor] = None, 
        inj_embeddings: Optional[Tensor] = None
    ) -> Tensor:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, t, c, inj_embeddings)

        if self.use_attention:
            x = self.transformer(x)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        return x
    
"""
UNet and EncDec
"""
    
class WAVenc1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        # efficient input transform:
        # 'LT'
        num_filters: int, # non-zero patchsize to activate learnable filterbank
        window_length: int, 
        stride: int,
    ):
        super().__init__()
        
        # input transform to improve efficiency
        ## conv enc-dec LTP in https://github.com/archinetai/audio-diffusion-pytorch
        padding = window_length // 2 - stride // 2
        self.to_in = Conv1d(in_channels=in_channels,
                            out_channels=in_channels*num_filters,
                            kernel_size=window_length,
                            stride=stride,
                            padding=padding,
                            bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.to_in(x)
        
class WAVdec1d(nn.Module):
    # efficient input transform for UNet:
    def __init__(
        self,
        in_channels: int,
        # 'LT'
        num_filters: int, # non-zero patchsize to activate learnable filterbank
        window_length: int, 
        stride: int,
        # output channels
        out_channels: Optional[int] = None
    ):

        super().__init__()
        
        out_channels = default(out_channels, in_channels)
        
        # input transform to improve efficiency
        ## conv enc-dec LTP in https://github.com/archinetai/audio-diffusion-pytorch
        padding = window_length // 2 - stride // 2
        self.to_out = nn.ConvTranspose1d(in_channels=in_channels*num_filters,
                                         out_channels=out_channels, 
                                         kernel_size=window_length,
                                         stride=stride, 
                                         padding=padding, 
                                         bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.to_out(x)
    
class UNet1d(nn.Module):
    def __init__(
        self,
        # efficient input transform:
        num_filters: int,
        window_length: int, 
        stride: int,
        # unet
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_features: int,
        attention_multiplier: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        use_attention_bottleneck: bool,
        out_channels: Optional[int] = None,
        injected_depth: Optional[int] = None,
        classes_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # outmost (input and output) blocks 
        self.to_in = WAVenc1d(in_channels=in_channels, 
                              num_filters=num_filters, 
                              window_length=window_length, 
                              stride=stride)

        self.to_out = WAVdec1d(in_channels=in_channels, 
                               num_filters=num_filters, 
                               window_length=window_length, 
                               stride=stride,
                               out_channels=out_channels)
        
        # dimension assertion
        time_embed_dim = channels * 4
        
        num_layers = len(multipliers) - 1
        self.num_layers = num_layers
        
        assert (len(factors) == num_layers
                and len(attentions) == num_layers
                and len(num_blocks) == num_layers
               )

        # time embedding
        self.to_time = nn.Sequential(
            TimePositionalEmbedding(dim=channels, out_features=time_embed_dim),
            nn.SiLU(),
            nn.Linear(
                in_features=time_embed_dim, out_features=time_embed_dim
            ),
        )

        # unet
        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i], #* 2 if injected_depth and i == injected_depth else channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],# * 2 if injected_depth and i == injected_depth else channels * multipliers[i],
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_dim,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_features=attention_features,
                    attention_multiplier=attention_multiplier,
                    use_injected_channels = True if injected_depth and i == injected_depth else False,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            num_groups=resnet_groups,
            use_attention=use_attention_bottleneck,
            attention_heads=attention_heads,
            attention_features=attention_features,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_dim
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_dim,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    skip_channels=channels * multipliers[i + 1],
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_features=attention_features,
                    attention_multiplier=attention_multiplier,
                )
                for i in reversed(range(num_layers))
            ]
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor, 
                inj_embeddings:Optional[Tensor] = None, 
                inj_channels:Optional[Tensor] = None, 
                **kwargs):

        # input transform
        x = self.to_in(x)

        # unet
        # time embedding
        t = self.to_time(t)
        skips_list = []
        for i, downsample in enumerate(self.downsamples):
            x, skips = downsample(x, t, c, inj_embeddings, inj_channels)
            skips_list += [skips]

        x = self.bottleneck(x, t, c, inj_embeddings)
        
        for _, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips, t, c, inj_embeddings)
            
        # output transform
        x = self.to_out(x) 
        return x

class UNet1dBase(nn.Module):
    # unconditonal unet1d based diffusion
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # input transform to improve efficiency
        self.unet = UNet1d(**kwargs)

    def forward(self, x: Tensor, t: Tensor, 
                inj_embeddings:Optional[Tensor] = None, 
                inj_channels:Optional[Tensor] = None, 
                **kwargs):

        c = None
        x = self.unet(x, t, c, inj_embeddings, inj_channels)
        return x
    
class UNet1dCFG(nn.Module):
    # unet1d based diffusion with CFG
    
    def __init__(self, channels: int, 
                 num_classes: int, 
                 cond_drop_prob: float, 
                 # conditional dropout of the time, must be greater than 0. to unlock classifier free guidance
                 **kwargs):
        super().__init__()

        # class embeddings for cfg
        self.cond_drop_prob = cond_drop_prob  # classifier free guidance dropout
        self.classes_emb = nn.Embedding(num_classes, channels)
        self.null_classes_emb = nn.Parameter(torch.randn(channels))

        classes_dim = channels * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(channels, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim))
        
        self.unet = UNet1d(channels=channels, 
                           classes_dim=channels*4,
                           **kwargs)

    def forward(self, x: Tensor, t: Tensor, 
                classes: Tensor, 
                cond_drop_prob=None, 
                inj_embeddings:Optional[Tensor] = None,
                **kwargs):

        batch_size, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        
        classes_emb = self.classes_emb(classes)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch_size)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
        c = self.classes_mlp(classes_emb)

        # unet
        x = self.unet(x, t, c, inj_embeddings)

        return x

# example

if __name__ == '__main__':
    num_classes = 10

    model = UNet1dCFG(
        num_filters=8,
        num_classes = 10,
        cond_drop_prob = 0.1,
        window_length=3, 
        stride=1,
        in_channels = 16,
        channels = 128,
        resnet_groups = 8,
        kernel_multiplier_downsample = 2,
        multipliers = [1, 2],
        factors = [4],
        num_blocks = [2],
        attentions = [False],
#         multipliers = [1, 2, 4, 4, 4, 4, 4],
#         factors = [4, 4, 4, 2, 2, 2],
#         num_blocks = [2, 2, 2, 2, 2, 2],
#         attentions = [False, False, False, True, True, True],
        attention_heads = 8,
        attention_features = 64,
        attention_multiplier = 2,
        use_nearest_upsample = False,
        use_skip_scale = True,
        use_attention_bottleneck = True)
    
#     torch.save(model.state_dict(), '/home/ge/modern.pt')
#     exit()

    training_images = torch.randn(8, 16, 512) # images are normalized from 0 to 1
    image_classes = torch.randint(0, num_classes, (8,))    # say 10 classes

    normal = -3.0  + 1.0 * torch.randn((8,), device=image_classes.device)
    sigmas = normal.exp()
    c_noise = torch.log(sigmas) * 0.25

    x_hat = model(training_images, c_noise, image_classes)
    print(x_hat.shape)
#     torchaudio.save('./demo.wav', x_hat[0], 44100)