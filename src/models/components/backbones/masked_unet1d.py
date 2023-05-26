from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops_exts.torch import EinopsToAndFrom
from torch import Tensor
from .utils import prob_mask_like, exists, default, sequence_mask
from .unet1d import TransformerBlock1d, Attention, TimePositionalEmbedding, Conv1d, scale_and_shift, Upsample1d, Downsample1d

def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
     
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask

class MaskedGroupNorm(nn.GroupNorm):
    """
    Masked verstion of the Group normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's GroupNorm implementation for argument details.
    """
    def __init__(self, num_groups, num_features, eps=1e-5,affine=True):
        super(MaskedGroupNorm, self).__init__(
            num_groups,
            num_features,
            eps,
            affine
        )

    def forward(self, inp, lengths):
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        
        assert inp.shape[1]%self.num_groups == 0, 'Feature size not divisible by groups'

        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        ave_mask = mask / lengths[:,None] / (inp.shape[-2] / self.num_groups) #also features
        ave_mask = ave_mask.unsqueeze(1)#.expand(inp.shape)

        # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
        # variance, we do not need to make any tensor shape manipulation.
        # mean = E[X] is simply the sum-product of our "probability" mask with the input...
        inp = inp*mask.unsqueeze(1) #mask out any extra bits of data - such as those left from conv bleeding
        inp_r = inp.reshape([inp.shape[0],self.num_groups,-1,inp.shape[-1]])
        ave_mask = ave_mask.unsqueeze(2)
        mean = (ave_mask * inp_r).sum([2, 3])
        # ...whereas Var(X) is directly derived from the above formulae
        # This should be numerically equivalent to the biased sample variance
        var = (ave_mask * inp_r ** 2).sum([2, 3]) - mean ** 2

        inp_r = (inp_r - mean[:,:,None,None]) / (torch.sqrt(var[:, :, None, None] + self.eps))
        out = inp_r.reshape(inp.shape)
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * mask.unsqueeze(1)

"""
Convolutional Helper Blocks
"""

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

        self.groupnorm = (MaskedGroupNorm(num_groups=num_groups, num_features=in_channels)
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
        self, x: Tensor, x_mask: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:

        x = x * x_mask
        mask_lengths = torch.sum(x_mask, dim=-1).squeeze(1)
        x = self.groupnorm(x, mask_lengths)
        if exists(scale_shift):
            x = scale_and_shift(x, scale=scale_shift[0], shift=scale_shift[1])
        x = self.activation(x)
        return self.project(x * x_mask) * x_mask

"""
UNet Helper Functions and Blocks
"""

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
                nn.Linear(in_features=int(time_embed_dim)+int(classes_embed_dim),
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

    def forward(self, x: Tensor, x_mask: Tensor, time_embed: Optional[Tensor] = None, 
                class_embed: Optional[Tensor] = None) -> Tensor:

        # Compute scale and shift from conditional embedding (time_embed + class_embed)
        scale_shift = None
        if exists(self.to_cond_embedding) and (exists(time_embed) or exists(class_embed)):
            cond_emb = tuple(filter(exists, (time_embed, class_embed)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.to_cond_embedding(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, x_mask)
        h = self.block2(h, x_mask, scale_shift=scale_shift)
        return h + self.to_out(x*x_mask)

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

    def forward(self, x: Tensor, x_mask: Tensor, t: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Tensor:
        x = self.pre_block(x, x_mask, t, c)
        if self.use_attention:
            x = self.attention(x, mask=x_mask.squeeze(1))
        x = self.post_block(x, x_mask, t, c)
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
        time_embed_dim: int = None, 
        classes_embed_dim: int = None
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_attention = use_attention
        self.use_extract = extract_channels > 0
        self.factor = factor

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
                    in_channels=channels,
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
        self, x: Tensor, x_mask: Tensor, t: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x*x_mask)
            x_mask = x_mask[:, :, ::self.factor]
        skips = []
        
        for block in self.blocks:
            x = block(x, x_mask, t, c)
            skips += [x] if self.use_skip else []
        
        if self.use_attention:
            x = self.transformer(x, x_mask)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x, x_mask)
            x_mask = x_mask[:, :, ::self.factor]

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted
        
        return (x*x_mask, skips, x_mask) if self.use_skip else (x, x_mask)

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
        x_mask: Tensor,
        skips: Optional[List[Tensor]] = None,
        t: Optional[Tensor] = None,
        c: Optional[Tensor] = None
    ) -> Tensor:

        if self.use_pre_upsample:
            # not using it
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, x_mask, t, c)

        if self.use_attention:
            x = self.transformer(x, x_mask)

        if not self.use_pre_upsample:
            x = self.upsample(x * x_mask)

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
    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        return self.to_in(x*x_mask)
        
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

    def forward(self, x: Tensor, x_mask:Tensor) -> Tensor:
        return self.to_out(x*x_mask)
    
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
        classes_dim: int,
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
    ):
        super().__init__()
        
        # outmost (input and output) blocks 
        self.stride = stride
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
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
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

    def forward(self, x: Tensor, x_mask: Tensor, t: Tensor, c: Tensor, **kwargs):

        # input transform
        x = self.to_in(x, x_mask)
        masks = [x_mask[:, :, ::self.stride]]

        # unet
        # time embedding
        t = self.to_time(t)
        skips_list = []
        for i, downsample in enumerate(self.downsamples):
            mask_down = masks[-1]
            x, skips, mask_ds = downsample(x, mask_down, t, c)
            masks.append(mask_ds)
            skips_list += [skips]

        mask_mid = masks[-1]
        x = self.bottleneck(x, mask_mid, t, c)
        
        for i, upsample in enumerate(self.upsamples):
            mask_up = masks.pop()
            skips = skips_list.pop()
            x = upsample(x, mask_up, skips, t, c)
            
        # output transform
        out_mask = masks[0]
        x = self.to_out(x, out_mask)  # t?
        return x * x_mask

class UNet1dBase(nn.Module):
    # unconditonal unet1d based diffusion
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # input transform to improve efficiency
        self.unet = UNet1d(classes_dim=0, **kwargs)

    def forward(self, x: Tensor, x_mask: Tensor, t: Tensor, classes: Tensor, **kwargs):

        c = None
        x = self.unet(x, x_mask, t, c)
        return x
    
class UNet1dCFG(nn.Module):
    # unet1d based diffusion with CFG
    
    def __init__(self, channels: int, num_classes: int, 
                 cond_drop_prob: float, # conditional dropout of the time, must be greater than 0. to unlock classifier free guidance
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

    def forward(self, x: Tensor, t: Tensor, classes: Tensor, 
                cond_drop_prob=None, x_mask: Tensor=None, **kwargs):
        
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
                null_classes_emb)
            
        c = self.classes_mlp(classes_emb)

        # unet
        x = self.unet(x, x_mask, t, c)
        return x

# example

if __name__ == '__main__':
    num_classes = 10

    model = UNet1dCFG(
        num_classes = num_classes,
        cond_drop_prob = 0.1,
        num_filters=128,
        window_length=32, 
        stride=16,
        in_channels = 1,
        channels = 128,
        resnet_groups = 8,
        kernel_multiplier_downsample = 2,
        multipliers = [1, 2, 4, 4, 4, 4, 4],
        factors = [4, 4, 4, 2, 2, 2],
        num_blocks = [2, 2, 2, 2, 2, 2],
        attentions = [False, False, False, True, True, True],
        attention_heads = 8,
        attention_features = 64,
        attention_multiplier = 2,
        use_nearest_upsample = True,
        use_skip_scale = True,
        use_attention_bottleneck = True)
    

    image_classes = torch.randint(0, num_classes, (2,))    # say 10 classes
    normal = -3.0  + 1.0 * torch.randn((2,), device=image_classes.device)
    sigmas = normal.exp()
    c_noise = torch.log(sigmas) * 0.25

    training_image_1 = torch.ones(1, 1, 2**15) # images are normalized from 0 to 1
    training_image_1[:,:,-2**14:] = 2.0 # images are normalized from 0 to 1
    training_image_2 = torch.ones(1, 1, 2**15) # images are normalized from 0 to 1
    training_images = torch.cat((training_image_1, training_image_2), 0)
    import numpy as np
    images_mask = sequence_mask(torch.from_numpy(np.array([2**14, 2**14])), 2**15).unsqueeze(1)
    training_images = training_images * images_mask
    x_hat = model(training_images, c_noise, image_classes, x_mask=images_mask)
    print(x_hat)
#     exit()
    
    training_image_1 = torch.ones(1, 1, 2**16) # images are normalized from 0 to 1
    training_image_1[:,:,-3*2**14:] = 0.0 # images are normalized from 0 to 1
    training_image_2 = torch.ones(1, 1, 2**16) # images are normalized from 0 to 1
    training_images = torch.cat((training_image_1, training_image_2), 0)
    import numpy as np
    images_mask = sequence_mask(torch.from_numpy(np.array([2**14, 2**14])), 2**16).unsqueeze(1)
    training_images = training_images * images_mask
    x_hat = model(training_images, c_noise, image_classes, x_mask=images_mask)
    print(x_hat)
    exit()
    torchaudio.save('./demo.wav', x_hat.squeeze(0), 44100)