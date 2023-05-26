from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from .utils import default, exists, prefix_dict
from .unet1d import Conv1d, Downsample1d, Upsample1d, ConvBlock1d, WAVenc1d, WAVdec1d 

"""
Convolutional Modules
"""

class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)

class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
    ):
        super().__init__()

        self.downsample = Downsample1d(
            in_channels=in_channels, out_channels=out_channels, factor=factor
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


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
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_groups=num_groups,
                )
                for _ in range(num_layers)
            ]
        )

        self.upsample = Upsample1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            factor=factor,
            use_nearest=use_nearest,
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.upsample(x)
        return x


"""
Encoders / Decoders
"""

class Encoder1d(nn.Module):
    def __init__(
        self,
        num_filters: int,
        window_length: int, 
        stride: int,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int = 8,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = len(multipliers) - 1
        self.out_channels = (
            out_channels if exists(out_channels) else channels * multipliers[-1]
        )
        assert len(factors) == self.num_layers and len(num_blocks) == self.num_layers

        self.to_in = WAVenc1d(in_channels=in_channels, 
                              num_filters=num_filters, 
                              window_length=window_length, 
                              stride=stride)

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factors[i],
                    num_groups=resnet_groups,
                    num_layers=num_blocks[i],
                )
                for i in range(self.num_layers)
            ]
        )

        self.to_out = (
            nn.Conv1d(
                in_channels=channels * multipliers[-1],
                out_channels=out_channels,
                kernel_size=1,
            )
            if exists(out_channels)
            else nn.Identity()
        )

    def forward(self, x: Tensor, with_info: bool = False) -> Union[Tensor, Tuple[Tensor, Any]]:
        xs = [x]
        x = self.to_in(x)
        xs += [x]

        for downsample in self.downsamples:
            x = downsample(x)
            xs += [x]

        x = self.to_out(x)
        xs += [x]
        info = dict(xs=xs)

        return (x, info) if with_info else x

class Decoder1d(nn.Module):
    def __init__(
        self,
        num_filters: int,
        window_length: int, 
        stride: int,
        out_channels: int,
        channels: int,
        use_nearest_upsample: bool,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int = 8,
        in_channels: Optional[int] = None,
    ):
        super().__init__()
        num_layers = len(multipliers) - 1

        assert len(factors) == num_layers and len(num_blocks) == num_layers

        self.to_in = (
            Conv1d(
                in_channels=in_channels,
                out_channels=channels * multipliers[0],
                kernel_size=1,
            )
            if exists(in_channels)
            else nn.Identity()
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factors[i],
                    num_groups=resnet_groups,
                    num_layers=num_blocks[i],
                    use_nearest=use_nearest_upsample
                )
                for i in range(num_layers)
            ]
        )

        print(channels * multipliers[-1], out_channels)

        self.to_out = WAVdec1d(in_channels=out_channels, 
                               num_filters=num_filters, 
                               window_length=window_length, 
                               stride=stride,
                               out_channels=out_channels)

    def forward(self, x: Tensor, with_info: bool = False) -> Union[Tensor, Tuple[Tensor, Any]]:

        xs = [x]
        x = self.to_in(x)
        xs += [x]

        for upsample in self.upsamples:
            x = upsample(x)
            xs += [x]

        x = self.to_out(x)
        xs += [x]

        info = dict(xs=xs)
        return (x, info) if with_info else x
        
class ContinousAutoEncoder1d(nn.Module):
    
    def __init__(
        self,
        num_filters: int,
        window_length: int, 
        stride: int,
        in_channels: int,
        channels: int,
        use_nearest_upsample: bool,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int = 8,
        out_channels: Optional[int] = None,
        bottleneck_channels: Optional[int] = None,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        
        self.encoder = Encoder1d(
            num_filters=num_filters, 
            window_length=window_length, 
            stride=stride,
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            channels=channels,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            resnet_groups=resnet_groups,
        )

        self.decoder = Decoder1d(
            num_filters=num_filters, 
            window_length=window_length, 
            stride=stride,
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            channels=channels,
            use_nearest_upsample=use_nearest_upsample,
            multipliers=multipliers[::-1],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            resnet_groups=resnet_groups,
        )
        
    def forward(self, x: Tensor, with_info: bool=False, 
                bottlenecks: Union[nn.Module, List[nn.Module]] = []) -> Union[Tensor, Tuple[Tensor, Any]]:
        
        z, info_encoder = self.encode(x, with_info=True, bottlenecks=bottlenecks)
        y, info_decoder = self.decode(z, with_info=True)
        info = {
            **dict(latent=z),
            **prefix_dict("encoder_", info_encoder),
            **prefix_dict("decoder_", info_decoder),
        }
        return (y, info) if with_info else y

    def encode(self, x: Tensor, with_info: bool = False, bottlenecks: Union[nn.Module, List[nn.Module]] = []) -> Union[Tensor, Tuple[Tensor, Any]]:
        
        x, info_encoder = self.encoder(x, with_info=with_info)
        for bottleneck in bottlenecks:
            x, info_bottleneck = bottleneck(x, with_info=True)
            info_encoder = {**info_encoder, **prefix_dict("bottleneck_", info_bottleneck)}
        return (x, info_encoder) if with_info else x

    def decode(self, x: Tensor, with_info: bool = False) -> Tensor:
        return self.decoder(x, with_info=with_info)
    
class VQAutoEncoder1d(nn.Module):
    
    def __init__(
        self,
        num_filters: int,
        window_length: int, 
        stride: int,
        in_channels: int,
        channels: int,
        use_nearest_upsample: bool,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        resnet_groups: int = 8,
        out_channels: Optional[int] = None,
        codebook_dim: int = 512,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        
        self.encoder = Encoder1d(
            num_filters=num_filters, 
            window_length=window_length, 
            stride=stride,
            in_channels=in_channels,
            out_channels=codebook_dim,
            channels=channels,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            resnet_groups=resnet_groups,
        )

        self.decoder = Decoder1d(
            num_filters=num_filters, 
            window_length=window_length, 
            stride=stride,
            in_channels=codebook_dim,
            out_channels=out_channels,
            channels=channels,
            use_nearest_upsample=use_nearest_upsample,
            multipliers=multipliers[::-1],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            resnet_groups=resnet_groups,
        )
        
    def forward(self, x: Tensor, 
                vq_bottleneck: nn.Module, 
                x_mask = None, # placeholder
                return_encoded: bool = False):
        
        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')
        
        z, indices, commit_loss = self.encode(x, vq_bottleneck=vq_bottleneck)
        
        if return_encoded:
            return z, indices, commit_loss
        
        y = self.decode(z)
        
        return x, y, commit_loss

    def encode(self, x: Tensor, vq_bottleneck: nn.Module):
        
        x = self.encoder(x, with_info=False)
        x = rearrange(x, 'b c n -> b n c')
        x, indices, commit_loss = vq_bottleneck(x)
        x = rearrange(x, 'b n c -> b c n')
        
        return x, indices, commit_loss

    def decode(self, x: Tensor) -> Tensor:
        
        return self.decoder(x, with_info=False)

"""
Discriminators
"""
class Discriminator1d(nn.Module):
    def __init__(self, use_loss: Optional[Sequence[bool]] = None, **kwargs):
        super().__init__()
        self.discriminator = Encoder1d(**kwargs)
        num_layers = self.discriminator.num_layers
        # By default we activate discrimination loss extraction on all layers
        self.use_loss = default(use_loss, [True] * num_layers)
        # Check correct length
        msg = f"use_loss length must match the number of layers ({num_layers})"
        assert len(self.use_loss) == num_layers, msg

    def forward(
        self, true: Tensor, fake: Tensor, with_info: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict]]:
        # Get discriminator outputs for true/fake scores
        _, info_true = self.discriminator(true, with_info=True)
        _, info_fake = self.discriminator(fake, with_info=True)

        # Get all intermediate layer features (ignore input)
        xs_true = info_true["xs"][1:]
        xs_fake = info_fake["xs"][1:]

        loss_gs, loss_ds, scores_true, scores_fake = [], [], [], []

        for use_loss, x_true, x_fake in zip(self.use_loss, xs_true, xs_fake):
            if use_loss:
                # Half the channels are used for scores, the other for features
                score_true, feat_true = x_true.chunk(chunks=2, dim=1)
                score_fake, feat_fake = x_fake.chunk(chunks=2, dim=1)
                # Generator must match features with true sample and fool discriminator
                loss_gs += [F.l1_loss(feat_true, feat_fake) - score_fake.mean()]
                # Discriminator must give high score to true samples, low to fake
                loss_ds += [((1 - score_true).relu() + (1 + score_fake).relu()).mean()]
                # Save scores
                scores_true += [score_true.mean()]
                scores_fake += [score_fake.mean()]

        # Average all generator/discriminator losses over all layers
        loss_g = torch.stack(loss_gs).mean()
        loss_d = torch.stack(loss_ds).mean()

        info = dict(scores_true=scores_true, scores_fake=scores_fake)

        return (loss_g, loss_d, info) if with_info else (loss_g, loss_d)


if __name__ == '__main__':
    
    autoencoder = VQAutoEncoder1d(
        stride=2,
        num_filters=32,
        window_length=4,
        use_nearest_upsample=True,
        in_channels=2,              # Number of input channels
        channels=64,                # Number of base channels
        multipliers=[1, 1, 2, 2],   # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
        factors=[4, 4, 4],          # Downsampling/upsampling factor per layer
        num_blocks=[2, 2, 2]        # Number of resnet blocks per layer
    )

    x = torch.randn(1, 2, 2**18)    # [1, 2, 262144]
    x_recon = autoencoder(x)        # [1, 2, 262144]
    print(x_recon.shape)