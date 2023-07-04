from typing import Sequence, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchaudio import transforms
from einops import pack, unpack, rearrange
from torch import Tensor
import numpy as np
from typing import List, Union
from .unet1d import UNet1d, DownsampleBlock1d, BottleneckBlock1d, ResnetBlock1d
from .utils import groupby

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        sample_rate: int = 16000,
        n_mel_channels: int = 80,
        center: bool = False,
        normalize: bool = False,
        normalize_log: bool = False,
    ):
        super().__init__()
        self.padding = (n_fft - hop_length) // 2
        self.normalize = normalize
        self.normalize_log = normalize_log
        self.hop_length = hop_length

        self.to_spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            power=None,
        )

        self.to_mel_scale = transforms.MelScale(
            n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate
        )

    def forward(self, waveform: Tensor) -> Tensor:
        # Pack non-time dimension
        waveform, ps = pack([waveform], "* t")
        # Pad waveform
        waveform = F.pad(waveform, [self.padding] * 2, mode="reflect")
        # Compute STFT
        spectrogram = self.to_spectrogram(waveform)
        # Compute magnitude
        spectrogram = torch.abs(spectrogram)
        # Convert to mel scale
        mel_spectrogram = self.to_mel_scale(spectrogram)
        # Normalize
        if self.normalize:
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # Unpack non-spectrogram dimension
        return unpack(mel_spectrogram, ps, "* f l")[0]

class ConditionEncoder(nn.Module):
    """
    Encode mel spec features into semantic embedding (vector), following the 
    original diffautoencoder design.

    Architecture: the half UNet model with attention without timestep embedding.

    For usage, see UNet.
    """
    def __init__(
        self,
        # efficient input transform:
        mel_channels: int,
        # unet dsblock and bottleneck
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
        use_attention_bottleneck: bool,
        output_embeddings: Optional[bool]=False,
        **kwargs
    ):
        super().__init__()
        
        # outmost input blocks 
        mel_kwargs, kwargs = groupby("mel_", kwargs)
        self.mel = MelSpectrogram(n_mel_channels=mel_channels, **mel_kwargs)
        
        num_layers = len(multipliers) - 1
        self.num_layers = num_layers
        
        assert (len(factors) == num_layers
                and len(attentions) == num_layers
                and len(num_blocks) == num_layers
               )

        # entry conv block
        self.to_in = ResnetBlock1d(
            in_channels=mel_channels,
            out_channels=channels,
            num_groups=resnet_groups,
        )

        # unet
        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    time_embed_dim=None, 
                    classes_embed_dim=None,
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
            time_embed_dim=None, 
            classes_embed_dim=None
        )
        
        self.output_embeddings = output_embeddings
        if self.output_embeddings:
            self.out = nn.Sequential(
                nn.GroupNorm(num_groups=resnet_groups, num_channels=channels * multipliers[-1]),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels * multipliers[-1], channels * multipliers[-1]),
            )

    def forward(self, x: Tensor,
                output_bottleneck:nn.Module, 
                with_info: bool = False):
        
        # input transform
        x = rearrange(self.mel(x), "b c f l -> b (c f) l")
        x = self.to_in(x)

        # unet downsampling blocks
        for _, downsample in enumerate(self.downsamples):
            x, _ = downsample(x, t=None)

        x = self.bottleneck(x, t=None)
        
        if self.output_embeddings:
            # output MLP transform
            x = self.out(x)
            x, info= output_bottleneck(x, with_info=True)
            x = rearrange(x, "b c -> b c 1")
        else:
            x, info= output_bottleneck(x, with_info=True)

        return (x, info) if with_info else x
    
def assign_conditions(conditions: Union[Tensor, List[Tensor]]):
    
    embeddings, channels = None, None
    if isinstance(conditions, list) and len(conditions) != 1:
        assert len(conditions) == 2  # only support 2 for now
        embeddings, channels = (conditions[0], conditions[1]) if conditions[0].shape[-1] == 1 else (conditions[1], conditions[0])
    else:
        assert len(conditions) == 1
        condition = conditions[0]
        if condition.shape[-1] == 1:
            embeddings = condition
        else:
            channels = condition

    return embeddings, channels

class DiffAutoEncoder(nn.Module):
    """Diffusion Auto Encoder"""
    def __init__(self, **kwargs):
        super().__init__()
        
        # input transform to improve efficiency
        self.unet = UNet1d(**kwargs)
        
    def forward(self, x: Tensor, 
                t: Tensor=None, 
                classes: Tensor=None, 
                x_start: Tensor=None, 
                encoded_x: Union[Tensor, List[Tensor]]=None, 
                x_encoders: Union[nn.Module, List[nn.Module]]=None, 
                bottleneck: nn.Module=None, 
                **kwargs):
        """
        Apply the model to an input batch.
        Args:

            x_encode: external encoded x
        """

        c = None
        if encoded_x is not None:
            inj_embeddings, inj_channels = assign_conditions(encoded_x)
        else:
            cond_xs = []
            for x_encoder in x_encoders:
                cond_x = x_encoder(x_start, bottleneck)
                cond_xs.append(cond_x)
            inj_embeddings, inj_channels = assign_conditions(cond_xs)
        
        x = self.unet(x, t, c, inj_channels=inj_channels, inj_embeddings=inj_embeddings)

        return x
    
    def interpolate_semantic(self, x, x_encoder, bottleneck):

        assert x.shape[0] == 2
        
        x_semantic = x_encoder(x, bottleneck).squeeze(2)
        alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(x.device)
        
        intp = x_semantic[0][None] * (1 - alpha[:, None]) + x_semantic[1][None] * alpha[:, None]
        return intp.unsqueeze(2)
    

class DiffCompSpecAutoEncoder(nn.Module):

    pass