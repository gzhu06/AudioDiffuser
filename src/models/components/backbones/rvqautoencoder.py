'''
https://github.com/lucidrains/audiolm-pytorch.git
'''
import functools
from itertools import cycle
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias
from .utils import exists

# helper functions

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# better sequential
def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# discriminators
class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = 4,
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 7)
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _ in range(layers):
            chan_out = min(curr_channels*4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 8, 
                          stride=4, padding=4, groups=groups),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 3),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 1),
        )

    def forward(self, x, return_intermediates=False):
        x = self.init_conv(x)

        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

# complex stft discriminator
class ModReLU(nn.Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        ComplexConv2d(chan_in, chan_in, 3, padding = 1),
        ModReLU(),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class ComplexSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = False
    ):
        super().__init__()
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        self.layers = nn.ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

        # stft settings

        self.stft_normalized = stft_normalized

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, compute_stft=True, return_intermediates=False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        if compute_stft:
            x = torch.stft(
                x,
                self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                normalized = self.stft_normalized,
                return_complex = True
            )

        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)
        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        complex_logits = self.final_conv(x)

        complex_logits_abs = torch.abs(complex_logits)

        if not return_intermediates:
            return complex_logits_abs

        return complex_logits_abs, intermediates

# sound stream
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class Conv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.padding//2, self.padding-self.padding//2))
        return self.conv(x)

class ConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = int((kernel_size - stride) / 2)
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, 
                                       kernel_size, stride, 
                                       padding=self.padding,
                                       **kwargs)

    def forward(self, x):
        n = x.shape[-1]
        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]
        return out

def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7):
    return Residual(Sequential(
        Conv1d(chan_in, chan_out, kernel_size, dilation=dilation),
        nn.ELU(),
        Conv1d(chan_out, chan_out, 1),
        nn.ELU()
    ))

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        Conv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):

    residual_unit = partial(ResidualUnit)

    it = cycle(cycle_dilations)
    return nn.Sequential(
        ConvTranspose1d(chan_in, chan_out, 2*stride, stride=stride),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim = dim, heads=heads, qk_rmsnorm = True, 
                         window_size = window_size, 
                         use_rotary_pos_emb = not dynamic_pos_bias, 
                         use_xpos = True, **kwargs),
                FeedForward(dim = dim)
            ]))

    def forward(self, x, x_mask=None):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, mask=x_mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        return x

class VQAutoEncoer(nn.Module):
    def __init__(
        self,
        *,
        channels = 16,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        input_channels = 1,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        use_local_attn = False,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
    ):
        super().__init__()

        # rest of the class

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []
        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations))

        self.encoder = nn.Sequential(
            Conv1d(input_channels, channels, 7),
            *encoder_blocks,
            Conv1d(layer_channels[-1], codebook_dim, 3)
        )

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            depth = attn_depth,
            window_size = attn_window_size,
            xpos_scale_base = attn_xpos_scale_base,
            dynamic_pos_bias = attn_dynamic_pos_bias,
            prenorm = True,
            causal = False
        )

        self.encoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None
        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None
        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations))

        self.decoder = nn.Sequential(
            Conv1d(codebook_dim, layer_channels[-1], 7),
            *decoder_blocks,
            Conv1d(channels, input_channels, 7)
        )

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        vq_bottleneck,
        x_mask = None, # placeholder
        return_encoded = False,
    ):
        
        x, ps = pack([x], '* n')
        x = curtail_to_multiple(x, self.seq_len_multiple_of)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')
            
        orig_x = x.clone()

        if x_mask is None:
            x_mask = torch.ones(x.size(), device=x.device)

        # encoding
        x = self.encoder(x)
        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        x, indices, commit_loss = vq_bottleneck(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        if return_encoded:
            return x, indices, commit_loss

        recon_x = self.decoder(x)

        recon_x, = unpack(recon_x, ps, '* c n')
        return orig_x, recon_x, commit_loss

if __name__ == '__main__':
    
    autoencoder = VQAutoEncoer()

    x = torch.randn(1, 2**18)    # [1, 2, 262144]
    from bottleneck import VQBottleneck
    vqbtnk = VQBottleneck(dim=512, codebook_size=1024)
    total_loss, other_loss = autoencoder(x, vq_bottleneck=vqbtnk, return_loss_breakdown=True)        # [1, 2, 262144]
    print(total_loss, other_loss)