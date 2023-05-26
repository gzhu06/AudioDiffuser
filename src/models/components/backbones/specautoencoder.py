import torch
import torch.nn as nn
from functools import partial
from .unet2d import ResnetBlock, LinearAttentionTransformerBlock, Identity, CrossEmbedLayer, TransformerBlock, Parallel, \
    Downsample, Upsample, PixelShuffleUpsample, UpsampleCombiner, zero_init_
from .utils import default, cast_tuple, exists
from .rvqautoencoder import leaky_relu
import torch.nn as nn
from typing import List, Union

class NLayerDiscriminator(nn.Module):
    def __init__(self, channels = 16,
                 layers=4,
                 groups = 4,
                 chan_max = 1024,
                 input_channels = 1):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, channels, 4)
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _ in range(layers):
            chan_out = min(curr_channels*4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(curr_channels, chan_out, 4, 
                          stride=2, padding=2, groups=groups),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, 3),
            leaky_relu(),
            nn.Conv2d(curr_channels, 1, 1),
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

class SpecAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_resnet_blocks,
        channels = 1,
        channels_out = None,
        subband = 1,
        dim_mults=[1, 2, 4, 8],
        attn_heads = 8,
        attn_dim_head = 64,
        use_linear_attn=False,
        resnet_groups = 8,
        layer_attns=[False, False, True, True],
        layer_attns_depth = 1,
        layer_cross_attns=[False, False, True, True],
        use_linear_cross_attn = False,
        init_dim = None,
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        init_conv_kernel_size = 7,    # kernel size of initial conv, if not using cross embed
        layer_mid_attns_depth = 1,
        attend_at_middle = True, 
        ff_mult = 2.,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        resize_mode = 'nearest',
        self_cond = False,
        use_global_context_attn = True,
        memory_efficient = True,
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
    ):
        super().__init__()

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        # determine dimensions
        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(self_cond))
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond

        # initial convolution
        if init_cross_embed:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out=init_dim, 
                                             kernel_sizes=init_cross_embed_kernel_sizes, 
                                             stride=1)
        else:
            self.init_conv = nn.Conv2d(init_channels, init_dim, 
                                       init_conv_kernel_size, 
                                       padding=init_conv_kernel_size//2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # attention related params
        attn_kwargs = dict(heads=attn_heads, dim_head = attn_dim_head)
        num_layers = len(in_out)

        # resnet block klass
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)
        
        layer_attns = cast_tuple(list(layer_attns))
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(list(layer_cross_attns))

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers==num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])
        
        # downsample klass
        downsample_klass = Downsample

        # initial resnet block (for memory efficient unet)
        if memory_efficient:
            self.init_resnet_block = resnet_klass(init_dim, init_dim, 
                                                  time_cond_dim=None, 
                                                  groups=resnet_groups[0], 
                                                  use_gca=use_global_context_attn) 
        else:
            self.init_resnet_block = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, 
                        resnet_groups, 
                        layer_attns, 
                        layer_attns_depth, 
                        layer_cross_attns, 
                        use_linear_attn, 
                        use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet
            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet
            post_downsample = None
            if not memory_efficient:
                if not is_last:
                    post_downsample = downsample_klass(current_dim, dim_out)  
                else: 
                    post_downsample = Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), 
                                               nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, 
                             current_dim, 
                             cond_dim=None, 
                             linear_attn = layer_use_linear_cross_attn, 
                             time_cond_dim=None, groups = groups),
                nn.ModuleList([
                    ResnetBlock(
                        current_dim, 
                        current_dim, 
                        time_cond_dim=None, 
                        groups = groups, 
                        use_gca = use_global_context_attn
                    ) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, 
                                        depth = layer_attn_depth, 
                                        ff_mult = ff_mult, 
                                        context_dim=None, 
                                        **attn_kwargs),
                post_downsample]))

        # middle layers
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, 
                                      mid_dim, 
                                      cond_dim=None, 
                                      time_cond_dim=None, 
                                      groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, 
                                         depth=layer_mid_attns_depth, 
                                         **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, 
                                      mid_dim, 
                                      cond_dim=None, 
                                      time_cond_dim=None, 
                                      groups = resnet_groups[-1])

        # upsample klass
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers
        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity


            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out, 
                             dim_out, 
                             cond_dim=None, 
                             linear_attn=layer_use_linear_cross_attn, 
                             time_cond_dim=None, 
                             groups=groups),
                nn.ModuleList([
                    ResnetBlock(dim_out, 
                                dim_out, 
                                time_cond_dim=None, 
                                groups=groups, 
                                use_gca=use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim=dim_out, 
                                        depth=layer_attn_depth, 
                                        ff_mult=ff_mult, 
                                        context_dim=None,
                                        **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # final optional resnet block and convolution out
        final_conv_dim = self.upsample_combiner.dim_out
        if final_resnet_block:
            self.final_res_block = ResnetBlock(final_conv_dim, dim, 
                                               time_cond_dim=None, 
                                               groups=resnet_groups[0], 
                                               use_gca=True) 
        else:
            self.final_res_block = None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = nn.Conv2d(final_conv_dim_in, 
                                    self.channels_out, 
                                    final_conv_kernel_size, 
                                    padding=final_conv_kernel_size//2)

        zero_init_(self.final_conv)

        # resize mode
        self.resize_mode = resize_mode

    def encoder(self, x, bottlenecks: Union[nn.Module, List[nn.Module]]):
        batch_size, device = x.shape[0], x.device

        x = self.freq_split_subband(x)

        # condition on self
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)
            
        # initial convolution
        x = self.init_conv(x)

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x)

        # go through the layers of the unet, down and up
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)

            if exists(post_downsample):
                x = post_downsample(x)
        
        info_encoder = {}
        for bottleneck in bottlenecks:
            x, btnk_info = bottleneck(x, with_info=True)
            if bool(btnk_info): # only one with info for now
                info_encoder = btnk_info
                
        return x, info_encoder
    
    def decoder(self, x):

        x = self.mid_block1(x)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x)

        up_hiddens = []
        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner(x, up_hiddens)

        if exists(self.final_res_block):
            x = self.final_res_block(x)

        return self.final_conv(x)
    
    def forward(self, inputs, bottlenecks, sample_posterior=True):
        
        h, info_encoder = self.encoder(inputs, bottlenecks)

        if sample_posterior:
            z = h
        elif bool(info_encoder):
            z = info_encoder['variational_mean']

        dec = self.decoder(z)
        dec = self.freq_merge_subband(dec)

        return (dec, info_encoder)
    
    def freq_split_subband(self, fbank):
        if self.subband == 1:
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1:
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

if __name__ == '__main__':
    from bottleneck import VariationalBottleneck

    model = SpecAutoEncoder(channels=1, channels_out=1, dim=16, num_resnet_blocks=4)#.cuda()
    vb = VariationalBottleneck(channels=128, dim=2)
    bs = 1
    training_images = torch.randn(bs, 1, 256, 256) # images are normalized from 0 to 1

    x_hat = model(training_images, vb)