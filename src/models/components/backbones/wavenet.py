# WaveNet
# Based on https://github.com/philsyn/DiffWave-unconditional
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding)
        
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = WeightNorm(self.conv, ['weight'])
#         self.conv = nn.utils.weight_norm(self.conv)
#         nn.init.kaiming_normal_(self.conv.weight)
        
    def forward(self, x):
        out = self.conv(x)
        return out

@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)

def diffusion_embedding(diffusion_step, dim_in):
    dim_vec = torch.arange(dim_in//2).to(diffusion_step.device)
    table = diffusion_step.unsqueeze(1) * torch.exp(-dim_vec * 4.0 / (dim_in//2-1))     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        '''
        super().__init__()
        self.dilated_conv = Conv(residual_channels, 2 * residual_channels, 3, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        
        # residual conv1x1 layer, connect to next residual layer
        self.output_projection = Conv(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_embed):
        diffusion_embed = self.diffusion_projection(diffusion_embed).unsqueeze(-1)
        y = x + diffusion_embed
        y = self.dilated_conv(y)
        gate, filters = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(gate) * torch.tanh(filters)
        out = self.output_projection(out)
        residual, skip = torch.chunk(out, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
    
class ResidualGroup(nn.Module):
    def __init__(self, res_channels, skip_channels, 
                 num_res_layers=30, dilation_cycle=10, 
                 dim_in=128, dim_mid=512, dim_out=512):
        
        super(ResidualGroup, self).__init__()
        self.num_res_layers = num_res_layers
        self.dim_in = dim_in

        # the shared two fc layers for diffusion step embedding
        self.fc_t1 = nn.Linear(dim_in, dim_mid)
        self.fc_t2 = nn.Linear(dim_mid, dim_out)
        
        # stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(ResidualBlock(res_channels, 
                                                      dilation=2**(n%dilation_cycle)))

    def forward(self, x, diffusion_step):

        # embed diffusion step t
        diffusion_step_embed = diffusion_embedding(diffusion_step, self.dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            # use the output from last residual layer
            h, skip_n = self.residual_blocks[n](h, diffusion_step_embed)  
            skip += skip_n  # accumulate all skip outputs

        return skip * sqrt(1.0 / self.num_res_layers)  # normalize for training stability

class WaveNetNoise(nn.Module):
    def __init__(self, 
                 residual_channels=256, 
                 residual_layers=36,
                 dilation_cycle=12):
        super().__init__()
        self.input_projection = Conv(1, residual_channels, 1)

        self.residual_layer = ResidualGroup(res_channels=residual_channels, 
                                            skip_channels=residual_channels, 
                                            num_res_layers=residual_layers, 
                                            dilation_cycle=dilation_cycle)
        
        self.skip_projection = Conv(residual_channels, 
                                    residual_channels, 1)
        self.output_projection = ZeroConv1d(residual_channels, 1)

    def forward(self, audio, diffusion_step):
    
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)      
        x = self.residual_layer(x, diffusion_step)
        
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
