""" Diffusion Classes """

from math import pi
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
from .distribution import Distribution
from .utils import extend_dim, clip, to_batch

class EluDiffusion(nn.Module):
    """Elucidated Diffusion Models(EDM): https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        x_classes: Tensor,
        net: nn.Module = None,
        inference: bool = False,
        cond_scale: float = 1.0,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        x_mask: Optional[Tensor] = None,
        **kwargs) -> Tensor:

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        
        # Predict network output
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy.ndim)
        x_pred = net(c_in*x_noisy, c_noise, classes=x_classes, x_mask=x_mask, **kwargs)
        
        # cfg interpolation during inference, skip during training
        if inference and cond_scale != 1.0:
            null_logits = net(c_in*x_noisy, c_noise, x_classes, 
                              cond_drop_prob=1., x_mask=x_mask, **kwargs)
            x_pred = null_logits + (x_pred - null_logits) * cond_scale

        # eq.7
        x_denoised = c_skip * x_noisy + c_out * x_pred
        
        # Clips in [-1,1] range, with dynamic thresholding if provided
        return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, 
                x_classes: Tensor,
                net: nn.Module, 
                sigma_distribution: Distribution, 
                x_mask: Tensor = None,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size, device = x.shape[0], x.device
        
        if x_mask is None:
            x_mask = torch.ones(x.size(), device=device)

        # Sample amount of noise to add for each batch element
        sigmas = sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        # Compute denoised values
        x_noisy = x + sigmas_padded * noise * x_mask
        x_denoised = self.denoise_fn(x_noisy, x_classes, 
                                     net, sigmas=sigmas, 
                                     x_mask=x_mask, 
                                     inference=inference, 
                                     cond_scale=cond_scale,
                                     **kwargs)
        
        # noise level weighted loss (weighted eq.2)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "sum")
        losses = losses * self.loss_weight(sigmas) / torch.sum(x_mask)
        losses = losses.mean()

        return losses

class VKDiffusion(nn.Module):

    alias = "vk"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = 1.0
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = -sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return (t * pi / 2).tan()

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = torch.randn_like(x)
        x_noisy = x + sigmas_padded * noise

        # Compute model output
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-7)

        # Compute loss
        loss = F.mse_loss(x_pred, v_target)
        return loss