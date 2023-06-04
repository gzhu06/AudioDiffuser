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
from abc import ABC, abstractmethod

class Diffusion(nn.Module):
    
    @abstractmethod
    def loss_weight(self):
        pass
    
    @abstractmethod
    def forward(self, x: Tensor):
        pass
    
    @abstractmethod
    def get_scale_weights(self):
        pass
    
    @abstractmethod
    def denoise_fn(self):
        pass
    
    
class VEDiffusion(Diffusion):
    def __init__(
        self,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
        
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1
        c_noise = torch.log(0.5 * sigmas)
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = sigmas
        c_in = 1
        
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
        return 1/(sigmas**2)
    
    def forward(self, x: Tensor, 
                x_classes: Tensor,
                net: nn.Module, 
                distribution: Distribution, 
                x_mask: Tensor = None,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size, device = x.shape[0], x.device
        
        if x_mask is None:
            x_mask = torch.ones(x.size(), device=device)

        # Sample amount of noise to add for each batch element
        sigmas = distribution(num_samples=batch_size, device=device)
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
    
class VPDiffusion(Diffusion):
    """VP Diffusion Models formulated by EDM"""

    def __init__(
        self,
        beta_min: float, 
        beta_d: float,
        M: float,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.M = M
        self.dynamic_threshold = dynamic_threshold

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return sigmas ** (-2)
    
    def t_to_sigma(self, t):
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def sigma_to_t(self, sigmas):
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigmas ** 2).log()).sqrt() - self.beta_min) / self.beta_d
    
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        c_noise = (self.M - 1) * self.sigma_to_t(sigmas)
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = - sigmas
        c_in = 1 / (sigmas ** 2 + 1).sqrt()
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

    def forward(self, x: Tensor, 
                x_classes: Tensor,
                net: nn.Module, 
                distribution: Distribution, # t distribution
                x_mask: Tensor = None,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size, device = x.shape[0], x.device
        
        if x_mask is None:
            x_mask = torch.ones(x.size(), device=device)

        # Sample amount of noise to add for each batch element
        ts = distribution(num_samples=batch_size, device=device)
        sigmas = self.t_to_sigma(ts)
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

class EluDiffusion(Diffusion):
    """Elucidated Diffusion Models(EDM): https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        sigma_data: float,  # data distribution
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
                distribution: Distribution, # sigma distribution
                x_mask: Tensor = None,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        batch_size, device = x.shape[0], x.device
        
        if x_mask is None:
            x_mask = torch.ones(x.size(), device=device)

        # Sample amount of noise to add for each batch element
        sigmas = distribution(num_samples=batch_size, device=device)
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
