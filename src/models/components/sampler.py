from math import sqrt, cos, sin, pi
from typing import Callable, Tuple, Optional
from einops import repeat
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from .utils import extend_dim

""" Samplers """

class EDMSampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) sampler:
    
    Deterministic sampler (by default, the noise becomes 0 in line 6 in Algorithm 2): 
    s_churn=0, s_noise=1, s_tmin=0, s_tmax=float('inf')
    the second order correction seems to be problematic in deterministic mode 
    
    Stochastic sampler: 
    s_churn=40 s_noise=1.003, s_tmin=0.05, s_tmax=50 

    Because EDM uses second order ODE solver, the NFE is around the 
    twice the number of steps. 
    
    """

    def __init__(
        self,
        s_tmin: float = 0.0001,
        s_tmax: float = 3.0,
        s_churn: float = 150.0,
        s_noise: float = 1.04,
        num_steps: int = 200,
        cond_scale: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.num_steps = num_steps
        self.cond_scale = cond_scale

    def step(self, x: Tensor, 
             x_classes: Tensor, 
             fn: Callable, net: nn.Module, 
             sigma: float, sigma_next: float, 
             gamma: float, x_mask: Tensor=None, 
             use_heun: bool=True, **kwargs) -> Tensor:
        
        """One step of EDM sampler"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma

        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon

        # Evaluate ∂x/∂sigma at sigma_hat
        denoised_cur =  fn(x_hat, x_classes, net=net, 
                           sigma=sigma_hat, inference=True, 
                           cond_scale=self.cond_scale, 
                           x_mask=x_mask, **kwargs)
        d = (x_hat - denoised_cur) / sigma_hat

        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d

        # Second order correction
        if sigma_next != 0 and use_heun:
            denoised_next = fn(x_next, x_classes, 
                               net=net, sigma=sigma_next, 
                               inference=True, 
                               cond_scale=self.cond_scale, 
                               x_mask=x_mask, 
                               **kwargs)
            d_prime = (x_next - denoised_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_prime)

        return x_next
    
    def reverse_step(self, x: Tensor,
                     x_classes: Tensor,
                     fn: Callable, 
                     net: nn.Module,
                     sigma: float, 
                     sigma_next: float,
                     x_mask: Tensor=None,
                     **kwargs) -> Tensor:
        
        "reverse sampling for encoding latents with Euler sampler"
        denoised_cur = fn(x, x_classes, net=net, 
                          sigma=sigma, inference=True, 
                          cond_scale=self.cond_scale, 
                          x_mask=x_mask, **kwargs)
        
        # first order
        x0_t = (x - denoised_cur) / sigma
        xt_next = sigma_next * x0_t + denoised_cur
        
        # second order
        denoised_next = fn(xt_next, x_classes, net=net, 
                           sigma=sigma_next, inference=True, 
                           cond_scale=self.cond_scale, 
                           x_mask=x_mask, **kwargs)
        x0_t_next = (xt_next - denoised_next) / sigma_next
        xt_next = x + (sigma_next - sigma) / 2 * (x0_t + x0_t_next)

        return xt_next

    def encode(self, x_input:Tensor, 
               x_classes:Tensor,
               fn: Callable, 
               net: nn.Module, 
               sigmas: Tensor, 
               decode: bool=False,
               **kwargs) -> Tensor:
        
        reversed_sigmas = sigmas.flip(0)[1:]

        # Denoise to sample
        x = x_input
        for i in range(self.num_steps-1):
            x = self.reverse_step(x, x_classes, 
                                  fn=fn, net=net, 
                                  sigma=reversed_sigmas[i], 
                                  sigma_next=reversed_sigmas[i+1], 
                                  **kwargs)
            
        if decode:
            for j in range(self.num_steps-1):
                x = self.step(x, x_classes, 
                              fn=fn, net=net, 
                              sigma=sigmas[j], 
                              sigma_next=sigmas[j+1], 
                              gamma=0.0, 
                              use_heun=True, 
                              **kwargs)
        return x
        

    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, 
                use_heun: bool=True,
                **kwargs) -> Tensor:
        
        # pay attention to this step
        x = sigmas[0] * noise

        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )

        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, x_classes, 
                          fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          gamma=gammas[i], 
                          use_heun=use_heun, 
                          **kwargs)
            
        return x

class AEulerSampler(nn.Module):

    # diffusion_types = [KDiffusion, VKDiffusion]

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Euler method
        x_next = x + d * (sigma_down - sigma)
        # Add randomness
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise

        # Denoise to sample
        for i in range(num_steps-1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i+1])  # type: ignore # noqa
        return x

class ADPM2Sampler(nn.Module):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""
    # for inference only

    def __init__(self, rho: float = 1.0, num_steps: int = 50, cond_scale: float=1.0):
        super().__init__()
        self.rho = rho
        self.num_steps = num_steps
        self.cond_scale = cond_scale

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, 
             x_classes: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             sigma: float, 
             sigma_next: float, 
             x_mask: Tensor=None, 
             **kwargs) -> Tensor:
        
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        
        x_epis = fn(x, x_classes, 
                    x_mask=x_mask, 
                    net=net, 
                    sigma=sigma, 
                    cond_scale=self.cond_scale, 
                    **kwargs)
        d = (x - x_epis) / sigma
        
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)

        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        x_mid_epis = fn(x_mid, x_classes, 
                        x_mask=x_mask, 
                        net=net, sigma=sigma_mid,
                        inference=True, 
                        cond_scale=self.cond_scale,
                        **kwargs)
        d_mid = (x_mid - x_mid_epis) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)

        if x_mask is not None:
            x_next = x + torch.randn_like(x) * sigma_up * x_mask
        else:
            x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                x_mask: Tensor=None, **kwargs) -> Tensor:
        
        x = sigmas[0] * noise
        
        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, x_classes, 
                          x_mask=x_mask, fn=fn, 
                          net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          **kwargs)  # type: ignore # noqa

        return x.clamp(-1.0, 1.0)