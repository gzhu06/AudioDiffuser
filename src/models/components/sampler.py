from math import sqrt, cos, sin, pi
from typing import Callable, Tuple, Optional
from einops import repeat
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from tqdm import tqdm
from .utils import extend_dim

""" Samplers """

class VESampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) VE sampler:
    """
    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
        s_churn: float = 200,
        s_noise: float = 1,
        num_steps: int = 200,
        cond_scale: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        
        # Define sigma helper function
        self.sigma_inv = lambda sigma: sigma ** 2
        self.sigma_deriv = lambda t: 0.5 / t.sqrt()
        self.sigma = lambda t: t.sqrt()
    
    def step(self, x: Tensor, 
             x_classes: Tensor, 
             fn: Callable, net: nn.Module, 
             t: float, t_next: float, 
             gamma: float, x_mask: Tensor=None, 
             use_heun: bool=True, **kwargs) -> Tensor:

        # Increase noise temporarily.
        t_hat = self.sigma_inv((self.sigma(t) + gamma * self.sigma(t)))
        x_hat = x + (self.sigma(t_hat)**2 - self.sigma(t)**2).clip(min=0).sqrt() * self.s_noise * torch.randn_like(x)

        # Euler step.
        denoised_cur = fn(x_hat, x_classes, 
                          net=net, sigma=self.sigma(t_hat), 
                          inference=True, cond_scale=self.cond_scale, 
                          x_mask=x_mask, **kwargs)
        
        d = (self.sigma_deriv(t_hat)/self.sigma(t_hat))*x_hat - self.sigma_deriv(t_hat)/self.sigma(t_hat) * denoised_cur
        h = t_next - t_hat
        x_next = x_hat + h * d
        
        # Apply 2nd order correction.
        if t_next != 0 and use_heun:
            t_prime = t_hat + h
            
            denoised_prime = fn(x_next, x_classes, 
                                net=net, sigma=self.sigma(t_prime), 
                                inference=True, cond_scale=self.cond_scale, 
                                x_mask=x_mask, **kwargs)
            d_prime = self.sigma_deriv(t_prime) / self.sigma(t_prime) * x_next - self.sigma_deriv(t_prime) / self.sigma(t_prime) * denoised_prime
            x_next = x_hat + 1/2 * h * (d + d_prime)
        
        return x_next
    
    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t
                use_heun: bool=True,
                **kwargs) -> Tensor:
        
        # here sigmas means t
        # input parameters have to be consistent with the above
        
        t_steps = sigmas
        sigma_steps = self.sigma(t_steps)
        
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )
        
        # Denoise to sample
        x = noise * self.sigma(t_steps[0])
        for i in range(self.num_steps - 1):
            x = self.step(x, x_classes, 
                          fn=fn, 
                          net=net, 
                          gamma=gammas[i],
                          t=t_steps[i], 
                          t_next=t_steps[i+1],
                          use_heun=use_heun, 
                          **kwargs)

        return x.clamp(-1.0, 1.0)
    
class VPSampler(nn.Module):
    """ 
    EDM version (https://arxiv.org/abs/2206.00364) VP sampler in Algorithm 1:
    """
    def __init__(
        self,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        epsilon_s: float = 1e-3,
        s_churn: float = 200.0,
        s_noise: float = 1.0,
        s_min: float = 0.0,
        s_max: float = float('inf'),
        num_steps: int = 200,
        cond_scale: float = 1.0,
    ):
        super().__init__()
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_s = epsilon_s
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        
        ## Sampler helper function
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        
        sigma = vp_sigma(self.beta_d, self.beta_min)
        sigma_deriv = vp_sigma_deriv(self.beta_d, self.beta_min)
        sigma_inv = vp_sigma_inv(self.beta_d, self.beta_min)
        
        # Define scaling schedule.
        scale = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        scale_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (scale(t) ** 3)
        
        self.scale = scale
        self.scale_deriv = scale_deriv
        self.vp_sigma = vp_sigma
        self.sigma_inv = sigma_inv
        self.sigma_deriv = sigma_deriv
        self.sigma = sigma
        
    def step(self, x: Tensor, 
             x_classes: Tensor, 
             fn: Callable, net: nn.Module, 
             t: float, t_next: float, 
             gamma: float, x_mask: Tensor=None, 
             use_heun: bool=True, **kwargs) -> Tensor:

        # Increase noise temporarily.
        t_hat = self.sigma_inv((self.sigma(t) + gamma * self.sigma(t)))
        x_hat = self.scale(t_hat) / self.scale(t) * x + (self.sigma(t_hat)**2-self.sigma(t)**2).clip(min=0).sqrt() * self.scale(t_hat) * self.s_noise * torch.randn_like(x)

        # Euler step.
        denoised_cur = fn(x_hat / self.scale(t_hat), x_classes, 
                          net=net, sigma=self.sigma(t_hat), 
                          inference=True, cond_scale=self.cond_scale, 
                          x_mask=x_mask, **kwargs)
        
        d = (self.sigma_deriv(t_hat)/self.sigma(t_hat) + self.scale_deriv(t_hat)/self.scale(t_hat))*x_hat - self.sigma_deriv(t_hat) * self.scale(t_hat) / self.sigma(t_hat) * denoised_cur
        h = t_next - t_hat
        x_next = x_hat + h * d
        
        # Apply 2nd order correction.
        if t_next != 0 and use_heun:
            t_prime = t_hat + h
            
            denoised_prime = fn(x_next / self.scale(t_prime), x_classes, 
                                net=net, sigma=self.sigma(t_prime), 
                                inference=True, cond_scale=self.cond_scale, 
                                x_mask=x_mask, **kwargs)
            d_prime = (self.sigma_deriv(t_prime) / self.sigma(t_prime) + self.scale_deriv(t_prime) / self.scale(t_prime)) * x_next - self.sigma_deriv(t_prime) * self.scale(t_prime) / self.sigma(t_prime) * denoised_prime
            x_next = x_hat + 1/2 * h * (d + d_prime)
        
        return x_next
        
    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t
                x_mask: Tensor=None, 
                use_heun: bool=True,
                **kwargs) -> Tensor:
        
        t_steps = sigmas
        sigma_steps = self.vp_sigma(self.beta_d, self.beta_min)(t_steps)
        
        # Compute gammas
        gammas = torch.where(
            (self.sigma(t_steps) >= self.s_min) & (self.sigma(t_steps) <= self.s_max),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )
        
        x = noise * self.sigma(t_steps[0]) * self.scale(t_steps[0])
        for i in range(self.num_steps - 1):
            
            x = self.step(x, x_classes, 
                          fn=fn, net=net, 
                          t=t_steps[i], 
                          t_next=t_steps[i+1], 
                          gamma=gammas[i], 
                          use_heun=use_heun, 
                          **kwargs)

        return x

class EDMAlphaSampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) deterministic sampler Algo 3:
    when setting Heun=False, it amounts to DDIM (Euler)
    with Heun's sampler, the NFE is around the twice the number of steps. 
    
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_steps: int = 50,
        cond_scale: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps
        self.cond_scale = cond_scale

    def step(self, x: Tensor, 
             x_classes: Tensor, 
             fn: Callable, net: nn.Module, 
             sigma: float, sigma_next: float, 
             x_mask: Tensor=None, 
             use_heun: bool=True, **kwargs) -> Tensor:
        
        """One step of EDM alpha sampler"""
        # Select temporarily increased noise level
        h = sigma_next - sigma

        # Evaluate ∂x/∂sigma at sigma_hat
        denoised_cur =  fn(x, x_classes, net=net, 
                           sigma=sigma, inference=True, 
                           cond_scale=self.cond_scale, 
                           x_mask=x_mask, **kwargs)
        d = (x - denoised_cur) / sigma

        sigma_p = sigma + self.alpha * h
        if sigma_p != 0 and use_heun:
            # Second order correction
            x_p = x + self.alpha * h * d
            denoised_p = fn(x_p, x_classes, 
                            net=net, sigma=sigma_p, 
                            inference=True, 
                            cond_scale=self.cond_scale, 
                            x_mask=x_mask, **kwargs)
            d_p = (x_p - denoised_p) / sigma_p
            x_next = x + h * ((1 - 0.5/self.alpha) * d + 0.5/self.alpha * d_p)
        else:
            x_next = x + h * d
            
        return x_next
    
    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, 
                use_heun: bool=True,
                **kwargs) -> Tensor:
        
        # pay attention to this step
        x = sigmas[0] * noise

        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, x_classes, 
                          fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          use_heun=use_heun, 
                          **kwargs)
            
        return x

class EDMSampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) stochastic sampler:
    s_churn=40 s_noise=1.003, s_tmin=0.05, s_tmax=50 
    when setting s_churn = 0, it amounts to DDIM

    with Heun's sampler, the NFE is around the twice the number of steps. 
    
    """

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
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
    
    def forward(self, noise: Tensor, 
                x_classes: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, 
                use_heun: bool=False,
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
        for i in range(self.num_steps - 1):
            x = self.step(x, x_classes, 
                          fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          gamma=gammas[i], 
                          use_heun=use_heun, 
                          **kwargs)
            
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