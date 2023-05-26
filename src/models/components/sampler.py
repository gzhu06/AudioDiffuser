from math import sqrt, cos, sin, pi
from typing import Callable, Tuple, Optional
from einops import repeat
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from .utils import extend_dim

""" Samplers """

class VanillaSampler(nn.Module):
    """DDPM: original formulation"""

    def get_sigmas(self, beta):
    
        alpha = 1 - beta
        alpha_bar = alpha + 0
        beta_tilde = beta + 0
        for t in range(1, len(beta)):
            alpha_bar[t] *= alpha_bar[t-1]
            beta_tilde[t] *= (1-alpha_bar[t-1])/(1-alpha_bar[t]) 
            
        sigma = torch.sqrt(beta_tilde)
        return alpha, alpha_bar, sigma

    @torch.no_grad()
    def forward(self, noise: Tensor, fn: Callable, beta: Tensor) -> Tensor:
        
        # unconditional generation
        alpha, alpha_bar, sigma = self.get_sigmas(beta)

        # audio initialization
        audio = noise # initial noise
        for t in tqdm(range(len(alpha_bar)-1, -1, -1)):
            epsilon_theta = fn(audio, torch.tensor([t], device=audio.device)).squeeze(1)
            audio = (audio-(1-alpha[t])/torch.sqrt(1-alpha_bar[t])*epsilon_theta)/torch.sqrt(alpha[t])

            if t > 0:
                noise_rand = torch.randn_like(audio)
                audio = audio + sigma[t] * noise_rand
        return audio
    
class DDIMSampler(nn.Module):

    """
    DDIM (https://arxiv.org/abs/2010.02502) sampler: eq(12)
    WIP
    """ 

    def get_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        alpha = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return alpha

    @torch.no_grad()
    def step(self, x: Tensor, fn: Callable):

        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et = fn(xt, t)

        # predict x0
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to('cpu'))
        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.to('cpu'))

        return x_next
    
    @torch.no_grad()
    def forward(self, noise: Tensor, 
                x_classes: Tensor,                
                net: nn.Module, 
                **kwargs):

        batch_size = noise.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x = noise
        for i, j in zip(reversed(seq), reversed(seq_next)):

            x_next = self.step(x, fn)


        return xs, x0_preds

class EDMVanillaSampler(nn.Module):
    """DDPM: EDM (https://arxiv.org/abs/2206.00364) formulation, Table 1"""
    pass

class VSampler(nn.Module):

    # diffusion_types = [VDiffusion]

    def __init__(self, num_steps: int = 50, cond_scale: float=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = cond_scale

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, 
                x_noisy: Tensor,
                x_classes: Tensor, 
                net: nn.Module, 
                sigmas: Tensor, 
                x_mask: Tensor=None, 
                inference: bool=True,
                **kwargs) -> Tensor:
        
        if x_mask is None:
            x_mask = torch.ones_like(x_noisy)
        
        x_noisy = x_mask * x_noisy
        b = x_noisy.shape[0]
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        for i in range(self.num_steps):

            v_pred = net(x_noisy, sigmas[i], x_classes, x_mask=x_mask, **kwargs)
            if inference and self.cond_scale != 1.0:
                null_logits = net(x_noisy, sigmas[i], x_classes, 
                                  cond_drop_prob=1., x_mask=x_mask, **kwargs)
                v_pred = null_logits + (v_pred - null_logits) * self.cond_scale
                
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x_noisy

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

class ARVSampler(nn.Module):
    def __init__(self, net: nn.Module, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = net

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
        self, current: Tensor, sigmas: Tensor, show_progress: bool = False, **kwargs
    ) -> Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            channels = torch.cat([current, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return current

    def sample_start(self, num_items: int, num_steps: int, **kwargs) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Same sigma schedule over all chunks
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    def forward(
        self,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        # Return start if only num_splits chunks
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))

        # Loop over ladder shifts
        num_shifts = num_chunks  # - self.num_splits
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)

        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs
            )
            # Update chunks
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            shape = (b, self.in_channels, self.split_length)
            chunks += [torch.randn(shape, device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)