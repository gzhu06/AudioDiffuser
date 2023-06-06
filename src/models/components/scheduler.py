import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class KarrasSchedule(nn.Module):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0, num_steps: int = 50):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

    def forward(self) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (self.num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas

class LinearSchedule(nn.Module):
    def __init__(self, start: float = 1.0, end: float = 0.0, num_steps: int = 50):
        super().__init__()
        self.start, self.end = start, end
        self.num_steps = num_steps

    def forward(self) -> Tensor:
        sigmas = torch.linspace(self.start, self.end, self.num_steps)
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas
    
class GeometricSchedule(nn.Module):
    def __init__(self, sigma_max: float = 100, sigma_min: float = 0.02, num_steps: int = 50):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_steps = num_steps
        
    def forward(self) -> Tensor:
        
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        sigmas = (self.sigma_max**2) * ((self.sigma_min**2/self.sigma_max**2) ** (steps / (self.num_steps-1)))
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        
        return sigmas
