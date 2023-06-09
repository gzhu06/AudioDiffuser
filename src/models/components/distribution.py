import torch
from math import atan, pi
from torch import Tensor
import math

class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()

class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmin - vmax) * torch.rand(num_samples, device=device) + vmax
    
class LogUniformDistribution(Distribution):
    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 100):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        
        rnd_uniform = torch.rand(num_samples, device=device)
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
