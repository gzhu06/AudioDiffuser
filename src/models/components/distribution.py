import torch
from math import atan, pi
from torch import Tensor

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
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class VKDistribution(Distribution):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = float("inf"),
        sigma_data: float = 1.0,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        sigma_data = self.sigma_data
        min_cdf = atan(self.min_value / sigma_data) * 2 / pi
        max_cdf = atan(self.max_value / sigma_data) * 2 / pi
        u = (max_cdf - min_cdf) * torch.randn((num_samples,), device=device) + min_cdf
        return torch.tan(u * pi / 2) * sigma_data
