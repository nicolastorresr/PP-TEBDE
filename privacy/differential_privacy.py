import numpy as np
import torch
from torch.optim import Optimizer
from typing import List, Optional

class DifferentialPrivacy:
    def __init__(self, config):
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.clip_norm = config.clip_norm
        self.noise_multiplier = config.noise_multiplier

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to a tensor for differential privacy.
        """
        noise = torch.normal(0, self.noise_multiplier * self.clip_norm, tensor.shape, device=tensor.device)
        return tensor + noise

    def clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Clip gradients to ensure bounded sensitivity.
        """
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2) for grad in gradients]), 2)
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        return [grad.detach() * clip_coef_clamped for grad in gradients]

    def compute_privacy_spent(self, steps: int, batch_size: int, data_size: int) -> float:
        """
        Compute the privacy spent using the moments accountant method.
        """
        q = batch_size / data_size
        orders = torch.arange(2, 64).float()
        rdp = self.compute_rdp(q, self.noise_multiplier, steps, orders)
        epsilon = self.get_privacy_spent(orders, rdp, target_delta=self.delta)
        return epsilon

    def compute_rdp(self, q: float, noise_multiplier: float, steps: int, orders: torch.Tensor) -> torch.Tensor:
        """
        Compute Renyi Differential Privacy (RDP) for Gaussian mechanism.
        """
        rdp = orders * torch.log(1 + q**2 / noise_multiplier**2) / 2
        rdp = rdp + orders * (noise_multiplier**2) / (2 * (noise_multiplier**2 + 1))
        return rdp * steps

    def get_privacy_spent(self, orders: torch.Tensor, rdp: torch.Tensor, target_delta: float) -> float:
        """
        Compute epsilon given a target delta and RDP values.
        """
        epsilon = torch.tensor(float('inf'))
        for order, specific_rdp in zip(orders, rdp):
            epsilon = torch.min(epsilon, (specific_rdp - torch.log(target_delta)) / (order - 1))
        return epsilon.item()

class DPOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, dp: DifferentialPrivacy):
        self.optimizer = optimizer
        self.dp = dp

    def step(self, closure: Optional[callable] = None):
        """
        Perform a single optimization step with differential privacy.
        """
        # Get gradients
        gradients = [p.grad.data for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]

        # Clip gradients
        clipped_gradients = self.dp.clip_gradients(gradients)

        # Add noise to gradients
        noisy_gradients = [self.dp.add_noise(grad) for grad in clipped_gradients]

        # Update gradients
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data = noisy_gradients[idx]
                    idx += 1

        # Perform optimization step
        self.optimizer.step(closure)

def apply_dp_to_model(model: torch.nn.Module, dp: DifferentialPrivacy) -> torch.nn.Module:
    """
    Apply differential privacy to a model's parameters.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.add_(dp.add_noise(torch.zeros_like(param)))
    return model

def privatize_dataset(data: torch.Tensor, dp: DifferentialPrivacy) -> torch.Tensor:
    """
    Apply differential privacy to a dataset.
    """
    return dp.add_noise(data)
