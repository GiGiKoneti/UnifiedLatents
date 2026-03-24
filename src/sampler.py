# src/sampler.py
import torch
from src.noise_schedule import log_snr, log_snr_to_alpha_sigma

class DDPMSampler:
    def __init__(self, model, min_log_snr=-10.0, max_log_snr=10.0):
        self.model = model
        self.min_log_snr = min_log_snr
        self.max_log_snr = max_log_snr

    @torch.no_grad()
    def step(self, x_t, t, t_prev, conditioning=None):
        """
        One step of DDPM denoising.
        x_t: (B, C, H, W) or (B, N, D)
        t: current timestep (0 to 1)
        t_prev: previous timestep (closer to 0)
        """
        device = x_t.device
        b = x_t.shape[0]
        
        # 1. Model prediction (eps_hat)
        t_batch = torch.full((b,), t, device=device)
        if conditioning is not None:
            eps_hat = self.model(x_t, t_batch, conditioning)
        else:
            eps_hat = self.model(x_t, t_batch)
            
        # 2. Get alpha/sigma for current and previous
        alpha_t, sigma_t = log_snr_to_alpha_sigma(log_snr(torch.tensor([t], device=device), self.min_log_snr, self.max_log_snr))
        alpha_prev, sigma_prev = log_snr_to_alpha_sigma(log_snr(torch.tensor([t_prev], device=device), self.min_log_snr, self.max_log_snr))
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, *([1] * (x_t.ndim - 1)))
        sigma_t = sigma_t.view(-1, *([1] * (x_t.ndim - 1)))
        alpha_prev = alpha_prev.view(-1, *([1] * (x_t.ndim - 1)))
        sigma_prev = sigma_prev.view(-1, *([1] * (x_t.ndim - 1)))

        # 3. Predict x_0 (clean)
        # x_t = alpha_t * x_0 + sigma_t * eps
        x_0_hat = (x_t - sigma_t * eps_hat) / alpha_t
        x_0_hat = x_0_hat.clamp(-1, 1) # Optional but common for images
        
        # 4. Posterior mean direction (DDPM)
        x_prev = alpha_prev * x_0_hat + sigma_prev * eps_hat
        
        return x_prev

    @torch.no_grad()
    def sample(self, shape, steps=50, conditioning=None, device='cpu'):
        """
        Full sampling loop from T=1 to 0.
        """
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1)
        
        for i in range(steps):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item()
            x = self.step(x, t, t_prev, conditioning)
            
        return x
