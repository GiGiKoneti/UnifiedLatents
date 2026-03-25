# src/sampler.py
import torch
from src.noise_schedule import log_snr, log_snr_to_alpha_sigma

class DDPMSampler:
    def __init__(self, model, min_log_snr=-10.0, max_log_snr=10.0):
        self.model = model
        self.min_log_snr = min_log_snr
        self.max_log_snr = max_log_snr

    @torch.no_grad()
    def step(self, x_t, t, t_prev, conditioning=None, max_log_snr_override=None):
        """
        One step of DDPM denoising.
        """
        device = x_t.device
        b = x_t.shape[0]
        
        # 1. Model prediction (eps_hat)
        t_batch = torch.full((b,), t, device=device)
        if conditioning is not None:
            eps_hat = self.model(x_t, t_batch, conditioning)
        else:
            eps_hat = self.model(x_t, t_batch)
            
        # 2. Get alpha/sigma for current and previous using the SAME schedule as training
        log_snr_t = log_snr(torch.tensor([t], device=device), self.min_log_snr, self.max_log_snr, max_log_snr_override)
        alpha_t, sigma_t = log_snr_to_alpha_sigma(log_snr_t)
        
        log_snr_prev = log_snr(torch.tensor([t_prev], device=device), self.min_log_snr, self.max_log_snr, max_log_snr_override)
        alpha_prev, sigma_prev = log_snr_to_alpha_sigma(log_snr_prev)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, *([1] * (x_t.ndim - 1)))
        sigma_t = sigma_t.view(-1, *([1] * (x_t.ndim - 1)))
        alpha_prev = alpha_prev.view(-1, *([1] * (x_t.ndim - 1)))
        sigma_prev = sigma_prev.view(-1, *([1] * (x_t.ndim - 1)))

        # 3. Predict x_0 (clean)
        x_0_hat = (x_t - sigma_t * eps_hat) / alpha_t
        
        # 4. Posterior mean direction (DDPM)
        x_prev = alpha_prev * x_0_hat + sigma_prev * eps_hat
        
        return x_prev

    @torch.no_grad()
    def sample(self, shape, steps=50, conditioning=None, device='cpu', max_log_snr_override=None):
        """
        Full sampling loop from T=1 to 0.
        """
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1)
        
        for i in range(steps):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item()
            x = self.step(x, t, t_prev, conditioning, max_log_snr_override=max_log_snr_override)
            
        return x
