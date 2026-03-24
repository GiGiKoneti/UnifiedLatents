# src/losses.py
import torch
import torch.nn.functional as F
from src.noise_schedule import log_snr, add_noise, sample_timesteps

def prior_loss(
    prior,
    z_clean: torch.Tensor,
    min_log_snr: float,
    max_log_snr: float,
    lambda_0: torch.Tensor = None
) -> torch.Tensor:
    """
    Unweighted diffusion loss for the prior (Stage 1).
    """
    device = z_clean.device
    batch_size = z_clean.shape[0]
    
    # 1. Sample t
    t = sample_timesteps(batch_size, device)
    
    # 2. Add noise to z_clean (with optional lambda_0 override)
    z_t, eps, _, _ = add_noise(z_clean, t, min_log_snr, max_log_snr, max_log_snr_override=lambda_0)
    
    # 3. Predict eps_hat
    eps_hat = prior(z_t, t)
    
    # 4. Return MSE
    return F.mse_loss(eps_hat, eps)

def prior_loss_weighted(
    prior,
    z_clean: torch.Tensor,
    min_log_snr: float,
    max_log_snr: float,
    lambda_0: torch.Tensor = None
) -> torch.Tensor:
    """
    Sigmoid-weighted diffusion loss for the prior (Stage 2).
    """
    device = z_clean.device
    batch_size = z_clean.shape[0]
    
    t = sample_timesteps(batch_size, device)
    z_t, eps, _, _ = add_noise(z_clean, t, min_log_snr, max_log_snr, max_log_snr_override=lambda_0)
    eps_hat = prior(z_t, t)
    
    # Use the sample-specific lambda_0 for weighting as well
    lambda_t = log_snr(t, min_log_snr, max_log_snr, max_log_snr_override=lambda_0)
    weight = torch.sigmoid(lambda_t)
    
    # Reshape weight for broadcasting
    while weight.ndim < eps.ndim:
        weight = weight.unsqueeze(-1)
    
    mse_per_element = (eps_hat - eps) ** 2
    return (weight * mse_per_element).mean()

def decoder_loss(
    decoder,
    x: torch.Tensor,
    z_0: torch.Tensor,
    min_log_snr: float,
    max_log_snr: float,
    lambda_0: torch.Tensor = None
) -> torch.Tensor:
    """
    Sigmoid-weighted diffusion loss for the decoder.
    """
    device = x.device
    batch_size = x.shape[0]
    
    # 1. Sample t
    t = sample_timesteps(batch_size, device)
    
    # 2. Add noise to x (with optional lambda_0 override)
    x_t, eps, _, _ = add_noise(x, t, min_log_snr, max_log_snr, max_log_snr_override=lambda_0)
    
    # 3. Predict eps_hat
    eps_hat = decoder(x_t, t, z_0)
    
    # 4. Compute lambda_t and weight
    lambda_t = log_snr(t, min_log_snr, max_log_snr, max_log_snr_override=lambda_0)
    weight = torch.sigmoid(lambda_t)
    
    # Reshape weight for broadcasting
    while weight.ndim < eps.ndim:
        weight = weight.unsqueeze(-1)
    
    # 5. Weighted MSE
    mse_per_element = (eps_hat - eps) ** 2
    weighted_mse = weight * mse_per_element
    
    # 6. Return mean
    return weighted_mse.mean()
