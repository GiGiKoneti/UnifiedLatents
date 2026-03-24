import torch

def log_snr(
    t: torch.Tensor, 
    min_log_snr: float, 
    max_log_snr: float, 
    max_log_snr_override: torch.Tensor = None
) -> torch.Tensor:
    """
    Linear interpolation of log-SNR between min and max.
    lambda(t) = max_log_snr + (min_log_snr - max_log_snr) * t
    t=0 => max_log_snr (clean), t=1 => min_log_snr (pure noise)
    
    If max_log_snr_override is provided, it is used instead of the fixed max_log_snr.
    """
    max_val = max_log_snr if max_log_snr_override is None else max_log_snr_override
    return max_val + (min_log_snr - max_val) * t

def log_snr_to_alpha_sigma(log_snr_values: torch.Tensor):
    """
    From log-SNR lambda, recover alpha and sigma such that:
    alpha^2 = sigmoid(lambda)
    sigma^2 = sigmoid(-lambda)
    alpha^2 + sigma^2 = 1 (variance preserving)
    """
    alpha_sq = torch.sigmoid(log_snr_values)
    sigma_sq = torch.sigmoid(-log_snr_values)
    return torch.sqrt(alpha_sq), torch.sqrt(sigma_sq)

def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform sampling of t in [0, 1]"""
    return torch.rand(batch_size, device=device)

def add_noise(
    x: torch.Tensor, 
    t: torch.Tensor, 
    min_log_snr: float, 
    max_log_snr: float, 
    max_log_snr_override: torch.Tensor = None
):
    """
    Given clean x and timestep t, return:
      x_t = alpha_t * x + sigma_t * eps
      eps ~ N(0, I)
    Returns (x_t, eps, alpha_t, sigma_t)
    """
    log_snr_t = log_snr(t, min_log_snr, max_log_snr, max_log_snr_override)
    alpha_t, sigma_t = log_snr_to_alpha_sigma(log_snr_t)
    
    # Reshape for broadcasting
    while alpha_t.ndim < x.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
    while sigma_t.ndim < x.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
        
    eps = torch.randn_like(x)
    x_t = alpha_t * x + sigma_t * eps
    
    return x_t, eps, alpha_t, sigma_t
