import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Pre-activation ResBlock with GroupNorm.
    If in_channels != out_channels OR stride != 1, use a 1x1 conv shortcut.
    Architecture:
        GroupNorm(8, in_ch) → SiLU → Conv2d(in_ch, out_ch, 3, stride, pad=1)
        → GroupNorm(8, out_ch) → SiLU → Conv2d(out_ch, out_ch, 3, 1, pad=1)
        + shortcut
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class NoiseSchedulePredictor(nn.Module):
    """
    Predicts optimal lambda(0) for each sample.
    Input: features from the final resblock (B, 256, 4, 4)
    Output: lambda_0 (B, 1) in range [2, 8]
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        h = self.avg_pool(x).flatten(1)
        h = self.net(h)
        # Constrain to [2, 8] and flatten to (B,) for broadcasting
        return (2.0 + 6.0 * torch.sigmoid(h)).view(-1)


class Encoder(nn.Module):
    """
    Maps image (B,3,H,W) → clean latent f_enc(x): (B, latent_dim, H//8, W//8)
    Then adds Gaussian noise to get z_0.
    Noise can be fixed (sigma_0) or learned (lambda_0 predictor).
    """
    def __init__(self, latent_dim=128, channels=[64, 128, 256], sigma_0=0.1):
        super().__init__()
        self.sigma_0 = float(sigma_0)
        
        # Stem: 3 -> 64
        self.stem = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        
        # ResBlocks
        self.res1 = ResBlock(channels[0], channels[1], stride=2)  # 32x32 -> 16x16
        self.res2 = ResBlock(channels[1], channels[2], stride=2)  # 16x16 -> 8x8
        self.res3 = ResBlock(channels[2], channels[2], stride=2)  # 8x8 -> 4x4
        
        # Extension: Learned Noise Schedule
        self.noise_predictor = NoiseSchedulePredictor(channels[2])
        
        # Final output layers
        self.final_norm = nn.GroupNorm(8, channels[2])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels[2], latent_dim, kernel_size=1, bias=True)

    def forward(self, x, add_noise=True, use_learned_schedule=False):
        """
        Returns:
            z_0:      (B, latent_dim, 4, 4) — noisy latent
            z_clean:  (B, latent_dim, 4, 4) — clean latent
            lambda_0: (B, 1) or None — predicted noise level
        """
        # Stem
        h = self.stem(x)
        
        # ResBlocks
        h = self.res1(h)
        h = self.res2(h)
        features = self.res3(h)
        
        # Final layers
        z_clean = self.final_conv(self.final_act(self.final_norm(features)))
        
        if use_learned_schedule:
            lambda_0 = self.noise_predictor(features) # (B, 1)
            # lambda = log(alpha^2 / sigma^2). 
            # sigma^2 = sigmoid(-lambda)
            sigma_learned = torch.sqrt(torch.sigmoid(-lambda_0))
            # Reshape for broadcasting (B, 1, 1, 1)
            sigma_val = sigma_learned.view(-1, 1, 1, 1)
        else:
            lambda_0 = None
            sigma_val = self.sigma_0
            
        if add_noise:
            z_0 = z_clean + sigma_val * torch.randn_like(z_clean)
        else:
            z_0 = z_clean
            
        return z_0, z_clean, lambda_0
