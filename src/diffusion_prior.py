import torch
import torch.nn as nn
import math
from einops import rearrange

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) float tensor in [0, 1]
    Returns: (B, dim) sinusoidal embedding
    """
    t = t.float().reshape(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :] * 1000  # scale t to [0, 1000]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

class TimestepMLP(nn.Module):
    """Sinusoidal embedding → Linear(dim, 4*dim) → GELU → Linear(4*dim, d_model)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, d_model) from sinusoidal_embedding
        return self.net(t)

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.
    Attention: nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    MLP: d_model → 4*d_model → d_model, GELU activation
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # Attention path
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        
        # MLP path
        x = x + self.mlp(self.norm2(x))
        return x

class DiffusionPrior(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        latent_spatial: int = 4,
        d_model: int = 512,
        depth: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_spatial = latent_spatial
        self.n_tokens = latent_spatial ** 2
        
        # Latent projection
        self.proj_in = nn.Linear(latent_dim, d_model)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        
        # Timestep embedding
        self.time_mlp = TimestepMLP(d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(depth)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, conditioning=None) -> torch.Tensor:
        """
        z_t: (B, latent_dim, H, W)
        t:   (B,) float in [0, 1]
        conditioning: ignored (included for sampler compatibility)
        returns eps_hat: (B, latent_dim, H, W)
        """
        """
        z_t: (B, latent_dim, H, W)
        t:   (B,) float in [0, 1]
        returns eps_hat: (B, latent_dim, H, W)
        """
        B, C, H, W = z_t.shape
        
        # 1. Flatten spatial dimensions and move channel to end
        # (B, C, H, W) -> (B, H*W, C)
        x = rearrange(z_t, 'b c h w -> b (h w) c')
        
        # 2. Linear projection to d_model
        x = self.proj_in(x)
        
        # 3. Add positional embedding
        x = x + self.pos_embed
        
        # 4. Add timestep embedding to all tokens
        # sinusoidal_embedding returns (B, d_model)
        t_embed = sinusoidal_embedding(t, x.shape[-1])
        t_feat = self.time_mlp(t_embed) # (B, d_model)
        x = x + t_feat.unsqueeze(1) # Broadcast over sequence length (16)
        
        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # 6. Final projection back to latent_dim
        x = self.proj_out(self.final_norm(x))
        
        # 7. Reshape back to original z_t shape
        # (B, H*W, latent_dim) -> (B, latent_dim, H, W)
        eps_hat = rearrange(x, 'b (h w) c -> b c h w', h=self.latent_spatial, w=self.latent_spatial)
        
        return eps_hat
