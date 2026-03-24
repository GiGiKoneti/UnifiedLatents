import torch
import torch.nn as nn
from einops import rearrange
from src.diffusion_prior import sinusoidal_embedding, TimestepMLP

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B, C, H, W)
    returns: (B, n_patches, C*patch_size*patch_size)
    """
    return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size)

def unpatchify(x: torch.Tensor, patch_size: int, H: int, W: int, C: int) -> torch.Tensor:
    """
    x: (B, n_patches, patch_dim)
    returns: (B, C, H, W)
    """
    h, w = H // patch_size, W // patch_size
    return rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=h, w=w, p1=patch_size, p2=patch_size, c=C)

class CrossAttentionBlock(nn.Module):
    """
    Pre-norm transformer block with BOTH self-attention and cross-attention.
    Order:
        LayerNorm → SelfAttention(x, x) → residual
        LayerNorm → CrossAttention(x, z_0_tokens) → residual
        LayerNorm → MLP → residual
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, z_0_tokens: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # z_0_tokens: (B, 16, d_model)
        
        # 1. Self-attention
        x_norm = self.norm1(x)
        h, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + h
        
        # 2. Cross-attention
        x_norm = self.norm2(x)
        h, _ = self.cross_attn(x_norm, z_0_tokens, z_0_tokens)
        x = x + h
        
        # 3. MLP
        x = x + self.mlp(self.norm3(x))
        return x

class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        latent_dim: int = 128,
        latent_spatial: int = 4,
        d_model: int = 512,
        depth: int = 6,          # must be even
        n_heads: int = 8,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch and Latent projection
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Positional embedding (for patches)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        # Timestep embedding
        self.time_mlp = TimestepMLP(d_model)
        
        # Encoder, Middle, and DecoderBlocks
        num_encoder_blocks = depth // 2
        self.encoder_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads) for _ in range(num_encoder_blocks)
        ])
        
        self.middle_block = CrossAttentionBlock(d_model, n_heads)
        
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads) for _ in range(num_encoder_blocks)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model, self.patch_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 3, 32, 32)
        t:   (B,)
        z_0: (B, 128, 4, 4)
        """
        # 1. Patchify and project x_t
        x = patchify(x_t, self.patch_size)
        x = self.patch_embed(x) # (B, 64, 512)
        
        # 2. Add positional and timestep embeddings
        x = x + self.pos_embed
        t_embed = sinusoidal_embedding(t, self.d_model)
        t_feat = self.time_mlp(t_embed)
        x = x + t_feat.unsqueeze(1)
        
        # 3. Project z_0 to tokens
        z_tokens = rearrange(z_0, 'b c h w -> b (h w) c')
        z_tokens = self.latent_proj(z_tokens) # (B, 16, 512)
        
        # 4. Encoder pass (with skip connections)
        skips = []
        for block in self.encoder_blocks:
            x = block(x, z_tokens)
            skips.append(x)
            
        # 5. Middle block
        x = self.middle_block(x, z_tokens)
        
        # 6. Decoder pass (with skip connections)
        for i, block in enumerate(self.decoder_blocks):
            # Skip connection is from last encoder to first decoder
            x = x + skips[-(i+1)]
            x = block(x, z_tokens)
            
        # 7. Final projection and unpatchify
        x = self.final_proj(self.final_norm(x))
        eps_hat = unpatchify(x, self.patch_size, self.image_size, self.image_size, self.in_channels)
        
        return eps_hat
