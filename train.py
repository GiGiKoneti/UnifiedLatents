# train.py
import torch
import yaml
import argparse
import os
from tqdm import tqdm
from torchvision.utils import save_image
from src.encoder import Encoder
from src.diffusion_prior import DiffusionPrior
from src.diffusion_decoder import DiffusionDecoder
from src.losses import prior_loss, prior_loss_weighted, decoder_loss
from src.data import get_cifar10_loaders
from src.sampler import DDPMSampler

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse)) # Assuming range [-1, 1] -> 2.0 span

def train(config: dict, stage: int = 1, smoke_test: bool = False):
    device = torch.device(config['training']['device'])
    if device.type == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        device = torch.device('cpu')
    
    print(f"--- Training Stage {stage} on {device} ---")

    # 1. Build models
    encoder = Encoder(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['encoder_channels'],
        sigma_0=config['noise']['sigma_0']
    ).to(device)
    
    prior = DiffusionPrior(
        latent_dim=config['model']['latent_dim'],
        latent_spatial=config['model']['latent_spatial'],
        d_model=config['model'].get('d_model', 512),
        depth=config['model']['prior_depth'],
        n_heads=config['model']['prior_heads']
    ).to(device)
    
    decoder = DiffusionDecoder(
        image_size=config['model']['image_size'],
        latent_dim=config['model']['latent_dim'],
        latent_spatial=config['model']['latent_spatial'],
        d_model=config['model'].get('d_model', 512),
        depth=config['model']['decoder_depth'],
        n_heads=config['model']['decoder_heads']
    ).to(device)

    # Load weights if provided or if in Stage 2
    if config.get('ckpt_path'):
        print(f"Loading weights from {config['ckpt_path']}")
        ckpt = torch.load(config['ckpt_path'], map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        prior.load_state_dict(ckpt['prior'])
        decoder.load_state_dict(ckpt['decoder'])
    elif stage == 2:
        default_ckpt = f"{config['logging']['output_dir']}/ckpt_stage1_final.pt"
        if os.path.exists(default_ckpt):
            print(f"Loading default Stage 1 weights from {default_ckpt}")
            ckpt = torch.load(default_ckpt, map_location=device)
            encoder.load_state_dict(ckpt['encoder'])
            prior.load_state_dict(ckpt['prior'])
            decoder.load_state_dict(ckpt['decoder'])
        else:
            print("Warning: Stage 1 checkpoint not found, starting Stage 2 from scratch.")

    # 2. Optimizer
    if stage == 1:
        # Joint training
        params = list(encoder.parameters()) + list(prior.parameters()) + list(decoder.parameters())
    else:
        # Stage 2: Retrain prior only (or refine decoder)
        # As per paper: retrain prior with sigmoid weighting
        encoder.eval() 
        for p in encoder.parameters(): p.requires_grad = False
        params = list(prior.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])
    )

    # 3. Data
    train_loader, val_loader = get_cifar10_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    # 4. Samplers for visualization
    prior_sampler = DDPMSampler(prior, config['noise']['min_log_snr'], config['noise']['max_log_snr'])
    decoder_sampler = DDPMSampler(decoder, config['noise']['min_log_snr'], config['noise']['max_log_snr'])

    # 5. Training loop
    min_log_snr = config['noise']['min_log_snr']
    max_log_snr = config['noise']['max_log_snr']
    n_epochs = config['training'][f'stage{stage}_epochs'] if not smoke_test else 1

    for epoch in range(n_epochs):
        encoder.train() if stage == 1 else encoder.eval()
        prior.train()
        decoder.train()
        
        pbar = tqdm(train_loader, desc=f"Stage {stage} Ep {epoch}")
        for step, (x, _) in enumerate(pbar):
            x = x.to(device)
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(stage == 1):
                use_learned = config['training'].get('use_learned_schedule', False)
                z_0, z_clean, lambda_0 = encoder(x, use_learned_schedule=use_learned)
            
            if stage == 1:
                p_loss = prior_loss(prior, z_clean, min_log_snr, max_log_snr, lambda_0=lambda_0)
            else:
                p_loss = prior_loss_weighted(prior, z_clean, min_log_snr, max_log_snr, lambda_0=lambda_0)
                
            d_loss = decoder_loss(decoder, x, z_0, min_log_snr, max_log_snr, lambda_0=lambda_0)
            loss = p_loss + d_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config['training']['grad_clip'])
            optimizer.step()

            if step % config['logging']['log_every'] == 0:
                pbar.set_postfix({"p_loss": f"{p_loss.item():.4f}", "d_loss": f"{d_loss.item():.4f}"})

            # Periodic Sampling / Visualization
            if (step == 0 and epoch == 0) or (step % 500 == 0 and not smoke_test):
                visualize_reconstruction(decoder_sampler, x[:4], z_0[:4], epoch, step, config['logging']['output_dir'])

            if smoke_test and step >= 3:
                break

        # Save checkpoint
        ckpt_path = f"{config['logging']['output_dir']}/ckpt_stage{stage}_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'prior':   prior.state_dict(),
            'decoder': decoder.state_dict(),
        }, ckpt_path)
        
    # Final save
    torch.save({
        'encoder': encoder.state_dict(),
        'prior': prior.state_dict(),
        'decoder': decoder.state_dict(),
    }, f"{config['logging']['output_dir']}/ckpt_stage{stage}_final.pt")

def visualize_reconstruction(sampler, x_orig, z_0, epoch, step, output_dir):
    print(f"\nGenerating visualization at Epoch {epoch} Step {step}...")
    # Sample from decoder given latent z_0
    # x_t shape matches x_orig
    device = x_orig.device
    x_rec = sampler.sample(x_orig.shape, steps=20, conditioning=z_0, device=device)
    
    # Calculate PSNR
    psnr = calculate_psnr(x_orig, x_rec)
    print(f"Reconstruction PSNR: {psnr.item():.2f} dB")
    
    # Save a comparison grid
    # Denormalize from [-1, 1] to [0, 1]
    comparison = torch.cat([x_orig, x_rec], dim=0)
    comparison = (comparison + 1) / 2
    save_path = f"{output_dir}/recon_ep{epoch}_step{step}.png"
    save_image(comparison, save_path, nrow=4)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_cifar10.yaml')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--use-learned-schedule', action='store_true', help='Extension: Use trainable lambda_0')
    parser.add_argument('--ckpt', type=str, help='Path to load checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.use_learned_schedule:
        config['training']['use_learned_schedule'] = True
    if args.ckpt:
        config['ckpt_path'] = args.ckpt

    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    train(config, stage=args.stage, smoke_test=args.smoke_test)
