import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import os
from src.encoder import Encoder
from src.data import get_cifar10_loaders

def analyze_noise_schedule(ckpt_path, config_path):
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint NOT FOUND at: {os.path.abspath(ckpt_path)}")
        print("Please check your 'outputs/' folder in Colab.")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init encoder
    encoder = Encoder(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['encoder_channels'],
        sigma_0=config['noise']['sigma_0']
    ).to(device)
    
    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    
    # Get test data
    _, val_loader = get_cifar10_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=128
    )
    
    lambdas = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            _, _, l0 = encoder(x, add_noise=False, use_learned_schedule=True)
            lambdas.append(l0.cpu())
    
    lambdas = torch.cat(lambdas).numpy()
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lambdas, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Learned Noise Levels (λ₀)")
    plt.xlabel("log-SNR (λ₀) [Higher = Less Noise/Compression]")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("outputs/lambda_dist.png")
    print(f"Mean λ₀: {lambdas.mean():.4f}")
    print(f"Std λ₀: {lambdas.std():.4f}")
    print("Saved histogram to outputs/lambda_dist.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--config', default='configs/config_cifar10.yaml', help='Path to config file')
    args = parser.parse_args()

    analyze_noise_schedule(args.ckpt, args.config)
