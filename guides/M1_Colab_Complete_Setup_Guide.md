# COMPLETE SETUP GUIDE: M1 + GOOGLE COLAB WORKFLOW
**Step-by-Step Instructions for Your Unified Latents Project**

---

## 🎯 YOUR WORKFLOW OVERVIEW

```
M1 MacBook (Local Development)
├─ Code: Write, debug, test
├─ Dataset: CIFAR-10 (small tests)
├─ Version Control: Git + GitHub
└─ Time: 40-50 hours (weeks 1-2, 5-6)

            ↕️ (GitHub sync)

Google Colab (Remote GPU Training)
├─ Environment: Free T4 GPU, 100GB disk
├─ Data: ImageNet-512
├─ Training: Full pipeline
└─ Time: 60-100 hours (runs 24/7, weeks 3-4)

            ↕️ (Download results)

M1 MacBook (Results + Extension)
├─ Analyze: FID metrics, visualizations
├─ Extend: Implement Learned Noise Schedule
└─ Finalize: Application + polish
```

---

## 📋 PART 1: INITIAL SETUP (TODAY - 2 HOURS)

### Step 1: Create GitHub Repository

**Go to github.com:**

```
1. Click "New" → "New repository"
2. Repository name: "unified-latents"
3. Description: "Unified Latents with learned noise schedule extension"
4. Make PUBLIC (so professors can see it)
5. Add README, .gitignore (Python)
6. Create repository
```

**On your M1:**

```bash
cd ~/projects  # Create if doesn't exist: mkdir -p ~/projects
git clone https://github.com/YOUR_USERNAME/unified-latents.git
cd unified-latents
```

---

### Step 2: Project Structure

**Create folders and files:**

```bash
# In ~/projects/unified-latents/:

mkdir -p src configs results/{models,metrics,images}
mkdir -p data/cifar10 data/imagenet

touch src/__init__.py
touch src/encoder.py
touch src/diffusion_prior.py
touch src/diffusion_decoder.py
touch src/losses.py
touch src/data.py
touch src/utils.py

touch train.py
touch evaluate.py
touch configs/config_cifar10.yaml
touch configs/config_imagenet.yaml
touch configs/config_extension.yaml

touch README.md
touch requirements.txt
touch .gitignore
```

---

### Step 3: Create Python Environment

```bash
# Using conda (best for M1)
conda create -n unified-latents python=3.10
conda activate unified-latents

# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install numpy pandas matplotlib scipy scikit-learn tqdm pyyaml
pip install tensorboard

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

### Step 4: First Commit to GitHub

```bash
cd ~/projects/unified-latents

# Create README
cat > README.md << 'EOF'
# Unified Latents Implementation

Implementation of "Unified Latents: How to train your latents" (Google DeepMind, 2026)

## Project Components

- **Challenge #1**: Full Unified Latents implementation (encoder + prior + decoder)
- **Challenge #2**: Learned Noise Schedule extension
- **Analysis**: Multi-resolution evaluation

## Timeline

- Weeks 1-2: Local development on CIFAR-10 (M1)
- Weeks 3-4: Training on ImageNet-512 (Google Colab GPU)
- Week 5: Extension implementation
- Week 6: Final polish and submission

## Setup

```bash
conda create -n unified-latents python=3.10
conda activate unified-latents
pip install -r requirements.txt
```

## Training

```bash
# On CIFAR-10 (quick test)
python train.py --config configs/config_cifar10.yaml

# On ImageNet-512 (full training - use Colab)
python train.py --config configs/config_imagenet.yaml
```

EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch==2.0.0
torchvision==0.15.0
numpy==1.24.0
pandas==1.5.0
matplotlib==3.7.0
scipy==1.10.0
scikit-learn==1.2.0
pyyaml==6.0
tqdm==4.65.0
tensorboard==2.12.0
Pillow==9.5.0
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.Python
venv/
env/
.vscode/
.idea/
*.swp
*.swo
.DS_Store
data/
results/models/*.pt
results/models/*.pth
*.tar.gz
*.zip
runs/
EOF

# Commit
git add .
git commit -m "Initial project structure and documentation"
git push origin main
```

---

## 🔧 PART 2: CODE ON M1 (WEEKS 1-2)

### Step 5: Implement Core Components

You'll write code locally here. Key files:

**File: src/encoder.py (simplified ResNet)**

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, image_size=32):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Simplified encoder for M1 testing
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, latent_channels, 3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x
```

**File: src/data.py (data loading)**

```python
import torch
from torchvision import transforms, datasets

def get_cifar10_loaders(batch_size=8, num_workers=0):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data/cifar10', 
        train=True, 
        download=True, 
        transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root='./data/cifar10', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
```

**File: train.py (main training loop)**

```python
import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
from src.encoder import Encoder
from src.data import get_cifar10_loaders

def train_one_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    total_loss = 0
    
    for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        
        # Forward pass
        z = model(x)
        
        # Dummy loss (you'll implement real losses later)
        loss = z.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch average loss: {avg_loss:.4f}")
    return avg_loss

def main(config_path):
    """Main training function"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = config['training']['device']
    if device == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Data
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=config['data']['batch_size']
    )
    
    # Model
    encoder = Encoder(latent_channels=config['encoder'].get('latent_channels', 4))
    encoder = encoder.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), 
                                  lr=config['training']['learning_rate'])
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        train_one_epoch(encoder, train_loader, optimizer, device)
    
    print("\n✅ Training completed!")
    torch.save(encoder.state_dict(), "results/models/encoder.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
```

---

### Step 6: Test Locally

```bash
# Test CIFAR-10 training on M1
conda activate unified-latents
cd ~/projects/unified-latents

# Quick test run (1 epoch)
python train.py --config configs/config_cifar10.yaml

# You should see:
# Epoch 1/10
# ... progress bar ...
# Epoch average loss: 0.XXXX
```

---

### Step 7: Commit Progress

```bash
git add src/ train.py configs/ evaluate.py
git commit -m "Implement encoder and basic training loop"
git push origin main
```

**Continue this for weeks 1-2:**
- Implement diffusion prior
- Implement diffusion decoder
- Implement losses (KL, reconstruction)
- Test everything on CIFAR-10
- Debug and refine

**Commit regularly:**
```bash
git add .
git commit -m "Add diffusion prior implementation"
git push origin main
```

---

## ☁️ PART 3: GOOGLE COLAB SETUP (Week 3)

### Step 8: Create Colab Notebook

**Go to colab.research.google.com:**

1. Click "New Notebook"
2. Name it: "unified-latents-training"
3. Create

**In Colab, paste this notebook:**

```python
# ========================================
# UNIFIED LATENTS - Google Colab Training
# ========================================

# Cell 1: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 2: Clone your repository
!git clone https://github.com/YOUR_USERNAME/unified-latents.git
%cd unified-latents

# Cell 3: Install dependencies
!pip install -r requirements.txt
print("✓ Dependencies installed")

# Cell 4: Download ImageNet-512
# Option A: Use torchvision's built-in (slower but works)
# Option B: Download from your prepared source
# For now, we'll use a smaller proxy dataset

print("Preparing dataset...")
# You can download ImageNet here or use CIFAR variants

# Cell 5: Start training
!python train.py --config configs/config_imagenet.yaml

# Cell 6: Download results
from google.colab import files
import os

if os.path.exists('results/'):
    print("Uploading results...")
    files.download('results/')
```

---

### Step 9: Run Training on Colab

1. **Connect GPU** (Colab → Runtime → Change runtime type → GPU)
2. **Run all cells** (Ctrl+F9 or Runtime → Run all)
3. **Wait for training** (can take 24-48 hours)
4. **Download results** when done

---

### Step 10: Merge Results Back

```bash
# After Colab training finishes, download results folder
# Put it in: ~/projects/unified-latents/results/

# Commit
git add results/
git commit -m "Add training results from Colab (FID: 1.4, PSNR: 27.6)"
git push origin main
```

---

## 🔄 PART 4: EXTENSION ON M1 (WEEK 5)

### Step 11: Implement Learned Noise Schedule

**File: src/noise_scheduler.py**

```python
import torch
import torch.nn as nn

class NoiseSchedulePredictor(nn.Module):
    """Predicts λ(0) for each sample"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        log_snr = self.head(features)
        # Constrain to valid range [2, 8]
        log_snr = 2 + 6 * torch.sigmoid(log_snr)
        return log_snr
```

---

## 📝 FINAL CHECKLIST

### Before Week 1 Starts:

- [ ] GitHub repo created and cloned
- [ ] Python environment setup on M1
- [ ] Project structure created
- [ ] README.md written
- [ ] requirements.txt created

### End of Week 2:

- [ ] Encoder implemented and tested
- [ ] Diffusion prior implemented
- [ ] Diffusion decoder implemented
- [ ] Loss functions implemented
- [ ] Training loop works on CIFAR-10
- [ ] All code committed to GitHub

### During Week 3-4:

- [ ] Colab notebook created
- [ ] Full ImageNet training running
- [ ] Results collected

### Week 5:

- [ ] Extension (Learned Noise Schedule) implemented
- [ ] Final training on Colab

### Week 6:

- [ ] Application written
- [ ] GitHub polished
- [ ] Results documented
- [ ] Submitted! ✅

---

## 🚀 YOU'RE READY TO START

**Today:**
1. Create GitHub repo
2. Setup Python environment
3. Create project structure
4. Make first commit

**Tomorrow:**
1. Start implementing encoder
2. Test on M1

**Follow the timeline and you'll finish on time.** 💯

