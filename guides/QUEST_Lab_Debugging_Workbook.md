# DEBUGGING PRACTICE WORKBOOK
**10 Real-World PyTorch & TensorFlow Bugs (with solutions)**

---

## 🎯 HOW TO USE THIS WORKBOOK

**For each bug:**
1. **Read** the code (5 minutes)
2. **Identify** ALL bugs (obvious + subtle)
3. **Write down** your answers (before looking at solution)
4. **Explain** why it's a bug and what happens
5. **Explain** the exact fix
6. **Run** the code (if you have environment) to verify
7. **Record yourself** explaining the bug (1-2 min explanation)

**This builds the skill for the live debugging section of the interview.**

---

## BUG #1: Data Leakage via Global Normalization
**Difficulty: OBVIOUS** | **Frequency in interviews: VERY HIGH**

### Buggy Code:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Data: 1000 samples, sequence length 50, 1 feature
X = torch.randn(1000, 50, 1)
y = torch.randn(1000, 1)

# ❌ LEAKAGE: Normalize using FULL dataset statistics
mean = X.mean()
std = X.std()
X_norm = (X - mean) / std

# Split AFTER normalization
X_train, X_val = X_norm[:800], X_norm[800:]
y_train, y_val = y[:800], y[800:]

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=32)

model = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for xb, yb in train_loader:
        pred, _ = model(xb)
        pred = pred[:, -1, :]
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Your Analysis (write before looking at solution):
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
The mean and std are computed using the FULL dataset (train + validation + test). This means the validation set is normalized using statistics that include validation data itself.

**Why does it matter?**
This is **data leakage**. The normalization "sees" the validation set during the normalization step. In practice:
- Your validation metrics will be **10-20% better than they actually are**
- The model "cheats" because validation data looks similar after normalization
- **In production**: You won't have validation data to compute statistics, so real performance drops significantly

**What's the fix?**
```python
# ✅ CORRECT: Split FIRST, then normalize separately

X_train_raw, X_val_raw = X[:800], X[800:]
y_train, y_val = y[:800], y[800:]

# Compute statistics from TRAINING data only
train_mean = X_train_raw.mean()
train_std = X_train_raw.std()

# Apply training statistics to both sets
X_train = (X_train_raw - train_mean) / train_std
X_val = (X_val_raw - train_mean) / train_std  # Use TRAINING statistics!

# Rest of code...
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
```

**Key principle**: Always compute normalization statistics on the training set ONLY. Apply those same statistics to validation/test.

---

## BUG #2: Wrong Loss Function for Task
**Difficulty: OBVIOUS** | **Frequency: VERY HIGH**

### Buggy Code:
```python
import torch
import torch.nn as nn

# Regression task: predict house prices
X_train = torch.randn(1000, 10)  # 10 features
y_train = torch.randn(1000, 1)   # Continuous target (price)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)  # Output: 1 value (regression)
)

# ❌ WRONG: Using classification loss for regression!
criterion = nn.CrossEntropyLoss()  # This expects class probabilities
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    pred = model(X_train)
    print(f"Pred shape: {pred.shape}")  # (1000, 1)
    print(f"Target shape: {y_train.shape}")  # (1000, 1)
    
    # This will crash or give garbage
    loss = criterion(pred, y_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
`CrossEntropyLoss` is for **classification** (predicting class labels). It expects:
- Predictions: shape (batch_size, num_classes) with logits
- Targets: shape (batch_size) with class indices

But we have:
- Predictions: shape (1000, 1) — continuous value
- Targets: shape (1000, 1) — continuous price

**Why does it matter?**
- **Will crash** with error like: "CrossEntropyLoss expects N-D input, got 2-D"
- Or if dimensions accidentally match, it produces **garbage loss values**
- Model learns nothing useful

**What's the fix?**
```python
# ✅ CORRECT: Use regression loss

# For regression, choose:
criterion = nn.MSELoss()      # Mean Squared Error (sensitive to outliers)
# OR
criterion = nn.L1Loss()       # Mean Absolute Error (robust to outliers)
# OR
criterion = nn.SmoothL1Loss() # Huber loss (balanced)

# Rest stays the same
for epoch in range(5):
    pred = model(X_train)
    loss = criterion(pred, y_train)  # Now works correctly!
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Quick reference:**
- **Classification**: `CrossEntropyLoss` (multi-class) or `BCEWithLogitsLoss` (binary)
- **Regression**: `MSELoss`, `L1Loss`, `SmoothL1Loss`
- **Ranking/Distances**: `TripletMarginLoss`, `ContrastiveLoss`

---

## BUG #3: Forgetting model.eval() During Validation
**Difficulty: SUBTLE** | **Frequency: VERY HIGH**

### Buggy Code:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_train = torch.randn(1000, 20)
y_train = torch.randn(1000, 1)
X_val = torch.randn(200, 20)
y_val = torch.randn(200, 1)

# Model with Batch Normalization
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.BatchNorm1d(64),  # Important: computes running statistics
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

for epoch in range(5):
    # Training
    model.train()  # ✓ Correct
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # ❌ Validation (FORGOT model.eval()!)
    val_loss = 0
    for xb, yb in val_loader:
        pred = model(xb)
        val_loss += criterion(pred, yb).item()
    
    print(f"Epoch {epoch}, Val Loss: {val_loss/len(val_loader)}")

print("Training done")
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
The code **never calls `model.eval()`** before validation. This means:
- Batch Normalization uses **batch statistics** (computed from current batch of 32 samples)
- Batch Norm should use **running statistics** (computed from all training batches)
- Dropout (if present) will still **randomly drop** activations in validation

**Why does it matter?**
1. **Batch norm behaves wrong**: 
   - Your validation metrics don't match real performance
   - If you validate on 1 sample at a time, batch norm uses meaningless statistics
   
2. **Metrics are unreliable**:
   - Validation loss looks better than it should (because batch norm helps)
   - When you deploy the model in production, it performs worse

3. **Memory/Speed issue** (bonus):
   - Dropout still active = slower inference
   - Gradients still accumulated = uses memory

**What's the fix?**
```python
# ✅ CORRECT: Set model to eval mode before validation

for epoch in range(5):
    # Training
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    model.eval()  # ← Add this line!
    with torch.no_grad():  # Also good practice: don't track gradients
        val_loss = 0
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    
    print(f"Epoch {epoch}, Val Loss: {val_loss/len(val_loader)}")

print("Training done")
```

**Key principle**: Always use `model.eval()` for validation/testing, and `model.train()` for training.

---

## BUG #4: Not Using no_grad() in Validation Loop
**Difficulty: MEDIUM** | **Frequency: COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_val = torch.randn(10000, 100)  # Large validation set
y_val = torch.randn(10000, 1)

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

criterion = nn.MSELoss()
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ❌ Validation loop without torch.no_grad()
model.eval()
total_loss = 0

for xb, yb in val_loader:
    pred = model(xb)
    loss = criterion(pred, yb)
    total_loss += loss.item()  # Trying to get scalar, but gradients still tracked!
    # Memory explosion! Each batch keeps a computation graph

print(f"Val Loss: {total_loss/len(val_loader)}")
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
The validation loop computes predictions **without disabling gradient tracking**. PyTorch, by default, builds a computation graph for every operation, even if you don't call `.backward()`.

**Why does it matter?**
1. **Memory usage explodes**: 
   - Each forward pass stores intermediate tensors
   - With large validation sets, you run out of memory (OOM)
   - 100 batches = 100x more memory than needed

2. **Slower inference**:
   - Computing gradients takes time (20-50% slower)
   - Even though you don't need them

3. **Silent failure**:
   - Code looks correct and runs
   - But crashes with "CUDA out of memory" on large validation sets

**What's the fix?**
```python
# ✅ CORRECT: Use torch.no_grad()

model.eval()
total_loss = 0

with torch.no_grad():  # Disable gradient tracking
    for xb, yb in val_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item()

print(f"Val Loss: {total_loss/len(val_loader)}")
```

**Why it works**: `torch.no_grad()` tells PyTorch "don't track operations for gradient computation". This:
- Frees memory immediately after each batch
- Makes inference 20-50% faster
- Prevents OOM errors on large validation sets

**Rule**: Always wrap validation/testing code with `with torch.no_grad():`

---

## BUG #5: Validation Loss ↓ but Metric ↑ (Most Common Subtle Bug)
**Difficulty: HARD** | **Frequency: VERY COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Synthetic regression data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1) * 100  # Scale: -100 to +100
X_val = torch.randn(200, 10)
y_val = torch.randn(200, 1) * 100

# Normalize X only (not y!)
X_train = (X_train - X_train.mean()) / X_train.std()
X_val = (X_val - X_val.mean()) / X_val.std()  # ❌ Different statistics!

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

for epoch in range(10):
    # Training
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)  # MSE on original scale
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_mae = 0
        
        for xb, yb in val_loader:
            pred = model(xb)
            
            # Loss on original scale
            loss = criterion(pred, yb)
            val_loss += loss.item()
            
            # MAE computed without denormalization
            mae = torch.abs(pred - yb).mean()  # ❌ On mismatched scales!
            val_mae += mae.item()
    
    print(f"Epoch {epoch}: "
          f"Train Loss={train_loss/len(train_loader):.2f}, "
          f"Val Loss={val_loss/len(val_loader):.2f}, "
          f"Val MAE={val_mae/len(val_loader):.2f}")
    
    # Output: Val Loss decreasing (10 → 0.5), MAE increasing (50 → 100)!
    # This makes no sense and confuses everyone
```

### Your Analysis:
**What are the three issues?**
```
Issue 1: [Your answer]

Issue 2: [Your answer]

Issue 3: [Your answer]
```

**Why does val_loss decrease but metric increase?**
```
[Your answer]
```

**What would you check first?**
```
1. [First thing to check]
2. [Second thing to check]
3. [Third thing to check]
```

---

### SOLUTION

**What are the three issues?**

**Issue 1: Different normalization statistics for X_val**
```
X_val = (X_val - X_val.mean()) / X_val.std()  # Uses VAL statistics
# Should use TRAIN statistics:
X_val = (X_val - X_train.mean()) / X_train.std()
```

**Issue 2: Targets are not normalized, but loss is computed on original scale**
- Model learns to predict values in [-100, +100] range
- Loss values are large (1000s)
- Hard to interpret

**Issue 3: Metric (MAE) and Loss (MSE) are on different scales**
- MSE = (pred - target)²  → Large numbers, decreasing
- MAE = |pred - target|   → Different scale, might increase
- They're not directly comparable

**Why does val_loss decrease but metric increase?**

The model is overfitting to the training data:
1. Training loss decreases (model learns training patterns)
2. But validation predictions get WORSE (higher error)
3. MSE is quadratic, so it can decrease even as MAE increases
4. The metrics measure different things!

**What would you check first?**

```python
# 1. CHECK: Is preprocessing identical for train/val?
print("X_train stats:", X_train.mean(), X_train.std())
print("X_val stats (using train params):", 
      (X_val_raw - X_train_mean) / X_train_std)

# 2. CHECK: Are loss and metric on same scale?
# Normalize targets too:
y_train_norm = (y_train - y_train.mean()) / y_train.std()
y_val_norm = (y_val - y_train.mean()) / y_train.std()  # Use TRAIN stats!

# 3. CHECK: Are predictions reasonable?
model.eval()
with torch.no_grad():
    sample_pred = model(X_val[:1])
    sample_target = y_val[:1]
    print(f"Sample prediction: {sample_pred.item():.2f}")
    print(f"Sample target: {sample_target.item():.2f}")
    print(f"Prediction range: {model(X_val).min():.2f} to {model(X_val).max():.2f}")
    print(f"Target range: {y_val.min():.2f} to {y_val.max():.2f}")
    # If prediction range is [-0.5, 0.5] but target is [-100, 100]: BUG!
```

**The fix:**
```python
# ✅ CORRECT approach:

# Split FIRST
X_train_raw, X_val_raw = X[:800], X[800:]
y_train_raw, y_val_raw = y[:800], y[800:]

# Compute statistics from TRAINING data only
X_mean, X_std = X_train_raw.mean(), X_train_raw.std()
y_mean, y_std = y_train_raw.mean(), y_train_raw.std()

# Apply identical preprocessing to both sets
X_train = (X_train_raw - X_mean) / X_std
y_train = (y_train_raw - y_mean) / y_std
X_val = (X_val_raw - X_mean) / X_std
y_val = (y_val_raw - y_mean) / y_std

# Now train and validate consistently
```

---

## BUG #6: Optimizer Step in Wrong Order
**Difficulty: MEDIUM** | **Frequency: COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn

X = torch.randn(1000, 10)
y = torch.randn(1000, 1)

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5):
    for i in range(0, len(X), 32):
        xb = X[i:i+32]
        yb = y[i:i+32]
        
        pred = model(xb)
        loss = criterion(pred, yb)
        
        # ❌ WRONG ORDER
        optimizer.step()        # Update before clearing gradients!
        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Compute new gradients
        
        print(f"Loss: {loss.item():.4f}")

# Output: Loss won't decrease (or very erratic)
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
The order of operations is wrong:
1. `optimizer.step()` uses **old gradients** (from previous iteration or random)
2. `optimizer.zero_grad()` clears those gradients
3. `loss.backward()` computes new gradients

So each `step()` is updating with stale gradients!

**Why does it matter?**
- **Training doesn't work**: Loss doesn't decrease predictably
- **Erratic behavior**: Model jumps around instead of converging
- **Silent failure**: Code runs, but learn nothing

**What's the fix?**
```python
# ✅ CORRECT order:
for epoch in range(5):
    for i in range(0, len(X), 32):
        xb = X[i:i+32]
        yb = y[i:i+32]
        
        pred = model(xb)
        loss = criterion(pred, yb)
        
        # Correct order:
        loss.backward()         # 1. Compute gradients
        optimizer.step()        # 2. Update using those gradients
        optimizer.zero_grad()   # 3. Clear for next iteration
        
        print(f"Loss: {loss.item():.4f}")

# Loss decreases smoothly ✓
```

**Common variant (also correct):**
```python
optimizer.zero_grad()   # Clear old gradients first (fine if first iteration)
loss.backward()
optimizer.step()
```

---

## BUG #7: Incorrect Tensor Shape Mismatch
**Difficulty: MEDIUM** | **Frequency: VERY COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn

# Batch size 32, sequence length 10, features 5
X = torch.randn(32, 10, 5)
y = torch.randn(32, 1)  # Target for each sequence

# LSTM model
model = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
linear_head = nn.Linear(64, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    # Forward pass
    lstm_out, (hidden, cell) = model(X)  # lstm_out shape: (32, 10, 64)
    
    # ❌ Bug: Using wrong output
    # Take first timestep instead of last
    pred = linear_head(lstm_out[:, 0, :])  # (32, 64) → (32, 1)
    
    # This might not crash, but it's using the FIRST output, not the LAST!
    # For sequence classification, we want the LAST timestep
    
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Model trains but learns wrong pattern (using first step instead of last)
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
The code takes `lstm_out[:, 0, :]` (first timestep) instead of `lstm_out[:, -1, :]` (last timestep).

For **sequence classification**, you typically want to make a prediction based on the entire sequence, which is captured in the **last hidden state** of the LSTM.

**Why does it matter?**
- **Wrong feature space**: Model uses information from beginning of sequence, not the end
- **Bad performance**: Can't leverage sequential dependencies properly
- **Silent failure**: Code runs fine, just learns wrong patterns

**What's the fix?**
```python
# ✅ CORRECT: Use last timestep

for epoch in range(3):
    lstm_out, (hidden, cell) = model(X)  # Shape: (32, 10, 64)
    
    # Option 1: Use last timestep explicitly
    last_output = lstm_out[:, -1, :]  # (32, 64) — last timestep for each sample
    pred = linear_head(last_output)   # (32, 1)
    
    # Option 2: Use hidden state directly (same as [:, -1, :])
    # pred = linear_head(hidden.squeeze(0))  # (32, 1)
    
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Key insight**: For RNN/LSTM/GRU sequence classification, always use the **last timestep** unless you have a specific reason to use something else.

---

## BUG #8: Double Backprop Without Detach
**Difficulty: HARD** | **Frequency: LESS COMMON but Tricky**

### Buggy Code:
```python
import torch
import torch.nn as nn

X = torch.randn(10, 5)
y = torch.randn(10, 1)

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    pred = model(X)
    loss = criterion(pred, y)
    
    # ❌ Computing second loss from first loss without detaching
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Trying to use the loss value again (bad practice)
    loss_value = loss  # Still connected to computation graph
    
    # This creates a meta-graph: gradients of gradients
    # If you try to backprop again, it fails:
    # loss_value.backward()  # ERROR: leaf variable doesn't require grad
    
    print(f"Loss: {loss_value}")  # OK, but unusual pattern

# This is confusing and error-prone
```

### Your Analysis:
**What's the potential bug?**
```
[Your answer here]
```

**Why does this pattern cause problems?**
```
[Your answer here]
```

**What's the better way?**
```
[Your answer here]
```

---

### SOLUTION

**What's the potential bug?**
If you want to accumulate loss values or reuse them, keeping them connected to the computation graph creates confusing higher-order derivatives.

**Why does this pattern cause problems?**
1. **Memory**: Keeps the entire computation graph in memory unnecessarily
2. **Confusion**: If you try to backprop twice, it fails with "leaf variable doesn't require grad"
3. **Performance**: Extra bookkeeping for gradients you don't need

**What's the better way?**
```python
# ✅ CORRECT: Detach scalars you want to keep

for epoch in range(3):
    pred = model(X)
    loss = criterion(pred, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Option 1: Use .item() to get Python float (recommended)
    loss_value = loss.item()  # Disconnect from computation graph
    
    # Option 2: Use .detach() to get tensor without gradients
    loss_tensor = loss.detach()
    
    print(f"Loss: {loss_value:.4f}")  # Clean, no gradient tracking

print(f"All losses: {[l.item() for l in losses]}")
```

**Rule**: Use `.item()` when you want a scalar value for logging/printing.

---

## BUG #9: Using train() vs eval() Model During Eval (Dropout Variant)
**Difficulty: SUBTLE** | **Frequency: COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_train = torch.randn(500, 20)
y_train = torch.randn(500, 1)
X_val = torch.randn(100, 20)
y_val = torch.randn(100, 1)

# Model with Dropout (for regularization)
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Dropout(0.5),  # Drop 50% during training
    nn.Linear(128, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

best_val_loss = float('inf')

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # ❌ Validation WITHOUT model.eval() (forgot it!)
    val_loss = 0
    num_batches = 0
    for xb, yb in val_loader:
        pred = model(xb)  # Dropout is still ACTIVE!
        loss = criterion(pred, yb)
        val_loss += loss.item()
        num_batches += 1
    
    avg_val_loss = val_loss / num_batches
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Epoch {epoch}: New best val loss = {best_val_loss:.4f}")

# Problem: Validation loss is noisy because dropout is active
# Each validation run gives different results!
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter specifically with Dropout?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
`model.eval()` is never called before validation. Dropout remains active during validation.

**Why does it matter specifically with Dropout?**
- **Dropout is stochastic**: It randomly drops different units each time
- **Validation metrics become noisy**: Each validation run gives different loss values
- **Unreliable early stopping**: You can't tell if the model is actually improving
- **Metrics don't match test performance**: Because test might drop different units

**Example**: Val loss sequence might look like:
```
Epoch 0: Val Loss = 0.453
Epoch 1: Val Loss = 0.421  (improved!)
Epoch 2: Val Loss = 0.438  (worse... but maybe just dropout noise?)
Epoch 3: Val Loss = 0.414  (improved!)
```

Hard to know if model is actually improving or if it's just dropout variance.

**What's the fix?**
```python
# ✅ CORRECT: model.eval() disables dropout

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation: Disable Dropout
    model.eval()
    with torch.no_grad():
        val_loss = 0
        num_batches = 0
        for xb, yb in val_loader:
            pred = model(xb)  # Dropout is DISABLED now
            loss = criterion(pred, yb)
            val_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = val_loss / num_batches
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"Epoch {epoch}: New best val loss = {best_val_loss:.4f}")

# Validation loss is now smooth and reliable
```

**Rule**: `model.eval()` disables BOTH Dropout AND Batch Norm randomness.

---

## BUG #10: Accumulating Gradients Unintentionally
**Difficulty: MEDIUM** | **Frequency: COMMON**

### Buggy Code:
```python
import torch
import torch.nn as nn

X = torch.randn(1000, 10)
y = torch.randn(1000, 1)

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ❌ Missing zero_grad() at the start
for epoch in range(5):
    for i in range(0, len(X), 32):
        xb = X[i:i+32]
        yb = y[i:i+32]
        
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()  # Accumulates gradients
        optimizer.step()
        
        # ❌ zero_grad() is inside the loop but MISSING first iteration
        if i > 0:  # Only clears after first batch!
            optimizer.zero_grad()

# Result: First batch has larger gradient steps (accumulated with random initial gradients)
# Training is unstable and slow to converge
```

### Your Analysis:
**What's the bug?**
```
[Your answer here]
```

**Why does it matter?**
```
[Your answer here]
```

**What's the fix?**
```
[Your answer here]
```

---

### SOLUTION

**What's the bug?**
`optimizer.zero_grad()` is skipped for the first batch (`if i > 0`). This means:
1. First batch computes gradients with uninitialized/random previous gradients
2. Gradient step for batch 0 is based on accumulated garbage
3. Only from batch 1 onwards does the gradient clearing work correctly

**Why does it matter?**
- **First batch has huge gradient updates**: Unstable optimization
- **Slower convergence**: Takes more epochs to stabilize
- **Poor results**: Model doesn't converge to good solution

**What's the fix?**
```python
# ✅ CORRECT: Always zero_grad() before backward()

for epoch in range(5):
    for i in range(0, len(X), 32):
        xb = X[i:i+32]
        yb = y[i:i+32]
        
        optimizer.zero_grad()   # Clear EVERY iteration (before backward)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

# Smooth, stable training from batch 0 onwards
```

**Standard pattern** (what you should always do):
```python
for epoch in range(num_epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()    # 1. Clear old gradients
        
        output = model(batch_data)
        loss = criterion(output, target)
        loss.backward()          # 2. Compute new gradients
        optimizer.step()         # 3. Update parameters
```

---

## 📊 SUMMARY TABLE

| Bug | Obvious? | Impact | Fix |
|-----|----------|--------|-----|
| Data leakage (normalization) | NO | Metrics 10-20% too good | Split→Normalize each set |
| Wrong loss function | YES | Crash or garbage | Match task (Regression=MSE, Classification=CE) |
| Forgot model.eval() | NO | Batch norm + dropout wrong | Add model.eval() before validation |
| No torch.no_grad() | NO | OOM on large datasets | Wrap validation in `with torch.no_grad():` |
| Loss ↓ Metric ↑ | NO | Confusing results | Check preprocessing, denormalization |
| Optimizer step order | YES | Training fails | backward → step → zero_grad |
| Shape mismatch | YES/NO | Crash or wrong learning | Check tensor dimensions |
| Forgot detach() | NO | Memory waste | Use .item() for logging |
| Dropout in validation | NO | Noisy metrics | Add model.eval() |
| Gradient accumulation | NO | Unstable training | zero_grad() every iteration |

---

## 🎤 PRACTICE SCRIPT

For each bug, practice explaining out loud:

**Template** (2-3 minutes per bug):
1. **"The issue in this code is..."** [Identify bug]
2. **"This happens because..."** [Explain mechanism]
3. **"The observable symptom would be..."** [What goes wrong]
4. **"The fix is..."** [Specific change]
5. **"Why this fix works..."** [Explain the correction]

**Record yourself** explaining each bug. Listen for:
- ✅ Clarity (no "ums" or "likes")
- ✅ Technical accuracy (use correct terminology)
- ✅ Confidence (sounds like you know it)
- ✅ Conciseness (2-3 min, not 10 min)

Good luck! 🚀

