# QUICK REFERENCE SHEETS
**Print these out and keep by your desk**

---

## SHEET 1: DEBUGGING BUGS AT A GLANCE

### Bug #1: Data Leakage (OBVIOUS)
**Symptom**: Val metrics look too good, test metrics are much worse
**Cause**: Using full dataset statistics for normalization
**Fix**: Compute mean/std from TRAIN only, apply to val/test
**Check**:
```python
# WRONG ❌
mean = X.mean()  # Includes val/test!
X_norm = (X - mean) / X.std()
X_train, X_val = X_norm[:800], X_norm[800:]

# RIGHT ✅
X_train, X_val = X[:800], X[800:]
mean = X_train.mean()  # Only train
X_train = (X_train - mean) / X_train.std()
X_val = (X_val - mean) / X_train.std()  # Use train stats
```

### Bug #2: Wrong Loss Function (OBVIOUS)
**Symptom**: Crash or garbage loss values
**Cause**: Using classification loss for regression (or vice versa)
**Fix**: Match loss to task
**Check**:
```
Regression → MSELoss, L1Loss, SmoothL1Loss
Classification → CrossEntropyLoss, BCEWithLogitsLoss
```

### Bug #3: Forgot model.eval() (SUBTLE)
**Symptom**: Batch norm and dropout behave wrong during val
**Cause**: Not switching model mode before validation
**Fix**: `model.eval()` before validation loop
**Check**:
```python
model.train()     # Training
# ... training loop ...

model.eval()      # Before validation!
with torch.no_grad():
    # validation loop
```

### Bug #4: No torch.no_grad() (SUBTLE)
**Symptom**: OOM error during validation on large datasets
**Cause**: Gradients accumulated even though not needed
**Fix**: Wrap validation in `with torch.no_grad():`
**Check**:
```python
model.eval()
with torch.no_grad():  # ← THIS LINE
    for batch in val_loader:
        pred = model(batch)
```

### Bug #5: Val Loss ↓ but Metric ↑ (HARD)
**Symptom**: Confusing: loss decreasing but metric getting worse
**Causes**:
1. Different preprocessing for train/val
2. Metric computed on wrong scale
3. Overfitting (model memorizing noise)
**Fix**:
1. Use identical preprocessing
2. Ensure metric uses denormalized predictions
3. Check preprocessing matches between train/val
**Check**:
```python
# 1. Are preprocessing identical?
print(f"Train mean: {X_train.mean()}, Val mean: {X_val.mean()}")

# 2. Are loss and metric on same scale?
# Normalize targets too: y = (y - y.mean()) / y.std()

# 3. Is model overfitting?
# Plot train vs val loss: should be similar
```

### Bug #6: Optimizer Step Order (OBVIOUS)
**Symptom**: Training doesn't work, loss erratic
**Cause**: Steps in wrong order
**Fix**: backward → step → zero_grad
**Check**:
```python
loss.backward()      # 1. Compute gradients
optimizer.step()     # 2. Update weights
optimizer.zero_grad()# 3. Clear gradients
```

### Bug #7: Shape Mismatch (OBVIOUS → MEDIUM)
**Symptom**: Crash or wrong learning
**Cause**: Tensor dimensions don't match
**Fix**: Print shapes, verify they match
**Check**:
```python
# For LSTM sequence classification
lstm_out, _ = model(X)  # (batch, seq_len, hidden)
last = lstm_out[:, -1, :]  # (batch, hidden) — use LAST timestep
pred = linear_head(last)  # (batch, 1)
```

### Bug #8: Forgot Detach (SUBTLE)
**Symptom**: Memory waste, confusing error messages
**Cause**: Keeping tensors in computation graph
**Fix**: Use `.item()` for scalars, `.detach()` for tensors
**Check**:
```python
loss_value = loss.item()    # Get scalar (recommended)
loss_tensor = loss.detach() # Get tensor without gradients
```

### Bug #9: Dropout in Validation (SUBTLE)
**Symptom**: Noisy validation metrics each run
**Cause**: Dropout still active during validation
**Fix**: `model.eval()` (disables both dropout and batch norm)
**Check**:
```python
model.eval()  # Disables Dropout + BatchNorm randomness
```

### Bug #10: Gradient Accumulation Unintended (SUBTLE)
**Symptom**: Training unstable, especially first batch
**Cause**: Missing `zero_grad()` in first iteration
**Fix**: Always call `zero_grad()` before `backward()`
**Check**:
```python
for batch in loader:
    optimizer.zero_grad()   # ALWAYS first
    loss = criterion(model(batch), target)
    loss.backward()
    optimizer.step()
```

---

## SHEET 2: SPEAKING CHECKLIST

### How to Sound Like You (Not ChatGPT)

**✅ DO SAY:**
- "When I tried X, what happened was..."
- "I realized that..."
- "The issue here is..."
- "This breaks because..."
- "Looking at the code, I see..."

**❌ DON'T SAY:**
- "One could argue..."
- "It is noteworthy that..."
- "Empirical evidence suggests..."
- "Leveraging state-of-the-art techniques..."
- "To ameliorate the aforementioned issues..."

### Talking About Your Project (2-3 min)
1. **What**: "My project is [task description]"
2. **Why**: "I chose this because [motivation]"
3. **How**: "I used [model/technique]"
4. **Results**: "Best metric was [value] vs baseline [value]"
5. **Learned**: "Key insight was [what you learned]"

### Explaining a Bug (1-2 min)
1. **"The issue is on line X where..."** (what)
2. **"This happens because..."** (why)
3. **"The symptom would be..."** (observable effect)
4. **"The fix is..."** (specific change)
5. **"Why this works: ..."** (explanation)

### Discussing Your Paper (2-3 min)
1. **"The main idea is..."** (contribution)
2. **"I liked..."** (specific, with example)
3. **"I disliked..."** (constructive criticism)
4. **"I'd improve..."** (realistic extension)

### If You Pause
- **2-3 seconds**: Thinking (good)
- **5+ seconds**: Probably shouldn't pause
- **Say**: "Let me think about that for a second..."
- **Never**: Stay silent for 10+ seconds

---

## SHEET 3: YOUR NUMBERS (MEMORIZE!)

**Copy and fill in, then memorize:**

```
DATASET: ________________
SIZE: ________________ samples

SPLIT:
  Train: ________________ samples
  Val:   ________________ samples
  Test:  ________________ samples
  
WHY THIS SPLIT?
  ________________

BEST METRIC:
  Name: ________________
  Value: ________________

BASELINE:
  ________________

IMPROVEMENT:
  ________________ %

ARCHITECTURE:
  ________________
```

**Memorization trick**: Repeat these numbers 5 times out loud every morning.

---

## SHEET 4: FAILED EXPERIMENTS (MEMORIZE!)

**For each failed experiment:**

```
EXPERIMENT #1: ________________

What I tried:
  ________________

Expected:
  ________________

What actually happened:
  ________________

What I learned:
  ________________
```

**Memorization trick**: Explain each experiment out loud 5 times. Record yourself.

---

## SHEET 5: INTERVIEW TIMELINE

**Typical interview structure** (60-90 min total):

```
00:00-05:00 → Introduction & warm-up (5 min)
              "Tell me about yourself"

05:00-35:00 → Written/Spoken answers (30 min)
              Section 1: Your Project (5-10 min)
              Section 2: Debugging (5-10 min)
              Section 3: Numbers (2-3 min)
              Section 4: Reflection (2-3 min)
              Section 5: Paper (5 min)

35:00-50:00 → Paper discussion (10-15 min)
              "What did you like? Dislike? Improve?"

50:00-80:00 → Technical debugging session (30 min)
              Read unfamiliar buggy code
              Identify bugs
              Explain fixes
              Live coding

80:00-90:00 → Buffer & follow-up questions (10 min)
              Anything you want to add?
              Questions for us?

Total: 60-90 minutes
```

**Time management**: 
- Don't rush early sections
- You'll need 30+ min for debugging
- Save ~5 min for wrap-up

---

## SHEET 6: TOP 5 INTERVIEW MISTAKES TO AVOID

### Mistake 1: Rambling
**❌ Bad**: "So like, uh, the project is kind of about data and models, you know? I was thinking about, um, using different techniques and it took a long time..."

**✅ Good**: "My project predicts house prices using a CNN. I used ResNet-50 pre-trained on ImageNet. Best validation RMSE was $50K vs $75K baseline."

### Mistake 2: Generic Debugging Explanations
**❌ Bad**: "The code doesn't work because there's probably a bug somewhere that makes it not learn properly."

**✅ Good**: "The issue is on line 8: `criterion = nn.CrossEntropyLoss()` is used for regression, but CrossEntropyLoss expects classification targets. The model will crash. Fix: use `nn.MSELoss()` instead."

### Mistake 3: Wrong Paper Review
**❌ Bad**: "This paper is very good and important. It contributes a lot to the field and is well-written."

**✅ Good**: "I liked how they use Vision Transformers instead of CNNs because it simplifies the architecture while achieving 88.6% on ImageNet-21K. I disliked that they pre-train on huge datasets (JFT-300M), so the comparison isn't fair to CNNs trained on limited data."

### Mistake 4: Trying to Fake Understanding
**❌ Bad**: [Interviewer asks something] "Uh, yeah, I totally know about that. It's like, you know, the thing with the computation graphs and stuff..."

**✅ Good**: [Interviewer asks something] "I haven't worked with that specific technique, but based on X, I'd guess the issue is Y. Should I run this code to verify?"

### Mistake 5: Obvious ChatGPT Language
**❌ Bad**: "The empirical results demonstrate a statistically significant amelioration of the aforementioned performance metrics..."

**✅ Good**: "The model gets 10% better results because I added batch normalization."

---

## SHEET 7: EMERGENCY RESPONSES

**Use these if you're stuck:**

### If asked "How would you debug this?"
```
"I would:
1. Print the shapes of all tensors (check for mismatches)
2. Check if preprocesses is identical for train/val
3. Run the model on a small batch manually
4. Check loss values and gradients
5. Compare train vs val to see if overfitting"
```

### If you make a mistake
```
"Wait, I misspoke. Let me correct that. 
[Correct explanation]"
```

### If you don't know something
```
"That's a good question. I haven't encountered that specific 
scenario, but I think [your attempt]. 
Could I run this code to verify?"
```

### If you're unsure about paper question
```
"I want to make sure I'm understanding your question correctly.
Are you asking about [interpretation]?
[If yes] Then I think the answer is [your response]."
```

---

## SHEET 8: CONFIDENCE BUILDERS

**Things to remember:**

1. **They want to hire you** (they don't want to reject good people)
2. **Your project is real** (they can tell if you built it)
3. **You've practiced 15 days** (you know the material)
4. **Mistakes are OK** (everyone makes them; recovery matters)
5. **They're checking if you can think** (not if you know everything)
6. **You're in control** (speak your truth)
7. **This is just a conversation** (not a test with pass/fail)

---

## SHEET 9: RED FLAGS TO AVOID

**These will trigger rejection immediately:**

| Red Flag | Why | Avoid By |
|----------|-----|----------|
| Can't explain your own code | Shows you didn't write it | Know every line you wrote |
| Speaking like ChatGPT | Clear cheating signal | Practice speaking naturally |
| Different preprocessing train/val | Silent failure, wrong metrics | Use identical preprocessing |
| Missing `model.eval()` during val | Batch norm fails silently | Always: model.eval() + no_grad() |
| Forgetting your project numbers | Shows you don't know your work | Memorize dataset size, splits, metrics |
| Lying about understanding | Interviewer will follow up & catch you | Admit what you don't know |
| No workspace visible | Setup screams cheating | Show workspace, phone camera positioned |

---

## SHEET 10: FINAL AFFIRMATIONS

**Read these every morning before interview:**

✓ "I prepared seriously for 15 days"
✓ "I know my project inside and out"
✓ "I can identify bugs in code"
✓ "I understand what I've built"
✓ "I can explain clearly"
✓ "If I don't know, I'll think out loud"
✓ "The interviewer wants to learn from me"
✓ "I'm ready for this"

---

## SHEET 11: QUICK DEBUGGING DECISION TREE

```
Code runs but results are wrong?
├─ Is validation metric wrong?
│  ├─ Different preprocessing? → Use identical train/val preprocessing
│  ├─ Metric computed on wrong scale? → Denormalize before metric
│  └─ Did you forget model.eval()? → Add model.eval() + no_grad()
│
├─ Is training loss not decreasing?
│  ├─ Wrong loss function? → Check task (regression vs classification)
│  ├─ Optimizer step in wrong order? → backward → step → zero_grad
│  └─ Data leakage? → Split BEFORE normalizing
│
└─ Is there OOM error?
   ├─ Validation loop without no_grad()? → Wrap in with torch.no_grad():
   └─ Keeping computation graph? → Use .item() or .detach()

Code crashes?
├─ Shape mismatch? → Print tensor shapes, check dimensions
├─ Data type issue? → Convert to same dtype (float32)
└─ Loss function wrong? → Check loss matches task
```

---

## PRINT & POST

**Print this page and put it:**
- [ ] On your desk (refer before interview)
- [ ] Next to your monitor (quick reference)
- [ ] On your wall (daily affirmations)

**Review every morning:**
- [ ] Debugging sheet (remember top 5 bugs)
- [ ] Your numbers (say out loud)
- [ ] Affirmations (confidence)

---

## 🚀 YOU'RE READY!

These sheets are your safety net. You've studied hard. You know this material. Now go show them what you've built.

**Good luck! 💪**

