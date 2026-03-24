# QUEST Lab Internship - 15 Day Intensive Preparation Guide
**Summer 2026 | Deep Learning Interview & Selection**

---

## 🎯 CRITICAL SUCCESS FACTORS
1. **Avoid ChatGPT Detection**: All responses must be genuinely yours (they check speaking clarity + screen sharing + workspace verification)
2. **Deep Technical Understanding**: Not just knowing how to run code, but WHY things work
3. **Debugging Mastery**: The in-person interview focuses heavily on identifying bugs in real code
4. **Paper Reading**: Must be genuine — they'll ask follow-ups
5. **Project Authenticity**: Your project is your "proof of work"

---

## 📅 TIMELINE: 15 DAYS BREAKDOWN

### **DAYS 1-3: Project Foundation (40% effort)**
**Goal**: Finalize and document your DL project

#### Day 1: Project Audit
- [ ] Review your GitHub/Colab notebook
- [ ] Check README exists with:
  - Problem statement (1-2 sentences)
  - Dataset description (what, size, splits)
  - Model architecture (diagram or description)
  - Results (best metric, baseline comparison)
  - How to run the code (exact commands)
- [ ] Ensure code is clean (no debug prints, unused variables)
- [ ] Verify reproducibility: Can someone run it with your instructions?

#### Day 2: Documentation & Experiment Writeup
- [ ] Document 2-3 experiments you tried
- [ ] For EACH failed experiment, write:
  - **What you tried**: Specific change (e.g., "added batch norm", "changed learning rate from 0.1 to 0.001")
  - **Expected outcome**: What you thought would happen
  - **Actual observation**: What actually happened (loss values, metric changes, error messages)
  - **What you learned**: Concrete conclusion (2-3 sentences)
- [ ] Be prepared to explain these verbally without notes (they'll ask in interview)

#### Day 3: Metrics & Baselines
- [ ] Know your numbers cold:
  - Dataset size, exact train/val/test split with counts
  - Why you chose that split (e.g., "80-10-10 for adequate validation data")
  - Best validation metric value (exact number)
  - Baseline (mean predictor, previous SOTA, or dummy classifier)
  - Improvement over baseline (%)
- [ ] Be ready to justify metric choice (why MSE vs MAE vs accuracy?)

---

### **DAYS 4-6: Debugging Deep Dive (35% effort)**
**Goal**: Master PyTorch debugging patterns (this is your interview focus!)

#### Day 4: Common DL Bugs Masterclass
Study and understand each bug pattern:

**Bug Category 1: Data Issues**
- Forgetting to normalize/standardize
- Using global statistics for normalization (⚠️ LEAKAGE!)
- Incorrect train/val/test splits (mixing before split)
- Batch size mismatches
- Data type mismatches (int vs float)

**Bug Category 2: Model Architecture Issues**
- Wrong loss function for task (CrossEntropyLoss for regression ❌)
- Output shape mismatch with target shape
- Missing softmax/sigmoid for multi-class/binary
- Forgetting to move model to GPU
- Incorrect feature dimensions

**Bug Category 3: Training Loop Issues**
- `optimizer.zero_grad()` in wrong place
- Not detaching hidden states in RNNs
- Gradient accumulation without clear intent
- Not setting `model.eval()` during validation
- Using training metrics on validation data

**Bug Category 4: Evaluation Issues**
- Computing loss without `with torch.no_grad()`
- Different preprocessing for train vs val/test
- Evaluating on train set instead of val set
- Not resetting metrics between epochs

#### Day 5: Live Debugging Practice
Create 5 buggy code snippets and practice:
1. Read the code (5 min)
2. Identify ALL bugs (obvious + subtle ones)
3. Write down the exact fix
4. Explain WHY it's a bug
5. Verify with actual code execution

**Focus on SUBTLE bugs:**
- `validation_loss ↓ but metric ↑` scenarios
- Off-by-one errors
- Silent failures (code runs but gives wrong results)

#### Day 6: Interview-Specific Debugging
Practice explaining bugs verbally:
- [ ] Record yourself explaining a bug (3 min explanation)
- [ ] Listen for: clarity, jargon use (explain like a senior), confidence
- [ ] Practice explaining WITHOUT looking at code
- [ ] Time yourself: 2-3 min per bug explanation

**Key Phrase Patterns**:
- "The issue is... [specific line], because..."
- "This leads to... [observable symptom]"
- "The fix is... [exact change]"
- "Why this matters: [impact on training/results]"

---

### **DAYS 7-9: Paper Reading Excellence (15% effort)**
**Goal**: Find & deeply understand 1 relevant paper

#### Day 7: Paper Selection
- [ ] Browse ICML 2025, NeurIPS 2025, ICLR 2025-2026 main track papers
- [ ] Pick a paper related to your project domain (or QUEST Lab papers if they exist)
- [ ] Choose papers that are **not** 20+ pages dense theory papers
- [ ] **Preferred**: Look for papers with clear methodology + results

**Where to find papers:**
- ICML 2025: https://icml.cc/virtual/2025/
- NeurIPS 2025: https://neurips.cc/
- ICLR 2025: https://iclr.cc/
- TMLR (Journal): https://openreview.net/venue/tmlr

#### Day 8: Deep Reading (NOT skimming)
Read in this order:
1. **Title + Abstract** (5 min) — What's the problem?
2. **Figures + Captions** (5 min) — What's the key insight?
3. **Introduction** (5 min) — Why does this matter?
4. **Method Section** (10 min) — What exactly did they do?
5. **Results Section** (5 min) — Did it work?
6. **One Deep Dive** (10 min) — Pick ONE interesting part and really understand it

**Do NOT read**: Heavy math sections if not essential

#### Day 9: Structured Feedback Writeup
Write 100 words each for:

**What you liked:**
- Clarity? Novel approach? Strong results? Good experimental design?
- Concrete example from paper

**What you disliked:**
- Limited scope? Unfair baselines? Unclear writing? Missing ablations?
- Specific example

**What to improve:**
- Missing experiment? Broader evaluation? Different approach to method?
- Be constructive, not just critical

**Practice Tip:** Write this without referencing the paper — then check accuracy. This proves real reading.

---

### **DAYS 10-12: Interview Simulation (7% effort)**
**Goal**: Practice speaking + screen sharing + live coding

#### Day 10: Mock Interview Setup
- [ ] Set up your workspace as it will be during interview:
  - Desk with monitor (or laptop)
  - Phone on tripod/stand showing workspace
  - PyTorch/Jupyter ready to go
  - GitHub repo accessible
  - Quiet room, good lighting
- [ ] Practice screen sharing (Zoom/Google Meet)
- [ ] Have your project running and accessible

#### Day 11: Speaking Practice
- [ ] Record yourself answering Section 1-4 questions (from application form)
- [ ] Focus on:
  - Speaking clearly (no "um", "uh", "like")
  - Technical accuracy (use correct terminology)
  - Conciseness (don't ramble for 5+ minutes)
  - Confidence (sound like you know your stuff)
- [ ] Target: 2-3 min per answer
- [ ] Re-record until satisfied

#### Day 12: Live Debugging Session
- [ ] Get a friend or do this yourself (record):
  - Show buggy code on screen
  - Explain the bugs
  - Live-edit the code to fix
  - Run it to verify fix works
- [ ] They want to see:
  - You reading and understanding code quickly
  - Identifying errors without running it first (then verifying)
  - Logical thinking out loud
  - Confidence in fixing

---

### **DAYS 13-15: Final Polish & Rest (3% effort)**
**Goal**: Review, memorize, and prepare mentally

#### Day 13: Knowledge Check
- [ ] Flashcard review:
  - Your project numbers (dataset size, splits, metrics)
  - 3 failed experiments (what, why, what learned)
  - 3 common DL bugs and how to spot them
  - Your paper (1-2 sentence summary)
- [ ] No new material — consolidate only

#### Day 14: Full Mock Interview (2 hours)
- [ ] Set up realistic conditions
- [ ] Do a full run-through:
  1. Answer written questions (30 min writing)
  2. Screen share + speak about project (15 min)
  3. Live debugging of unfamiliar code (30 min)
  4. Paper questions (10 min)
  5. Reflection + tie-breakers (5 min)
- [ ] Record and review

#### Day 15: Rest + Final Review
- [ ] Light review only (30 min)
- [ ] Get good sleep
- [ ] Mental prep: You've studied hard, you're ready
- [ ] Have links/documents ready:
  - GitHub URL
  - Paper PDF
  - Colab link (if applicable)

---

## 🐛 DEBUGGING PRACTICE: The 5 Critical Bug Patterns

### Bug Pattern 1: Data Leakage via Normalization (COMMON!)
```python
# ❌ WRONG
mean = X.mean()  # Computed on FULL dataset
std = X.std()
X_norm = (X - mean) / std
X_train, X_val = X_norm[:800], X_norm[800:]

# ✅ CORRECT
X_train, X_val = X[:800], X[800:]
mean = X_train.mean()  # Compute from TRAIN only
std = X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std  # Use train statistics!
```
**Why it matters**: Val metrics become unrealistically optimistic (10-20% inflation). In production, model fails.

### Bug Pattern 2: Wrong Loss Function
```python
# ❌ WRONG for regression
criterion = nn.CrossEntropyLoss()  # For classification!
pred = model(x)  # Shape: (32, 1) for regression
loss = criterion(pred, y)  # Crashes or gives garbage

# ✅ CORRECT for regression
criterion = nn.MSELoss()
pred = model(x)  # Shape: (32, 1)
loss = criterion(pred, y)  # Works correctly
```
**Why it matters**: Code might crash immediately, or silently fail with 0 loss.

### Bug Pattern 3: Not Using no_grad() for Validation
```python
# ❌ WRONG - Validation loop builds computation graph
model.eval()
for xb, yb in val_loader:
    pred = model(xb)
    loss = criterion(pred, yb)  # Accumulates gradients!

# ✅ CORRECT
model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
```
**Why it matters**: Memory usage explodes during validation (might OOM). Also slower.

### Bug Pattern 4: Forgetting model.eval()
```python
# ❌ WRONG
model.train()
# ... train loop ...
# Validation loop (forget to switch mode)
for xb, yb in val_loader:
    pred = model(xb)  # Batch norm uses batch stats, not running stats!
    loss = criterion(pred, yb)

# ✅ CORRECT
model.eval()  # Use running statistics
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
```
**Why it matters**: Validation metrics don't match real performance. Batch norm behaves differently on 1 sample vs batch.

### Bug Pattern 5: Validation Loss ↓ but Metric ↑
```python
# Symptom: Loss decreasing, but MAE/RMSE increasing
# Possible causes:

# Cause 1: Different train/val preprocessing
train_scale = train_data.std()
# Validation uses different scale ❌

# Cause 2: Metric computed before denormalization
# Loss sees normalized values (small)
# Metric sees denormalized values (large) ❌

# Cause 3: Train loss uses batch norm momentum
# Val loss computed with wrong batch norm statistics ❌

# Solution: Check your preprocessing pipeline
# Make sure train/val use identical preprocessing
# Verify metric computation on denormalized predictions
```

---

## 📋 WRITTEN ANSWERS CHECKLIST

### Section 1: Your Project
- [ ] GitHub/Colab link works and is clean
- [ ] Problem statement is clear (1 sentence)
- [ ] At least 1 failed experiment documented
  - [ ] What exactly you tried (specific parameter/change)
  - [ ] Expected outcome before trying
  - [ ] Actual observation (with numbers if possible)
  - [ ] Concrete conclusion (what this taught you)
  - [ ] 3-5 sentences, no fluff

### Section 2: Debugging
- [ ] Specific bug from your actual project (NOT generic)
- [ ] Exact error message copy-pasted
- [ ] First thing you tried (specific, not "I debugged")
- [ ] Whether it worked (be honest!)
- [ ] 3 specific checks for val_loss ↓ but metric ↑:
  - [ ] Not generic (tailored to YOUR project)
  - [ ] In priority order (check most likely first)
  - [ ] Specific commands/checks (not vague)

### Section 3: Numbers
- [ ] Dataset name + exact sample count
- [ ] Train/val/test split (exact counts or %)
- [ ] Justification for split choice (why those percentages?)
- [ ] Best metric value (exact number, not "good")
- [ ] Baseline description (what did you compare against?)

### Section 4: Reflection
- [ ] One concrete thing you'd redo
- [ ] Specific reason why
- [ ] How you'd approach it differently
- [ ] 2-3 sentences, shows maturity

### Section 5: Paper
- [ ] Paper is from ICML/NeurIPS/ICLR main track 2025-2026
- [ ] You actually read it (not ChatGPT summaries)
- [ ] Exactly 100 words for each section (liked/disliked/improve)
- [ ] Concrete examples from paper (not generic)
- [ ] Shows critical thinking (not just praise)

---

## 🎤 INTERVIEW SPEAKING TIPS

### Sound Like a Senior (Not ChatGPT)
✅ **Do this:**
- "The issue I ran into was..."
- "When I tried X, what happened was..."
- "The reason that matters is..."
- "If I had to do it again, I'd..."

❌ **Don't do this:**
- "One would argue that..."
- "It is noteworthy that..."
- "Extensive experimentation revealed..."
- "Leveraging state-of-the-art techniques..."

### Answering Unexpected Questions
- **Pause for 2-3 seconds** (shows thinking, not scripted)
- **Start with**: "That's a good question. Let me think about that."
- **Give your genuine attempt** (even if imperfect)
- **Admit uncertainty**: "I'm not entirely sure, but I think..."
- **Offer to check**: "I could run this code to verify..."

### If You Don't Know
- ✅ "I haven't worked with that specific technique, but based on X, I'd guess Y"
- ❌ "I don't know" (alone, without thinking)
- ✅ "Let me think through this step by step..."
- ❌ "I memorized that ChatGPT said..."

---

## ⚠️ THINGS THAT GET YOU REJECTED IMMEDIATELY

1. **ChatGPT Detection**
   - Speaking sounds robotic or too formal
   - Can't explain your code in natural language
   - Answers are memorized-sounding
   - Can't answer follow-up questions on your project

2. **Wrong Debugging Logic**
   - Claiming a bug that isn't one
   - Suggesting fix that doesn't solve the problem
   - Not understanding WHY the fix works

3. **Inaccurate Numbers**
   - Don't know your dataset size
   - Can't articulate your exact train/val split
   - Don't know your best metric value
   - Can't name your baseline

4. **Dishonest Paper Review**
   - You clearly skimmed it
   - Can't answer basic questions about method
   - Used ChatGPT to summarize (they'll ask follow-ups)

5. **Screen Setup Violations**
   - Can't show workspace
   - Phone positioned suspiciously
   - Others visible or audible
   - Suspicious pauses while "thinking"

---

## 🚀 FINAL REMINDERS

**You have 15 days.** This is enough if you:
1. **Focus ruthlessly** — Only essential material
2. **Practice speaking** — Record yourself 5+ times
3. **Verify authenticity** — Know your project inside-out
4. **Sleep well** — Don't burn out in last 2 days
5. **Trust your work** — If it's genuinely yours, it shows

**The biggest filter is ChatGPT detection.** If they believe you wrote it and understand it, you're 80% of the way to an offer.

Good luck! You've got this. 💪

---

## 📚 Quick Reference Links
- ICML 2025: https://icml.cc/virtual/2025/
- NeurIPS 2025: https://neurips.cc/
- ICLR 2025: https://iclr.cc/
- PyTorch Debugging Guide: https://pytorch.org/docs/stable/debugging.html

