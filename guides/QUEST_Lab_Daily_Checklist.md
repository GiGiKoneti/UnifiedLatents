# 15-DAY QUEST LAB PREP: DAILY CHECKLIST
**Print this out and check off tasks daily**

---

## 📅 DAYS 1-3: PROJECT FOUNDATION

### DAY 1 - Project Audit
- [ ] Review GitHub repo/Colab notebook for cleanliness
- [ ] Check README exists with:
  - [ ] Problem statement (clear, 1-2 sentences)
  - [ ] Dataset description (name, size, splits)
  - [ ] Model architecture explanation
  - [ ] Results summary (best metric vs baseline)
  - [ ] How to run (exact commands/instructions)
- [ ] Remove all debug code, print statements, unused variables
- [ ] Test: Can I run this project from scratch? YES / NO

### DAY 2 - Experiment Documentation
- [ ] Document 2-3 experiments that failed:
  - [ ] Exp 1: What I tried → What I expected → What happened → What I learned
  - [ ] Exp 2: What I tried → What I expected → What happened → What I learned
  - [ ] Exp 3: What I tried → What I expected → What happened → What I learned
- [ ] Practice explaining experiments OUT LOUD (no notes)
- [ ] Record yourself and listen to clarity
- [ ] Timing check: Can I explain each in 1-2 minutes? YES / NO

### DAY 3 - Know Your Numbers Cold
- [ ] Dataset: Name = __________, Size = __________
- [ ] Train/Val/Test Split: ___% / ___% / ___% = __________ / __________ / __________
- [ ] Why this split? __________________________________________
- [ ] Best validation metric: __________ = __________
- [ ] Baseline (mean predictor/SOTA): __________
- [ ] Improvement: __________ %
- [ ] Metric choice justification: __________________________________
- [ ] Quiz yourself: Cover answers, recite from memory

---

## 🐛 DAYS 4-6: DEBUGGING DEEP DIVE

### DAY 4 - Bug Patterns Study
- [ ] Study: Data Leakage (normalization, split order)
  - [ ] Understand the problem
  - [ ] Know the fix
  - [ ] Explain out loud
- [ ] Study: Wrong Loss Function
  - [ ] Classification vs Regression distinction
  - [ ] CrossEntropyLoss vs MSELoss vs L1Loss
- [ ] Study: Batch Norm in Train vs Eval
  - [ ] How batch norm differs in train/eval mode
  - [ ] Why model.eval() matters
- [ ] Study: Gradient Accumulation mistakes
  - [ ] When optimizer.zero_grad() is called
  - [ ] Why it matters
- [ ] Study: Validation Metric Issues
  - [ ] Different preprocessing between train/val
  - [ ] Metric computed on wrong scale

### DAY 5 - Live Debugging Practice
Create 5 buggy code snippets and for EACH:
- [ ] Bug 1: Read code (5 min) → Identify bugs → Explain fix → Run/verify
- [ ] Bug 2: Read code (5 min) → Identify bugs → Explain fix → Run/verify
- [ ] Bug 3: Read code (5 min) → Identify bugs → Explain fix → Run/verify
- [ ] Bug 4: Read code (5 min) → Identify bugs → Explain fix → Run/verify
- [ ] Bug 5: Read code (5 min) → Identify bugs → Explain fix → Run/verify

For EACH, check:
- [ ] Found obvious bug AND subtle bug
- [ ] Explained WHY it's a bug
- [ ] Identified the IMPACT (what goes wrong?)
- [ ] Gave exact fix
- [ ] Verified fix works with code execution

### DAY 6 - Speaking Practice
- [ ] Record yourself explaining Bug 1 (out loud, 2-3 min)
- [ ] Record yourself explaining Bug 2 (out loud, 2-3 min)
- [ ] Record yourself explaining Bug 3 (out loud, 2-3 min)
- [ ] Listen back: Clear? Technical? Confident?
- [ ] Re-record until satisfied
- [ ] Practice WITHOUT looking at code
- [ ] Key check: Can you explain from memory?

---

## 📚 DAYS 7-9: PAPER READING

### DAY 7 - Find Your Paper
- [ ] Browse ICML 2025 papers: https://icml.cc/virtual/2025/
- [ ] Browse NeurIPS 2025 papers: https://neurips.cc/
- [ ] Browse ICLR 2025-2026 papers: https://iclr.cc/
- [ ] Check QUEST Lab papers: __________
- [ ] Paper chosen: __________________________________________
- [ ] Paper is from main track (not workshop)? YES / NO
- [ ] Related to your project domain? YES / NO (optional but helpful)

### DAY 8 - Deep Reading
- [ ] Read in order:
  - [ ] Title + Abstract (5 min) — Problem statement clear?
  - [ ] Figures + Captions (5 min) — Key insight?
  - [ ] Introduction (5 min) — Why does this matter?
  - [ ] Method Section (10 min) — What exactly did they do?
  - [ ] Results (5 min) — Did it work? By how much?
  - [ ] One section deep dive (10 min) — Understand one interesting part
- [ ] Do NOT read: Dense math sections (unless essential)
- [ ] Highlight: Key claims, results, novel aspects
- [ ] Timing check: Did you finish? YES / NO

### DAY 9 - Write Structured Feedback (100 words each)

**What You Liked (exactly 100 words):**
```
[Write here - concrete example from paper]
[Focus on: clarity, novelty, methodology, or results]
[Count words: ___]
```

**What You Disliked (exactly 100 words):**
```
[Write here - specific limitation]
[Focus on: scope, baselines, clarity, or missing ablations]
[Count words: ___]
```

**What To Improve (exactly 100 words):**
```
[Write here - constructive suggestion]
[Focus on: broader evaluation, different approach, or extension]
[Count words: ___]
```

- [ ] All three sections written (100 ± 5 words each)
- [ ] No ChatGPT language (check for formality/robotics)
- [ ] Concrete examples from paper (not generic)
- [ ] Can you explain without referencing paper? (Proof of real reading)

---

## 🎤 DAYS 10-12: INTERVIEW SIMULATION

### DAY 10 - Setup & Technical Test
- [ ] Workspace setup:
  - [ ] Monitor/laptop positioned well
  - [ ] Phone on tripod/stand (shows desk + surroundings)
  - [ ] Good lighting
  - [ ] Quiet room
  - [ ] Headphones/mic working
- [ ] Software ready:
  - [ ] PyTorch/TensorFlow installed
  - [ ] Jupyter Notebook or Colab working
  - [ ] GitHub repo accessible
  - [ ] Paper PDF available
- [ ] Screen sharing test:
  - [ ] Open Zoom / Google Meet
  - [ ] Share screen (practice 2-3 times)
  - [ ] Switch between windows smoothly
  - [ ] Show code clearly

### DAY 11 - Speaking Practice (Sections 1-4)

**Section 1: Your Project (2-3 min)**
- [ ] Record answer to: "Describe your project in 2-3 minutes"
- [ ] Check: Clear problem statement? Results? Takeaways?
- [ ] Listen for: Confidence? Technical accuracy? No rambling?

**Section 2: Debugging (2-3 min)**
- [ ] Record answer to: "Describe a bug you encountered"
- [ ] Check: Specific error message? What you tried? Learning?
- [ ] Listen for: Concrete details, not vague answers

**Section 3: Your Numbers (1-2 min)**
- [ ] Record answer to: "What's your dataset, split, and baseline?"
- [ ] Check: Exact numbers? Not approximations?
- [ ] Timing: No more than 2 minutes

**Section 4: Reflection (1-2 min)**
- [ ] Record answer to: "What would you redo and why?"
- [ ] Check: Specific thing? Good reason? Realistic?

**Section 5: Paper (2-3 min)**
- [ ] Record answer to: "What's your paper about?"
- [ ] Check: Can explain without notes?

- [ ] Re-record any answer you're not happy with
- [ ] Final check: Do you sound like YOU (not ChatGPT)?

### DAY 12 - Live Debugging Session

**Setup:**
- [ ] Screen sharing active
- [ ] Phone positioned to show workspace
- [ ] Code editor open (VS Code or Jupyter)
- [ ] Have buggy code snippet ready

**Execution:**
- [ ] Display buggy code on screen
- [ ] Explain bugs (WITHOUT running first) — 2 min
- [ ] List all bugs you see: __________, __________, __________
- [ ] For each bug, explain: What's wrong? Why does it break? What's the fix?
- [ ] Live-edit: Fix the bugs on screen
- [ ] Run: Execute code to verify fixes work
- [ ] Explain: Why the fixed version works

**Check:**
- [ ] Did you identify 1 obvious + 1 subtle bug?
- [ ] Did you explain logically (not just say "it's wrong")?
- [ ] Did you verify your fix actually works?
- [ ] Were you confident and clear?

---

## 📋 DAYS 13-15: FINAL POLISH & REST

### DAY 13 - Knowledge Consolidation (REVIEW ONLY)

**Your Project Numbers** (flashcard style):
- Dataset: __________ samples
- Split: __% / __% / __%
- Best metric: __________
- Baseline: __________

**Your Failed Experiments** (3 required):
1. What: __________ → Expected: __________ → Got: __________ → Learned: __________
2. What: __________ → Expected: __________ → Got: __________ → Learned: __________
3. What: __________ → Expected: __________ → Got: __________ → Learned: __________

**Debugging Checklist** (3 critical bugs you know cold):
1. __________ (Obvious bug) — Fix: __________
2. __________ (Subtle bug) — Fix: __________
3. __________ (Pattern to avoid) — Fix: __________

**Paper Summary** (1-2 sentences):
- Problem: __________
- Solution: __________
- Result: __________

- [ ] Flashcard review (15 min)
- [ ] NO new material — consolidation only
- [ ] Can recite from memory? YES / NO

### DAY 14 - Full Mock Interview (2 hours)

**Part 1: Written Responses** (30 min)
- [ ] Write Section 1 answer: Project description
- [ ] Write Section 2 answer: Debugging experience
- [ ] Write Section 3 answer: Your numbers
- [ ] Write Section 4 answer: Reflection
- [ ] Write Section 5 answer: Paper review

**Part 2: Screen Share + Speaking** (30 min)
- [ ] Open GitHub repo, explain project (10 min)
- [ ] Describe failed experiment (5 min)
- [ ] Explain paper (5 min)
- [ ] Q&A on project (10 min)

**Part 3: Live Debugging** (40 min)
- [ ] Show unfamiliar buggy code (5 min reading)
- [ ] Explain bugs (10 min)
- [ ] Live fix on screen (10 min)
- [ ] Verify and run (10 min)
- [ ] Explain impacts (5 min)

**Part 4: Recording Review** (20 min)
- [ ] Watch recording
- [ ] Rate yourself 1-10 on: Clarity | Confidence | Technical Accuracy
- [ ] Note improvements needed

### DAY 15 - Rest + Final Review

- [ ] Light review only (30 min max)
  - [ ] Glance at your numbers one more time
  - [ ] Re-read your written answers
  - [ ] No new material
- [ ] Sleep well (8+ hours)
- [ ] Mental prep:
  - [ ] You've studied hard ✓
  - [ ] You know your material ✓
  - [ ] You can explain it clearly ✓
  - [ ] You're ready ✓
- [ ] Final prep:
  - [ ] GitHub link ready
  - [ ] Paper PDF downloaded
  - [ ] Colab/Notebook link accessible
  - [ ] Have water nearby
  - [ ] Clear workspace
  - [ ] Phone charged and positioned

---

## 🎯 FINAL CHECKLIST (Before Interview)

### Documents & Links
- [ ] GitHub URL (test click it 1 hour before interview)
- [ ] Colab/Notebook link (test it works)
- [ ] Paper PDF (download to disk, don't rely on internet)
- [ ] Your written answers (saved, printable)

### Workspace
- [ ] Desk clean and organized
- [ ] Phone positioned correctly (shows surroundings)
- [ ] Monitor at eye level
- [ ] Lighting good (no glare, no shadows)
- [ ] Background tidy
- [ ] No one else visible in frame

### Technical
- [ ] Internet speed tested (run speedtest.net)
- [ ] Zoom/Meet app updated
- [ ] Microphone and camera working
- [ ] Headphones charged (if using)
- [ ] 2 minutes before call: do quick audio/video test

### Mental
- [ ] You remember your project clearly? YES / NO
- [ ] You can explain your failed experiments? YES / NO
- [ ] You can identify bugs in code? YES / NO
- [ ] You can discuss your paper? YES / NO
- [ ] You feel confident? YES / NO (If NO, do Day 13 review again)

---

## 💪 YOU'VE GOT THIS!

Remember:
- **15 days is enough** if you focus
- **Authenticity is everything** — they detect ChatGPT instantly
- **Speaking clarity matters** — record yourself 5+ times
- **Your project is your proof** — know every detail
- **Bugs are learnings** — show you understand mistakes

Good luck! Check off tasks daily and trust the process. 🚀

