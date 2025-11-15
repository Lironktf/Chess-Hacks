# Modal Training - Updated Instructions (Modal 1.x)

## The Issue

Modal's API changed in version 1.x. The old `modal.Mount` no longer exists.

**New approach:** Upload data to Modal volumes, then train.

---

## Complete Workflow (3 Steps)

### **Step 1: Setup Modal** (One-time, 2 minutes)

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal setup
```

You get **30 free credits**!

---

### **Step 2: Upload Data** (One-time, 5-10 minutes)

```bash
# Create volume
modal volume create chess-data

# Upload your PGN files (4 files, 2.2 GB)
modal volume put chess-data ../data/*.pgn /
```

This uploads all 4 PGN files to Modal's persistent storage.

**Progress:**
```
Uploading lichess_elite_2021-01.pgn... ████████████ 477 MB
Uploading lichess_elite_2021-02.pgn... ████████████ 443 MB
Uploading lichess_elite_2021-04.pgn... ████████████ 596 MB
Uploading lichess_elite_2021-05.pgn... ████████████ 681 MB
✓ Uploaded 2.2 GB
```

---

### **Step 3: Train** (30-60 minutes)

```bash
modal run train_modal_fixed.py
```

This trains on Modal's T4 GPU using the uploaded data.

---

## Even Simpler: One-Click Script!

I created an automated script that does everything:

```bash
./modal_train.sh
```

This script:
1. ✅ Checks if Modal is installed and authenticated
2. ✅ Checks if data is already uploaded (skips if yes)
3. ✅ Uploads data if needed
4. ✅ Trains on GPU
5. ✅ Downloads the trained model
6. ✅ All in one command!

---

## Manual Commands (If You Prefer)

### First time only:

```bash
# Setup
pip install modal
modal setup

# Upload data
modal volume create chess-data
modal volume put chess-data ../data/*.pgn /
```

### Every time you train:

```bash
# Train
modal run train_modal_fixed.py

# Download model
modal volume get chess-models nnue_best.pth .

# Play
python3 nnue_play.py
```

---

## What Happens During Training

```bash
$ modal run train_modal_fixed.py

NNUE CHESS BOT - MODAL GPU TRAINING

⚠️  Make sure you uploaded data first:
    modal volume create chess-data
    modal volume put chess-data ../data/*.pgn /

This will train on Modal's T4 GPU (~30-60 min, ~$0.50)

Start training? (y/n): y

🚀 Training on Modal GPU...

======================================================================
 NNUE CHESS TRAINING ON MODAL GPU
======================================================================

✓ Device: cuda
✓ GPU: Tesla T4
✓ Memory: 15.0 GB

======================================================================
 LOADING DATA
======================================================================
✓ Found 4 PGN files
  • lichess_elite_2021-01.pgn: 477 MB
  • lichess_elite_2021-02.pgn: 443 MB
  • lichess_elite_2021-04.pgn: 596 MB
  • lichess_elite_2021-05.pgn: 681 MB

Loading lichess_elite_2021-01.pgn...
  Games: 100%|████████████| 10000/10000

✓ Loaded 38,542 games (2200+ ELO)

Creating positions...
Processing: 100%|████████████| 38542/38542

✓ 963,550 positions
  Wins: 435,301, Draws: 315,789, Loss: 212,460

✓ Train: 915,373, Val: 48,177

======================================================================
 MODEL
======================================================================
✓ Params: 1,895,233
✓ Size: 7.23 MB

======================================================================
 TRAINING
======================================================================

Epoch 1/50
Training: 100%|████████████| 894/894
  Train: 0.241234
  Val:   0.243891
  Time:  72.3s
  ✓ Saved!

Epoch 2/50
...

Epoch 50/50
  Train: 0.087234
  Val:   0.092123
  Time:  68.1s

======================================================================
 COMPLETE!
======================================================================
✓ Best: 0.092123
✓ Positions: 963,550

Download:
  modal volume get chess-models nnue_best.pth .

Play:
  python3 nnue_play.py
```

---

## Quick Reference

```bash
# One-click training (recommended)
./modal_train.sh

# OR manual steps:

# 1. Setup (once)
pip install modal
modal setup

# 2. Upload data (once)
modal volume create chess-data
modal volume put chess-data ../data/*.pgn /

# 3. Train
modal run train_modal_fixed.py

# 4. Download
modal volume get chess-models nnue_best.pth .

# 5. Play
python3 nnue_play.py
```

---

## Checking Upload Status

```bash
# List files in volume
modal volume ls chess-data

# Should show:
# lichess_elite_2021-01.pgn  477 MB
# lichess_elite_2021-02.pgn  443 MB
# lichess_elite_2021-04.pgn  596 MB
# lichess_elite_2021-05.pgn  681 MB
```

---

## Cost Breakdown

| Item | Cost | Time |
|------|------|------|
| **Data upload** | Free | 5-10 min |
| **Storage** | $0.10/GB/month | 2.2 GB = $0.22/month |
| **GPU training** | $0.50/hour | 0.5-1 hr = $0.25-0.50 |
| **TOTAL** | | **~$0.70** |

**You have 30 credits = ~40 training runs!**

---

## Troubleshooting

### "modal: command not found"
```bash
pip install modal
```

### "Not authenticated"
```bash
modal setup
```

### "Volume not found"
```bash
modal volume create chess-data
```

### Upload fails
Try uploading files one at a time:
```bash
modal volume put chess-data ../data/lichess_elite_2021-01.pgn /
modal volume put chess-data ../data/lichess_elite_2021-02.pgn /
# etc.
```

### Check if files are uploaded
```bash
modal volume ls chess-data
```

---

## Why the Change?

**Old way (doesn't work in Modal 1.x):**
```python
modal.Mount.from_local_dir(...)  # ❌ No longer exists
```

**New way (Modal 1.x):**
```bash
modal volume create chess-data         # Create storage
modal volume put chess-data files /    # Upload files
# Then access in Python via volumes
```

---

## Ready to Train!

**Easiest way:**
```bash
./modal_train.sh
```

**Or step-by-step:**
```bash
# Upload (once)
modal volume create chess-data
modal volume put chess-data ../data/*.pgn /

# Train
modal run train_modal_fixed.py
```

🚀 **30-60 minutes to a Master-level chess bot!**
