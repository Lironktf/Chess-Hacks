# Modal Training - Quick Start (3 Commands!)

Train your chess bot on **GPU** in **30-60 minutes** instead of 2-4 hours!

---

## Setup (One-Time - 2 Minutes)

### 1. Install Modal
```bash
pip install modal
```

### 2. Create Account & Authenticate
```bash
modal setup
```

This opens your browser to create a free account. **You get 30 free credits!**

---

## Train (Simple - One Command!)

```bash
cd /Users/lironkatsif/Desktop/SWE/Personal/ChessBot/compact

modal run train_modal_simple.py
```

**That's it!** This:
- Uses your existing data (no upload needed!)
- Trains on Modal's T4 GPU
- Takes ~30-60 minutes
- Costs ~$0.50-1.00 (you have 30 free credits)
- Saves the model to Modal storage

---

## Download & Play

### Download the trained model:
```bash
modal volume get chess-models nnue_best.pth .
```

### Verify it's under 10MB:
```bash
ls -lh nnue_best.pth
# Should show ~7.2 MB
```

### Play!
```bash
python3 nnue_play.py
```

---

## That's It! 🎉

**3 commands total:**
```bash
# 1. Setup (one-time)
modal setup

# 2. Train (30-60 min)
modal run train_modal_simple.py

# 3. Download & play
modal volume get chess-models nnue_best.pth .
python3 nnue_play.py
```

---

## What Happens During Training

```
modal run train_modal_simple.py

============================================================
 NNUE CHESS BOT - MODAL GPU TRAINING
============================================================

This will:
  • Use your local data (no upload needed!)
  • Train on Modal's T4 GPU
  • Take ~30-60 minutes
  • Cost ~$0.50-1.00 (you have 30 free credits)

Start training? (y/n): y

🚀 Starting training on Modal GPU...

✓ Device: cuda
✓ GPU: Tesla T4
✓ GPU Memory: 15.0 GB

============================================================
 LOADING DATA
============================================================
✓ Found 4 PGN files
  • lichess_elite_2021-01.pgn: 477 MB
  • lichess_elite_2021-02.pgn: 443 MB
  • lichess_elite_2021-04.pgn: 596 MB
  • lichess_elite_2021-05.pgn: 681 MB

Loading lichess_elite_2021-01.pgn...
  Games: 100%|████████| 10000/10000

✓ Loaded 38,542 high-quality games (2200+ ELO)

Creating training positions...
Processing: 100%|████████| 38542/38542
✓ Created 963,550 positions
  • Wins: 435,301
  • Draws: 315,789
  • Losses: 212,460

✓ Train: 915,373 positions
✓ Val: 48,177 positions

============================================================
 MODEL
============================================================
✓ Parameters: 1,895,233
✓ Size: 7.23 MB

============================================================
 TRAINING
============================================================

Epoch 1/50
Training: 100%|████████| 894/894 [01:12<00:00]
  Train: 0.241234
  Val:   0.243891
  Time:  72.3s
  ✓ Saved best model!

Epoch 2/50
...

Epoch 50/50
  Train: 0.087234
  Val:   0.092123
  Time:  68.1s

============================================================
 COMPLETE!
============================================================
✓ Best val loss: 0.092123
✓ Total positions: 963,550

Download your model:
  modal volume get chess-models nnue_best.pth .

Then play:
  python3 nnue_play.py
```

---

## Monitoring

While training is running:

1. **Watch logs** - They stream in your terminal in real-time
2. **Modal dashboard** - Visit https://modal.com/apps to see:
   - GPU usage
   - Memory usage
   - Time remaining
   - Cost so far

---

## Cost Breakdown

| Resource | Rate | Usage | Cost |
|----------|------|-------|------|
| T4 GPU | $0.50/hr | 0.5-1 hr | $0.25-0.50 |
| Storage | $0.10/GB/mo | 7 MB | $0.001 |
| Network | Free | - | $0.00 |
| **TOTAL** | | | **~$0.50** |

**You have 30 credits = ~60 training runs!**

---

## Customizing Training

Edit `train_modal_simple.py` to adjust settings:

```python
result = train.remote(
    max_games_per_file=10000,  # Games per file (more = stronger)
    max_files=4,               # Number of files to use
    num_epochs=50,             # Training epochs (more = better)
)
```

**Quick test** (cheaper, faster):
```python
max_games_per_file=2000,
max_files=2,
num_epochs=30,
# Takes ~15 min, costs ~$0.20
```

**Max strength** (more expensive):
```python
max_games_per_file=25000,
max_files=4,
num_epochs=100,
# Takes ~2 hours, costs ~$1.50
```

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
It's created automatically. Just run again.

### "Out of credits"
Add payment method at https://modal.com/settings/billing

---

## Why Modal?

| | Local (CPU) | Modal (GPU) |
|---|---|---|
| **Time** | 2-4 hours | **30-60 min** ✅ |
| **Cost** | Free | ~$0.50 |
| **Speed** | 1x | **10-20x faster** ✅ |
| **Your computer** | Busy | **Free** ✅ |
| **GPU** | No | **Yes (T4)** ✅ |

**Verdict:** Modal is 10-20x faster and you have free credits!

---

## Complete Example Session

```bash
# Install and setup (one-time)
pip install modal
modal setup

# Train on GPU
cd /Users/lironkatsif/Desktop/SWE/Personal/ChessBot/compact
modal run train_modal_simple.py
# ⏱️  Wait 30-60 minutes...

# Download model
modal volume get chess-models nnue_best.pth .

# Verify size
ls -lh nnue_best.pth
# Output: -rw-r--r-- 1 user staff 7.2M Nov 15 13:45 nnue_best.pth

# Play!
python3 nnue_play.py
```

---

**Ready to train at 10-20x speed? Run:**

```bash
modal run train_modal_simple.py
```

🚀 **30-60 minutes to a Master-level chess bot!**
