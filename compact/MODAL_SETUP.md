# Training on Modal - Complete Guide

## What is Modal?

Modal is a serverless platform that lets you run Python code in the cloud with **GPUs**. Perfect for training your chess bot!

**Benefits:**
- ✅ **GPU acceleration**: 10-20x faster than CPU (30-60 min vs 2-4 hours)
- ✅ **30 free credits**: More than enough for training
- ✅ **No local resources**: Your computer stays free
- ✅ **Easy to use**: Just Python, no complex setup

---

## Setup (5 Minutes)

### Step 1: Install Modal

```bash
pip install modal
```

### Step 2: Create Modal Account & Setup

```bash
modal setup
```

This will:
1. Open your browser to create a Modal account (free)
2. Generate an API token
3. Save credentials locally

**You get 30 free credits!** (Training costs ~$0.50-1.00)

### Step 3: Verify Setup

```bash
modal token show
```

Should show your token. You're ready!

---

## Training on Modal (2 Options)

### Option A: Upload Data First (Recommended for Multiple Runs)

**Step 1: Upload your PGN files to Modal**

```bash
cd /Users/lironkatsif/Desktop/SWE/Personal/ChessBot/compact

# Upload your 4 PGN files (~2.2 GB)
modal run upload_data_modal.py
```

This uploads your data to Modal's persistent storage (one-time ~5-10 min).

**Step 2: Train with GPU**

```bash
modal run train_modal.py
```

**What happens:**
- Runs on Modal's T4 GPU (free tier)
- Loads your uploaded data
- Trains for 50 epochs
- Takes ~30-60 minutes
- Costs ~$0.50-1.00
- Saves model to Modal storage

**Step 3: Download trained model**

```bash
modal volume get chess-models nnue_best.pth
```

This downloads `nnue_best.pth` to your current directory.

**Step 4: Play!**

```bash
python3 nnue_play.py
```

### Option B: Direct Training (Simpler, but uploads data each time)

If you only plan to train once, you can skip the upload step and use the simplified version.

---

## Cost Breakdown

**Modal Pricing (with 30 free credits):**

| Resource | Cost | Time | Total |
|----------|------|------|-------|
| T4 GPU | $0.50/hour | 0.5-1 hour | **~$0.50** |
| Storage | $0.10/GB/month | 2.2 GB | **~$0.20** |
| **TOTAL** | | | **~$0.70** |

**You have 30 credits, so you can train ~40 times! 🎉**

---

## What Happens During Training

```
→ modal run train_modal.py

Creating app...
✓ Initialized
✓ Created objects

Running app.main...

============================================================
TRAINING NNUE CHESS BOT ON MODAL
============================================================

This will upload your data and train on Modal's GPU

Training Configuration:
  Max games per file: 10,000
  Max files: 4
  Positions per game: 25
  Min ELO: 2200
  Batch size: 1024
  Epochs: 50

Estimated time with GPU: 30-60 minutes
Estimated cost: ~$0.50-1.00 (well within 30 free credits)

Start training? (y/n): y

Starting training on Modal...

Device: cuda
GPU: Tesla T4
GPU Memory: 15.0 GB

LOADING DATA FROM MODAL STORAGE
Found 4 PGN files
Loading PGN files: 100%|████████| 4/4 [02:30<00:00]
Loaded 38,542 high-quality games (ELO >= 2200)

Creating training data...
Processing games: 100%|████████| 38542/38542 [05:15<00:00]
Created 963,550 training positions

MODEL
Parameters: 1,895,233
Model size: 7.23 MB

TRAINING
Epoch 1/50
Training: 100%|████████| 918/918 [01:15<00:00]
  Train Loss: 0.241234
  Val Loss:   0.243891
  ✓ Saved best model

Epoch 2/50
...

Epoch 50/50
  Train Loss: 0.087234
  Val Loss:   0.092123

TRAINING COMPLETE
✓ Best model saved (val_loss: 0.092123)
✓ Models saved to Modal volume

Download with: modal volume get chess-models nnue_best.pth

============================================================
TRAINING COMPLETE!
============================================================
```

---

## Monitoring Training

### View logs in real-time:

While training is running, you can monitor in the Modal dashboard:

```
https://modal.com/apps
```

### Check GPU usage:

Modal will show:
- GPU utilization
- Memory usage
- Training progress
- Estimated completion time

---

## Downloading the Model

After training completes:

```bash
# Download the best model
modal volume get chess-models nnue_best.pth

# Check the file
ls -lh nnue_best.pth
# Should be ~7.2 MB
```

Move it to your compact directory if needed:

```bash
mv nnue_best.pth /Users/lironkatsif/Desktop/SWE/Personal/ChessBot/compact/
```

---

## Troubleshooting

### "modal: command not found"

Install Modal:
```bash
pip install modal
```

### "You are not authenticated"

Run setup:
```bash
modal setup
```

### "Volume not found"

The volume is created automatically on first use. Just run again.

### "Out of memory"

Reduce batch size in `train_modal.py`:
```python
batch_size=512  # Instead of 1024
```

### "Exceeded free tier"

You used your 30 credits. Add a payment method or reduce training time.

---

## Advanced: Adjust Training Settings

Edit `train_modal.py` and change the parameters in `main()`:

```python
result = train_on_modal.remote(
    max_games_per_file=10000,  # More games = stronger bot
    max_files=4,               # Use all 4 files
    positions_per_game=25,     # Positions sampled per game
    min_elo=2200,              # Minimum ELO filter
    batch_size=1024,           # GPU batch size
    num_epochs=50,             # Training epochs
)
```

**For maximum strength (uses more credits):**
```python
max_games_per_file=25000,  # 100K games total
num_epochs=100,            # More training
min_elo=2400,              # Only elite games
```

**For quick test (cheaper):**
```python
max_games_per_file=2000,   # 8K games
num_epochs=30,             # Faster
```

---

## Comparing: Local vs Modal

| Aspect | Local (CPU) | Modal (GPU) |
|--------|-------------|-------------|
| **Time** | 2-4 hours | 30-60 min |
| **Cost** | Free | ~$0.70 |
| **GPU** | No | Yes (T4) |
| **Your computer** | Busy | Free |
| **Setup** | None | 5 min |
| **Credits** | N/A | 30 free |

**Recommendation:** Use Modal! 10-20x faster and you have free credits.

---

## Quick Commands Reference

```bash
# Setup (one-time)
pip install modal
modal setup

# Upload data (one-time, optional)
modal run upload_data_modal.py

# Train (main command)
modal run train_modal.py

# Download model
modal volume get chess-models nnue_best.pth

# Play
python3 nnue_play.py
```

---

## Next Steps After Training

1. **Verify size:**
   ```bash
   python3 size_analysis.py
   ```

2. **Test it:**
   ```bash
   python3 nnue_test.py
   ```

3. **Play:**
   ```bash
   python3 nnue_play.py
   ```

4. **Compete!** 🏆

---

## FAQ

**Q: How much do 30 free credits last?**
A: You can train ~40 times! Each training costs ~$0.70.

**Q: Do I need to re-upload data each time?**
A: No! Once uploaded to Modal volume, it persists. Just run `train_modal.py`.

**Q: Can I use a different GPU?**
A: Yes! Edit `train_modal.py` and change `gpu="T4"` to `gpu="A10G"` (faster but costs more).

**Q: What if training fails?**
A: Check logs in Modal dashboard. Common issues:
- Out of memory: Reduce batch size
- Timeout: Increase timeout in decorator
- Data not found: Upload data first

**Q: How do I check my credit balance?**
A: Visit https://modal.com/settings/billing

---

**You're ready to train on Modal with GPU! 🚀**

30-60 minutes to a Master-level chess bot!
