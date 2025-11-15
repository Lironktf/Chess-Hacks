# Complete Training Guide - NNUE Chess Bot

## 📥 Step 1: Get Training Data

### Option A: Lichess Elite Database (RECOMMENDED) ⭐

**Best source for high-quality games!**

1. **Visit Lichess Database:**
   ```
   https://database.lichess.org/
   ```

2. **Download a recent month** (example: January 2024)
   ```bash
   # In your terminal:
   cd ~/Desktop/SWE/Personal/ChessBot/data

   # Download (pick one month - files are large!)
   wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
   ```

3. **Decompress the file:**
   ```bash
   # Install zstd if you don't have it
   brew install zstd   # macOS
   # or: sudo apt install zstd   # Linux

   # Decompress (this creates a .pgn file)
   zstd -d lichess_db_standard_rated_2024-01.pgn.zst

   # This creates: lichess_db_standard_rated_2024-01.pgn
   ```

4. **Filter for high-ELO games (IMPORTANT!):**

   The full file has millions of games. We only want high-quality ones.

   ```bash
   # Use this Python script to filter:
   python3 filter_pgn.py lichess_db_standard_rated_2024-01.pgn
   ```

### Option B: Smaller Test Dataset (Quick Start)

If you just want to test training quickly:

1. **Download a smaller PGN collection:**
   ```
   https://www.pgnmentor.com/files.html
   ```

2. **Look for "Master Games" or "Elite Players"**
   - Download 5-10 files
   - Each should have 1000-5000 games
   - Save to `../data/` directory

### Option C: CCRL Computer Games

High-quality computer chess games:

1. **Visit:**
   ```
   https://computerchess.org.uk/ccrl/404/games.html
   ```

2. **Download PGN files**
   - These are very high quality (computer vs computer)
   - Save to `../data/` directory

---

## 📁 Step 2: Set Up Data Directory

### Create the data folder:

```bash
# From the compact directory:
cd ~/Desktop/SWE/Personal/ChessBot

# Create data directory
mkdir -p data

# Check it exists
ls -la
```

You should see:
```
ChessBot/
├── data/           ← PUT YOUR PGN FILES HERE!
│   ├── game1.pgn
│   ├── game2.pgn
│   └── ...
└── compact/
    ├── nnue_train.py
    └── ...
```

### Put your PGN files in the data folder:

```bash
# Move/copy your PGN files
mv ~/Downloads/*.pgn ~/Desktop/SWE/Personal/ChessBot/data/

# OR if you downloaded and decompressed Lichess data:
mv lichess_db_standard_rated_2024-01.pgn ~/Desktop/SWE/Personal/ChessBot/data/

# Verify files are there
ls ~/Desktop/SWE/Personal/ChessBot/data/*.pgn
```

---

## 🔍 Step 3: Filter PGN Files (IMPORTANT!)

The training script filters for 2200+ ELO, but pre-filtering saves time.

Create this helper script:

```bash
cd ~/Desktop/SWE/Personal/ChessBot/compact
```

**Filter your PGN files:**

```bash
# Filter a large PGN file to only include high-quality games
python3 filter_pgn.py ../data/lichess_db_standard_rated_2024-01.pgn \
                      ../data/filtered_games.pgn \
                      --min-elo 2200 \
                      --max-games 50000 \
                      --time-control classical

# This will:
# - Keep only 2200+ ELO games
# - Limit to 50,000 games (adjust as needed)
# - Only classical time control (no blitz/bullet)
# - Skip aborted/timeout games
```

**Or use multiple smaller files (easier):**

If you have several smaller PGN files, you don't need to filter:

```bash
# Just copy them to data/
cp ~/Downloads/*.pgn ../data/

# The training script will filter them automatically
```

---

## 🚀 Step 4: Run Training!

### Quick Training (Test - ~30 minutes)

For testing the pipeline:

```bash
cd ~/Desktop/SWE/Personal/ChessBot/compact

python3 nnue_train.py
```

This will use default settings:
- **Games:** 2,000 per file × 20 files = 40,000 games
- **Min ELO:** 2200+
- **Positions:** ~25 per game = 1,000,000 positions
- **Epochs:** 50
- **Time:** 2-4 hours (CPU), 30-60 min (GPU)

### Full Training (Maximum Strength)

For best results, edit `nnue_train.py` first:

```bash
# Open the file
nano nnue_train.py   # or use your favorite editor
```

**Adjust these settings** (around line 225):

```python
# For MAXIMUM strength (if you have lots of data):
MAX_GAMES_PER_FILE = 5000    # More games per file
MAX_FILES = 50               # Use more files
POSITIONS_PER_GAME = 30      # More positions per game
MIN_ELO = 2400               # Even higher quality
NUM_EPOCHS = 100             # More training epochs

# For FASTER training (if limited data):
MAX_GAMES_PER_FILE = 1000
MAX_FILES = 10
POSITIONS_PER_GAME = 20
MIN_ELO = 2000
NUM_EPOCHS = 30
```

**Then run:**

```bash
python3 nnue_train.py
```

---

## 📊 What Happens During Training

### Phase 1: Data Loading (5-30 minutes)

```
Loading PGN files: 100%|████████| 20/20 [15:30<00:00]
Loaded 42,153 high-quality games (ELO >= 2200)

Creating training data...
Processing games: 100%|████████| 42153/42153 [08:45<00:00]
Created 1,053,825 training positions

Result distribution:
  White wins: 425,301
  Draws: 315,789
  Black wins: 312,735

Training samples: 1,001,134
Validation samples: 52,691
```

### Phase 2: Model Training (1-3 hours)

```
Using device: cpu

NNUE Model:
  Parameters: 1,895,233
  Size: 7.23 MB

Training:
Epoch 1/50
Training: 100%|████████| 977/977 [02:15<00:00]

  Train Loss: 0.245123
  Val Loss:   0.247891
  Time: 142.3s
  ✓ Saved best model

Epoch 2/50
Training: 100%|████████| 977/977 [02:12<00:00]

  Train Loss: 0.231456
  Val Loss:   0.235789
  Time: 138.7s
  ✓ Saved best model

...

Epoch 50/50
  Train Loss: 0.089234
  Val Loss:   0.095123
  Time: 135.2s
```

### Phase 3: Complete!

```
TRAINING COMPLETE
✓ Best model: nnue_best.pth (val_loss: 0.091234)
✓ Final model: nnue_final.pth

Use 'nnue_play.py' to play against the bot!
```

---

## 🎮 Step 5: Test Your Trained Model

### Check the model size:

```bash
ls -lh nnue_best.pth

# Should show: ~7.2 MB
```

### Run verification:

```bash
python3 size_analysis.py
```

Expected output:
```
TOTAL: 9.94 MB
✅ UNDER 10MB LIMIT!
```

### Play against it!

```bash
python3 nnue_play.py

# Choose difficulty (depth 5 recommended)
# Play and see how strong it is!
```

---

## 💡 Training Tips

### Data Quality Matters Most! ⭐

| ELO Range | Quality | Recommended |
|-----------|---------|-------------|
| 2400+ | Excellent | ✅ Best |
| 2200-2400 | Very Good | ✅ Good |
| 2000-2200 | Good | ⚠️ OK for mixing |
| < 2000 | Poor | ❌ Skip |

**Use only 2200+ games for best results!**

### How Much Data?

| Games | Positions | Quality | Training Time |
|-------|-----------|---------|---------------|
| 10,000 | 250K | Basic | 30 min |
| 50,000 | 1.25M | Good | 2 hours |
| 100,000 | 2.5M | Very Good | 4 hours |
| 500,000+ | 12.5M+ | Excellent | 12+ hours |

**Recommended: 50,000-100,000 high-quality games**

### Time Control Matters

✅ **Classical** (30+ min): Best quality, players think deeply
✅ **Rapid** (10-30 min): Good quality, faster to get data
⚠️ **Blitz** (3-10 min): Lower quality, more mistakes
❌ **Bullet** (< 3 min): Poor quality, many blunders

**Use classical games for best results!**

### Balanced Dataset

Your dataset should have:
- **45-50%** White wins
- **25-30%** Draws
- **20-25%** Black wins

This is natural for high-level chess.

---

## ⚙️ Advanced: Training Configuration

### Hyperparameters (in `nnue_train.py`)

```python
# Model architecture
HIDDEN1_SIZE = 1408        # Larger = stronger but bigger
HIDDEN2_SIZE = 32          # Keep at 32

# Training
BATCH_SIZE = 1024          # Larger = faster (if enough RAM)
LEARNING_RATE = 0.001      # 0.001 is good default
NUM_EPOCHS = 50            # More = better (if not overfitting)

# Data
POSITIONS_PER_GAME = 25    # More = more data, slower loading
MIN_ELO = 2200             # Higher = better quality
```

### GPU vs CPU Training

**With GPU (NVIDIA/CUDA):**
```bash
# Training time: 30-60 minutes
# Much faster!

# Check if GPU available:
python3 -c "import torch; print(torch.cuda.is_available())"
```

**With CPU only:**
```bash
# Training time: 2-4 hours
# Slower but works fine

# The script automatically uses CPU if no GPU
```

### Monitor Training

Watch for:
- **Val loss decreasing** ✅ Good! Model is learning
- **Val loss increasing** ❌ Overfitting! Stop early
- **Val loss plateaus** ⏸️ Training complete

**Early stopping:** If val loss doesn't improve for 10 epochs, stop training.

---

## 🐛 Troubleshooting

### "No PGN files found"

```bash
# Check data directory exists:
ls ../data/*.pgn

# If empty:
cd ../data
wget [URL to PGN file]
```

### "Out of memory"

Reduce batch size in `nnue_train.py`:

```python
BATCH_SIZE = 512   # Instead of 1024
# or even
BATCH_SIZE = 256
```

### "Training too slow"

Reduce data:

```python
MAX_GAMES_PER_FILE = 1000   # Instead of 2000
MAX_FILES = 10              # Instead of 20
```

### "Model plays poorly after training"

Possible issues:
1. **Low-quality data**: Use only 2200+ ELO
2. **Too little data**: Need at least 20,000 games
3. **Overfitting**: Reduce epochs or increase data
4. **Wrong time control**: Use classical, not blitz

### "Val loss not decreasing"

Try:
- Increase data size
- Decrease learning rate to 0.0005
- Add more epochs
- Check data quality

---

## 📈 Expected Results

### After Training on Good Data (50K games, 2200+ ELO):

| Depth | Strength | Description |
|-------|----------|-------------|
| 3 | 1600-1700 | Decent club player |
| 4 | 1800-1900 | Strong club player |
| 5 | 2000-2100 | Expert level ⭐ |
| 6 | 2200-2300 | Master level |
| 7 | 2400+ | International Master |

### Validation Loss Guide:

- **< 0.08**: Excellent! Very strong model
- **0.08-0.12**: Good! Strong model
- **0.12-0.20**: Decent, but could be better
- **> 0.20**: Poor quality, need better data

---

## 🎯 Quick Start Checklist

- [ ] Install dependencies: `pip install torch chess numpy tqdm`
- [ ] Create data directory: `mkdir -p ../data`
- [ ] Download PGN files (Lichess/CCRL/etc.)
- [ ] Put PGN files in `../data/` folder
- [ ] (Optional) Filter with `filter_pgn.py`
- [ ] Run training: `python3 nnue_train.py`
- [ ] Wait 2-4 hours (or 30-60 min with GPU)
- [ ] Test model: `python3 nnue_play.py`
- [ ] Verify size: `python3 size_analysis.py`
- [ ] Compete and win! 🏆

---

## 📚 Data Sources Summary

### Best Sources:

1. **Lichess Elite** (https://database.lichess.org/)
   - Millions of games
   - 2200+ ELO available
   - Classical time control
   - **RECOMMENDED** ⭐

2. **CCRL** (https://computerchess.org.uk/ccrl/)
   - Computer games (very high quality)
   - Strong play
   - Limited quantity

3. **PGN Mentor** (https://www.pgnmentor.com/)
   - Master games
   - Well-organized
   - Smaller files (good for testing)

### Where Files Go:

```
Your Project:
ChessBot/
├── data/                    ← PUT PGN FILES HERE!
│   ├── lichess_filtered.pgn
│   ├── masters_2024.pgn
│   └── ccrl_games.pgn
└── compact/
    └── nnue_train.py        ← RUN TRAINING FROM HERE
```

---

## 🚀 Ready to Train!

**Recommended path for beginners:**

```bash
# 1. Download small test dataset
cd ~/Desktop/SWE/Personal/ChessBot/data
wget https://www.pgnmentor.com/files/WorldChampions.zip
unzip WorldChampions.zip

# 2. Run training (quick test)
cd ../compact
python3 nnue_train.py

# 3. Play!
python3 nnue_play.py
```

**For serious training (max strength):**

```bash
# 1. Download Lichess database
cd ~/Desktop/SWE/Personal/ChessBot/data
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
zstd -d lichess_db_standard_rated_2024-01.pgn.zst

# 2. Filter to high-quality games
cd ../compact
python3 filter_pgn.py ../data/lichess_db_standard_rated_2024-01.pgn \
                      ../data/elite_games.pgn \
                      --min-elo 2400 \
                      --max-games 100000

# 3. Train (this will take 3-4 hours on CPU)
python3 nnue_train.py

# 4. Compete and win!
python3 nnue_play.py
```

---

**Good luck training your Master-level chess bot!** 🏆
