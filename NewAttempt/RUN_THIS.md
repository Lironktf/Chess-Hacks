# 🚀 CORRECT COMMANDS - Run These!

## ⚠️ IMPORTANT: Training Data Quality

**The quality of your training data determines the quality of your model!**

There are two main options:

### Option A: Stockfish Labels (RECOMMENDED - High Quality)
- Real position evaluations
- Takes time but produces a strong model
- Best for serious training

### Option B: Lichess Puzzles (Fast but NOT Recommended)
- Very fast to download
- **⚠️ WARNING: Causes overfitting!**
- Puzzle ratings measure difficulty, not position strength
- Model won't generalize well

**Recommendation:** Use Option A (Stockfish) with 10k positions at depth 15 for best cost/quality balance.

---

## ✅ Step 1: Generate Training Data (Option A - Stockfish)

### Recommended: 10k positions, depth 15 (~2 hours, ~$3)

```bash
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15
```

**This will:**
- Run in detached mode (you can close terminal and go to bed!)
- Extract 10,000 positions from your PGN
- Label each with Stockfish depth 15
- Take ~2 hours
- Cost ~$3 on Modal
- Save to Modal volume as `training_data.json`

**After running this, you can close your terminal and go to sleep!**

### Alternative: Full Dataset (50k positions, depth 18)

Only use this if you want maximum quality and don't mind the cost:

```bash
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 50000 \
  --stockfish-depth 18
```

⚠️ This takes 10+ hours and costs ~$20!

---

## 📊 Check Progress (Next Morning)

```bash
# View logs
modal app logs chessbot-nnue-training

# Check if data file exists
modal volume ls chessbot-data
```

You should see `training_data.json` when it's done.

---

## ✅ Step 2: Train Model (Also Runs in Background!)

```bash
modal run --detach train_modal_simple.py::train_model \
  --data-file training_data.json \
  --epochs 5 \
  --batch-size 4096 \
  --learning-rate 0.001
```

**This will:**
- Run in detached mode (you can close terminal!)
- Train on T4 GPU
- Take ~20-30 minutes
- Save best model to Modal volume as `model.pt`

**You can also close terminal after this starts!**

### Expected Training Results

**Good training looks like this:**
```
Epoch 1/5: Train Loss: 45.2, Val Loss: 46.8
Epoch 2/5: Train Loss: 38.1, Val Loss: 39.4 ✓ Saved
Epoch 3/5: Train Loss: 34.5, Val Loss: 36.2 ✓ Saved
Epoch 4/5: Train Loss: 32.1, Val Loss: 35.8 ✓ Saved
Epoch 5/5: Train Loss: 30.8, Val Loss: 35.9
```
Both train and validation loss should decrease together.

**Bad training (overfitting) looks like this:**
```
Epoch 1/5: Train Loss: 112.1, Val Loss: 112.8
Epoch 2/5: Train Loss: 82.5, Val Loss: 110.6
```
Train loss drops but validation loss stays high or increases - this means your data quality is poor!

---

## 📥 Step 3: Download Your Model

```bash
# Download the trained model
modal volume get chessbot-data model.pt ./model.pt

# Convert to NumPy for faster inference
python3 -c "from utils import convert_torch_to_numpy; convert_torch_to_numpy('model.pt', 'model.npz')"
```

---

## 🎮 Step 4: Use Your Trained Model!

```python
from engine import ChessEngine

# Load your trained model
engine = ChessEngine()
engine.load_weights('model.npz')

# Test it
move, score, info = engine.get_best_move(depth=6)
print(f"Best move: {engine.board.san(move)}")
print(f"Score: {score:+d} centipawns")
print(f"Nodes: {info['nodes']:,}")
```

---

## 🔧 Useful Commands

```bash
# Check what's running
modal app list

# View logs (Ctrl+C to exit, job keeps running)
modal app logs chessbot-nnue-training

# List files on Modal
modal volume ls chessbot-data

# Stop a job (if needed)
modal app stop chessbot-nnue-training

# Download a file
modal volume get chessbot-data <filename> ./<local_filename>
```

---

## 🎯 TL;DR - Copy & Paste These Commands (RECOMMENDED)

```bash
# 1. Start data generation (then go to bed!) - 10k positions, depth 15
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15

# 2. Next morning: Start training (then go to work!)
modal run --detach train_modal_simple.py::train_model \
  --data-file training_data.json \
  --epochs 5 \
  --batch-size 4096

# 3. When done: Download model
modal volume get chessbot-data model.pt ./model.pt
python3 -c "from utils import convert_torch_to_numpy; convert_torch_to_numpy('model.pt', 'model.npz')"
```

---

## 💰 Cost Estimate (Updated)

**RECOMMENDED: 10k positions, depth 15, 5 epochs:**
- Data generation: ~$3
- Training: ~$0.50
- **Total: ~$3.50**
- **Time: ~2.5 hours total**

**Maximum Quality: 50k positions, depth 18, 5 epochs:**
- Data generation: ~$20
- Training: ~$1
- **Total: ~$21**
- **Time: 10+ hours**

---

## ❌ What NOT to Do

### Don't Use Lichess Puzzles for Training

```bash
# DON'T DO THIS - It causes overfitting!
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
python3 download_data.py convert
```

**Why this fails:**
- Puzzle ratings measure "How hard is this tactic?" not "Who's winning?"
- A position rated 1500 might be +5 for white, or equal, or losing!
- Your model will learn the wrong thing and overfit badly
- Validation loss will stay high while training loss drops

**Real example from testing:**
```
Epoch 1/5: Train Loss: 112.1, Val Loss: 112.8
Epoch 2/5: Train Loss: 106.9, Val Loss: 108.8
Epoch 3/5: Train Loss: 96.5, Val Loss: 109.2  ← Val loss going UP!
Epoch 4/5: Train Loss: 87.3, Val Loss: 110.2
Epoch 5/5: Train Loss: 82.5, Val Loss: 110.6
```

This is textbook overfitting. The model memorized puzzle difficulty patterns but didn't learn to evaluate positions.

---

Good luck! 🎉
