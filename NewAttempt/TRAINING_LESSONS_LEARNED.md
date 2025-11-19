# Training Lessons Learned

## Summary of Issues Encountered and Solutions

This document captures the key lessons learned during the development and training of this chess engine.

---

## 1. Modal API Syntax

### Problem
Initial Modal commands used positional arguments, which aren't supported:
```bash
# WRONG - doesn't work
modal run train_modal.py generate data.pgn 10000 15
```

### Solution
Use flag-based arguments with `--detach` for background execution:
```bash
# CORRECT
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename data.pgn \
  --num-positions 10000 \
  --stockfish-depth 15
```

**Key points:**
- `--detach` lets the job run in background (you can close terminal)
- All parameters must use `--flag-name value` syntax
- Function name comes after `::` (e.g., `::generate_data`)

---

## 2. Modal File Mounting

### Problem
Multiple failed attempts to mount local files into Modal containers:
- `modal.Mount.from_local_dir()` doesn't exist
- `modal.mount.Mount.from_local_dir()` also doesn't work
- Files couldn't be found in container

### Solution
Use `Image.add_local_dir()` to bake files into the Docker image:
```python
LOCAL_DATA_PATH = Path(__file__).parent / "data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "python-chess", "tqdm")
    .apt_install("stockfish")
    .add_local_dir(LOCAL_DATA_PATH, remote_path="/pgn")  # This works!
)
```

**Key points:**
- Files are copied into the image at build time
- They appear at `/pgn` inside the container
- No need for volume mounting for read-only data

---

## 3. PyTorch Dtype Mismatch

### Problem
Training crashed with:
```
RuntimeError: Found dtype Double but expected Float
```

### Solution
Explicitly convert targets to float32:
```python
# WRONG
target = np.clip(item['score'] / 100.0, -100, 100)

# CORRECT
target = np.clip(item['score'] / 100.0, -100, 100).astype(np.float32)
```

**Key points:**
- NumPy defaults to float64 (double)
- PyTorch models use float32
- Always specify dtype explicitly

---

## 4. Training Data Quality (CRITICAL LESSON)

### Problem
Used Lichess puzzle database for training, which caused severe overfitting:

```
Epoch 1/5: Train Loss: 112.1, Val Loss: 112.8
Epoch 2/5: Train Loss: 106.9, Val Loss: 108.8
Epoch 3/5: Train Loss: 96.5, Val Loss: 109.2  ← Validation going UP!
Epoch 4/5: Train Loss: 87.3, Val Loss: 110.2
Epoch 5/5: Train Loss: 82.5, Val Loss: 110.6
```

Training loss dropped (112→82) but validation loss stayed high (~110) - classic overfitting.

### Root Cause

**Puzzle ratings ≠ Position evaluations!**

- **Puzzle rating**: Measures how difficult the tactic is to find
  - Rating 1500 = easy puzzle (beginner can solve it)
  - Rating 2500 = hard puzzle (master level required)

- **Position evaluation**: Measures who's winning and by how much
  - +100 cp = white is up ~1 pawn
  - -300 cp = black is winning by ~3 pawns

A position can be:
- Equal (0 cp) but have a hard tactic (2500 puzzle rating)
- Winning (+500 cp) but have an obvious move (1200 puzzle rating)

The model tried to learn puzzle difficulty instead of position strength!

### Solution

Use **Stockfish evaluations** from real game positions:

```bash
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15
```

**Expected good training results:**
```
Epoch 1/5: Train Loss: 45.2, Val Loss: 46.8
Epoch 2/5: Train Loss: 38.1, Val Loss: 39.4 ✓ Saved
Epoch 3/5: Train Loss: 34.5, Val Loss: 36.2 ✓ Saved
Epoch 4/5: Train Loss: 32.1, Val Loss: 35.8 ✓ Saved
Epoch 5/5: Train Loss: 30.8, Val Loss: 35.9
```
Both losses decrease together - the model is learning!

---

## 5. Cost vs Quality Tradeoffs

### Original Plan
- 50k positions
- Stockfish depth 18
- Cost: ~$20
- Time: 10+ hours

**Problem:** Too expensive and slow for testing

### Optimized Recommendation
- 10k positions
- Stockfish depth 15
- Cost: ~$3
- Time: ~2 hours

**Why this works:**
- Depth 15 is still very accurate (depth 18 only marginally better)
- 10k positions is enough to train a decent NNUE
- 3x faster and 7x cheaper
- Good for iteration and testing

### When to Use Full Dataset
Only use 50k @ depth 18 if:
- You've tested with 10k and it works
- You want maximum strength
- You don't mind the cost/time

---

## 6. Modal Volume Management

### Key Commands

```bash
# List files in Modal volume
modal volume ls chessbot-data

# Upload file to volume
modal volume put chessbot-data local_file.json remote_file.json

# Download file from volume
modal volume get chessbot-data model.pt ./model.pt

# Check running jobs
modal app list

# View logs (Ctrl+C to exit, job keeps running)
modal app logs chessbot-nnue-training

# Stop a job
modal app stop chessbot-nnue-training
```

**Important:**
- Volume data persists between runs
- `volume.commit()` required to save changes
- Files in volume are at `/data` path in container

---

## 7. Documentation Evolution

### Files Removed (Duplicates)
- `CORRECT_COMMANDS.md`
- `HOW_TO_TRAIN.md`
- `START_HERE.md`
- `TRAINING_GUIDE.md`

### Files Kept
- `README.md` - Project overview
- `RUN_THIS.md` - Training workflow (updated with correct commands)
- `QUICKSTART.md` - Usage examples
- `TEACHER.md` - Educational deep dive
- `BETTER_DATA_SOURCES.md` - Data quality guide
- `TRAINING_LESSONS_LEARNED.md` - This file!

---

## Key Takeaways

1. **Data quality matters more than data quantity**
   - 10k good positions > 50k puzzle ratings
   - Always use Stockfish evaluations for training

2. **Watch validation loss, not just training loss**
   - If val loss doesn't decrease, you're overfitting
   - Train and val should move together

3. **Modal syntax is picky**
   - Use `--flag-name value` format
   - Use `--detach` for background jobs
   - Use `Image.add_local_dir()` for local files

4. **Start small, then scale**
   - Test with 10k @ depth 15 first
   - Only go to 50k @ depth 18 if needed

5. **Explicit is better than implicit**
   - Always specify `.astype(np.float32)`
   - Don't rely on automatic type conversion

---

## Recommended Workflow

```bash
# 1. Generate quality training data (~2 hours, ~$3)
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15

# 2. Train the model (~30 minutes, ~$0.50)
modal run --detach train_modal_simple.py::train_model \
  --data-file training_data.json \
  --epochs 5 \
  --batch-size 4096

# 3. Download and convert
modal volume get chessbot-data model.pt ./model.pt
python3 -c "from utils import convert_torch_to_numpy; convert_torch_to_numpy('model.pt', 'model.npz')"

# 4. Test it
python3 test_engine.py
```

Total cost: ~$3.50
Total time: ~2.5 hours
Result: A properly trained chess engine!
