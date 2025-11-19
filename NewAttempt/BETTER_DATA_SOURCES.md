# Better Training Data Sources

## Problem with Lichess Puzzles

Puzzle ratings ≠ Position evaluations!
- Puzzle rating = How hard it is to find the tactic
- Position evaluation = Who's winning and by how much

This causes overfitting (your current issue).

## Better Alternatives

### Option 1: Generate with Stockfish (Best Quality) ✅ RECOMMENDED

```bash
# 10k positions, depth 15, ~2 hours, ~$3
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15
```

**Pros:**
- Real evaluations that actually reflect position strength
- Custom depth control
- Best for learning chess evaluation
- Proven to work (validation loss decreases properly)

**Cons:**
- Takes time (~2 hours)
- Costs money on Modal (~$3)

**Why this works:**
Stockfish evaluations measure "who's winning and by how much" - exactly what your NNUE needs to learn!

### Option 2: Lichess Game Database with Self-Play

Use game outcomes instead of puzzle ratings:

1. Extract positions from high-quality games
2. Label with game result: Win=+100, Draw=0, Loss=-100
3. Train on these (less precise but better than puzzles)

### Option 3: Download Pre-Labeled Datasets

**Kaggle Chess Datasets:**
- Search: https://www.kaggle.com/search?q=chess+evaluation
- Look for datasets with "stockfish" or "evaluation" labels

**GitHub Datasets:**
- Search: "chess training data stockfish"
- Common formats: CSV with FEN + centipawn evaluation

### Option 4: Use Current Model (NOT Recommended)

**The model trained on Lichess puzzles has severe overfitting:**

Training results showed:
```
Epoch 1/5: Train Loss: 112.1, Val Loss: 112.8
Epoch 2/5: Train Loss: 106.9, Val Loss: 108.8 ✓ Saved
Epoch 3/5: Train Loss: 96.5, Val Loss: 109.2  ← Going back up!
Epoch 4/5: Train Loss: 87.3, Val Loss: 110.2
Epoch 5/5: Train Loss: 82.5, Val Loss: 110.6
```

This is classic overfitting:
- Train loss dropped from 112 → 82 (good)
- Val loss stayed at ~110 and slightly increased (bad!)
- Model memorized puzzle patterns without learning evaluation

**Don't use this model - it won't play well.**

## My Recommendation

**Option 1: Stockfish labels at depth 15 for 10k positions**

Why 10k?
- Enough to train a decent model
- Fast enough (~2 hours)
- Cheap enough (~$3)
- Much better than puzzle ratings

Why depth 15 instead of 18?
- 3x faster
- Still very accurate
- Cost effective

This will give you a properly trained model that actually understands positions!
