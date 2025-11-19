# ChessBot Quick Start Guide

## Installation (5 minutes)

```bash
cd /Users/lironkatsif/desktop/swe/personal/chessbot/newattempt

# Install dependencies
pip install -r requirements.txt

# Test the engine
python3 test_engine.py
```

## Basic Usage

### 1. Analyze a Position

```python
from engine import ChessEngine

engine = ChessEngine()

# Set up a position
engine.set_position_from_moves(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6'])

# Get best move
move, score, info = engine.get_best_move(depth=6)

print(f"Best move: {engine.board.san(move)}")
print(f"Evaluation: {score:+d} centipawns")
print(f"Principal variation: {' '.join(info['pv'][:8])}")
print(f"Nodes: {info['nodes']:,}, Time: {info['time_ms']:.0f}ms")
```

### 2. Analyze Your Games

```python
from engine import ChessEngine

engine = ChessEngine()

# Your game moves
my_game = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6']

engine.set_position_from_moves(my_game)
analysis = engine.analyze_position(depth=8)

print(f"Position after {len(my_game)} moves:")
engine.print_board()
print(f"\nEngine suggests: {analysis[0]['move']}")
print(f"Evaluation: {analysis[0]['score_cp']:+d} cp")
```

### 3. Compare Moves

```python
from engine import ChessEngine

engine = ChessEngine()
engine.set_position_from_moves(['e4', 'e5', 'Nf3'])

# Analyze current position
best_move, best_score, _ = engine.get_best_move(depth=6)
print(f"Best move: {engine.board.san(best_move)} ({best_score:+d} cp)")

# Test a specific move
test_move = 'Nc6'
engine.make_move(test_move)
opponent_best, opponent_score, _ = engine.get_best_move(depth=6)
print(f"After {test_move}, opponent's best: {engine.board.san(opponent_best)} ({-opponent_score:+d} cp)")
```

## Training with Modal (Advanced)

### Step 1: Get PGN Data

Download high-quality games:
```bash
# Lichess elite database (recommended)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
zstd -d lichess_db_standard_rated_2024-01.pgn.zst
```

### Step 2: Set up Modal

```bash
pip install modal
modal token new
```

### Step 3: Generate Training Data

```bash
# Generate 100k positions labeled with Stockfish depth 18
modal run train_modal.py generate \
  lichess_db_standard_rated_2024-01.pgn \
  training_data.json \
  100000 \
  18
```

This takes ~2-4 hours on Modal depending on position count and Stockfish depth.

### Step 4: Train NNUE

```bash
# Train for 5 epochs with batch size 4096 (GPU recommended)
modal run train_modal.py train \
  training_data.json \
  model.pt \
  5 \
  4096
```

Training 100k positions takes ~15-30 minutes on a T4 GPU.

### Step 5: Convert for Fast Inference

```python
from utils import convert_torch_to_numpy

convert_torch_to_numpy('model.pt', 'model.npz')
```

### Step 6: Use Trained Model

```python
from engine import ChessEngine

engine = ChessEngine()
engine.load_weights('model.npz')

# Now the engine uses your trained weights!
move, score, info = engine.get_best_move(depth=6)
```

## Performance Tips

### Faster Search
- Reduce depth: `depth=5` instead of `depth=6` (much faster)
- Use time limits: `get_best_move(time_limit_ms=1000)`
- Increase TT size: `ChessEngine(tt_size_mb=256)`

### Better Analysis
- Increase depth: `depth=8` or higher
- Use trained weights for better positional understanding
- Analyze critical positions multiple times to verify

### Position Understanding
- **Depth 4-5**: Tactical patterns (forks, pins, skewers)
- **Depth 6-7**: Short combinations, immediate threats
- **Depth 8-10**: Deep tactics, long-term plans
- **Depth 10+**: Complex endgames, deep calculation

## Common Tasks

### Find Tactical Errors in Your Game

```python
from engine import ChessEngine

engine = ChessEngine()
my_moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'O-O', 'Nf6', 'd3', 'd6']

for i in range(2, len(my_moves), 2):  # Check every other move
    engine.new_game()
    engine.set_position_from_moves(my_moves[:i])

    move_played = my_moves[i]
    best_move, best_score, _ = engine.get_best_move(depth=6)

    # Make the move that was played
    engine.make_move(move_played)
    score_after, _, _ = engine.get_best_move(depth=6)

    # Check if move was suboptimal
    if abs(best_score + score_after) > 50:  # 0.5 pawn difference
        print(f"Move {i//2 + 1}: {move_played} - Inaccuracy!")
        print(f"  Better: {engine.board.san(best_move)}")
        print(f"  Loss: {abs(best_score + score_after)} cp")
```

### Create Opening Repertoire Analysis

```python
from engine import ChessEngine

opening_moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']  # Spanish Opening

engine = ChessEngine()
engine.set_position_from_moves(opening_moves)

# Get top 3 candidate moves
legal_moves = engine.get_legal_moves()
evaluations = []

for move_san in legal_moves[:10]:  # Test first 10 legal moves
    test_engine = ChessEngine()
    test_engine.set_position_from_moves(opening_moves)
    test_engine.make_move(move_san)

    _, score, _ = test_engine.get_best_move(depth=6)
    evaluations.append((move_san, -score))  # Negate for current side

evaluations.sort(key=lambda x: x[1], reverse=True)

print("Top 3 replies:")
for i, (move, score) in enumerate(evaluations[:3], 1):
    print(f"{i}. {move}: {score:+d} cp")
```

## Next Steps

1. **Train your model**: Follow the Modal training steps above
2. **Experiment with depths**: Try different search depths for speed/strength tradeoff
3. **Build a GUI**: Integrate with chess GUIs or web interface
4. **Analyze your games**: Review your games for tactical mistakes
5. **Create training tools**: Build spaced repetition for tactics

## Troubleshooting

**Slow search?**
- Reduce depth to 5
- Ensure you're using NumPy weights (not PyTorch)
- Increase TT size if you have RAM

**Poor move quality?**
- Train NNUE weights with Stockfish labels
- Increase search depth
- Check that weights loaded correctly

**High memory usage?**
- Reduce TT size: `ChessEngine(tt_size_mb=64)`
- Don't create multiple engine instances

**Import errors?**
- Install dependencies: `pip install -r requirements.txt`
- Use Python 3.8+

## Resources

- **python-chess docs**: https://python-chess.readthedocs.io/
- **Lichess database**: https://database.lichess.org/
- **Modal docs**: https://modal.com/docs
- **Chess programming wiki**: https://www.chessprogramming.org/

Enjoy analyzing your games!
