# ChessBot - High-Performance NNUE Chess Engine

A high-performance chess engine written in Python featuring NNUE evaluation (512→256→1), alpha-beta search with advanced move ordering, and Modal-based distributed training.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Engine
```bash
python3 test_engine.py
```

### 3. Train NNUE Weights (Optional)

**See RUN_THIS.md for complete training instructions.**

**⚠️ IMPORTANT:** Use Stockfish-labeled data, NOT Lichess puzzles! Puzzle ratings cause overfitting.

Quick version (recommended - 10k positions, depth 15):
```bash
# Generate training data (runs in background, ~2 hours, ~$3)
modal run --detach train_modal_simple.py::generate_data \
  --pgn-filename lichess_elite_2021-01.pgn \
  --num-positions 10000 \
  --stockfish-depth 15

# Train model (runs in background, 20-30 minutes)
modal run --detach train_modal_simple.py::train_model \
  --data-file training_data.json \
  --epochs 5 \
  --batch-size 4096

# Download and convert
modal volume get chessbot-data model.pt ./model.pt
python3 -c "from utils import convert_torch_to_numpy; convert_torch_to_numpy('model.pt', 'model.npz')"
```

### 4. Use the Engine

```python
from engine import ChessEngine

# Create engine
engine = ChessEngine()

# Load trained weights (optional)
# engine.load_weights('model.npz')

# Analyze a position
engine.set_position_from_moves(['e4', 'e5', 'Nf3', 'Nc6'])
move, score, info = engine.get_best_move(depth=6)

print(f"Best move: {engine.board.san(move)}")
print(f"Score: {score:+d} cp")
print(f"PV: {' '.join(info['pv'])}")
```

## Architecture

### Core Components

- **engine.py** - Main engine API
- **nnue.py** - NNUE evaluator (512→256→1) with incremental updates
- **search.py** - Alpha-beta search with PVS, iterative deepening, quiescence
- **tt.py** - Transposition table with age-based replacement
- **zobrist.py** - Zobrist hashing for position keys

### Training & Utilities

- **train_modal_simple.py** - Modal training pipeline with Stockfish labeling (RECOMMENDED)
- **utils.py** - Model conversion, benchmarking, testing utilities
- **analyze_game.py** - Interactive game analysis tool
- **download_data.py** - Alternative data sources (not recommended - causes overfitting)

### Features

✓ NNUE evaluation with incremental accumulator updates
✓ Negamax search with alpha-beta pruning
✓ Principal Variation Search (PVS)
✓ Iterative deepening (depth 1→6+)
✓ Quiescence search with delta pruning
✓ Advanced move ordering (TT, MVV-LVA, killers, history)
✓ Late move reductions (LMR)
✓ Null-move pruning
✓ Transposition table with age replacement
✓ Modal-based distributed training

## Documentation

- **README.md** - This file (overview and quick start)
- **RUN_THIS.md** - Training commands and workflow (UPDATED with correct Modal syntax)
- **QUICKSTART.md** - Detailed usage examples and API guide
- **TEACHER.md** - In-depth educational guide explaining how everything works
- **BETTER_DATA_SOURCES.md** - Why Lichess puzzles fail and what to use instead
- **TRAINING_LESSONS_LEARNED.md** - All issues encountered and solutions (troubleshooting guide)

## Performance

**Untrained weights:**
- ~50,000 NPS at depth 6
- Finds tactical patterns reliably
- Basic positional understanding

**With trained weights (50k positions, depth 18):**
- Strong positional evaluation
- ~1800-2000 Elo estimated
- Excellent tactical vision
- Good for game analysis

## Example Usage

### Analyze Your Games
```python
from engine import ChessEngine

engine = ChessEngine()
engine.load_weights('model.npz')  # If you have trained weights

# Your game
moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'O-O', 'Nf6']
engine.set_position_from_moves(moves)

# Get deep analysis
analysis = engine.analyze_position(depth=8)
print(f"Best: {analysis[0]['move']} ({analysis[0]['score_cp']:+d} cp)")
print(f"PV: {analysis[0]['pv_san']}")
```

### Find Mistakes in Your Games
```bash
python3 analyze_game.py
```

This runs an interactive analysis that:
- Compares your moves vs engine recommendations
- Highlights blunders (??) and mistakes (?)
- Shows evaluation loss for each move
- Suggests better alternatives

### Interactive Analysis Mode
```bash
python3 analyze_game.py interactive
```

Enter moves in real-time and get instant analysis.

## Training Data

**IMPORTANT:** You need Stockfish-labeled positions, not puzzle databases!

### Recommended Approach (Built-in)

Use the included `train_modal_simple.py::generate_data` function:
- Extracts positions from PGN files (download from Lichess database)
- Labels with Stockfish at configurable depth
- Recommended: 10k positions at depth 15 (~2 hours, ~$3)

### PGN Sources

Download high-quality games:
- **Lichess Elite Database**: https://database.lichess.org/ (2400+ rated games, NOT puzzles!)
- **CCRL Games**: https://computerchess.org.uk/ccrl/
- **FICS Database**: https://www.ficsgames.org/

### ❌ What NOT to Use

**Lichess Puzzle Database** - Causes severe overfitting!
- Puzzle ratings ≠ position evaluations
- Puzzle rating measures "how hard is the tactic" not "who's winning"
- Tested and confirmed: validation loss stays high while training loss drops
- See BETTER_DATA_SOURCES.md for full explanation

### Best Results

- 10k-50k positions
- Stockfish depth 15-18
- High-rated games (2400+)
- Mix of openings and middlegames

## Project Structure

```
newattempt/
├── Core Engine Files
│   ├── engine.py                    # Main engine API
│   ├── nnue.py                      # NNUE evaluator
│   ├── search.py                    # Search algorithm
│   ├── tt.py                        # Transposition table
│   └── zobrist.py                   # Zobrist hashing
│
├── Training & Utilities
│   ├── train_modal_simple.py        # Modal training (RECOMMENDED)
│   ├── utils.py                     # Model conversion, benchmarking
│   ├── analyze_game.py              # Game analysis tool
│   ├── test_engine.py               # Test suite
│   └── download_data.py             # Alternative data (causes overfitting - avoid!)
│
├── Documentation
│   ├── README.md                    # This file (start here!)
│   ├── RUN_THIS.md                  # Training workflow with correct commands
│   ├── QUICKSTART.md                # Usage examples
│   ├── TEACHER.md                   # How everything works (educational)
│   ├── BETTER_DATA_SOURCES.md       # Data quality explained
│   └── TRAINING_LESSONS_LEARNED.md  # Troubleshooting guide
│
└── requirements.txt                 # Dependencies
```

## Requirements

- Python 3.8+
- python-chess >= 1.9.0
- numpy >= 1.24.0
- torch >= 2.0.0 (for training only)
- modal >= 0.56.0 (for training only)

## License

MIT License - Free for learning and experimentation!

## Contributing

Contributions welcome! Areas for improvement:
- Multi-PV search
- UCI protocol
- Opening book integration
- Syzygy tablebase support
- Better time management

## Acknowledgments

- Stockfish team for NNUE architecture inspiration
- python-chess library for move generation
- Modal for distributed training infrastructure
