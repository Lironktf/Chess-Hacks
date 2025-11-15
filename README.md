# Chess-Hacks

A neural network chess engine trained on Lichess elite games.

## Progressive Training System

The training script now supports progressive training, allowing you to gradually increase the dataset size over multiple stages. This approach:

- **Starts small** for faster initial iterations and debugging
- **Scales up** progressively to larger datasets
- **Saves checkpoints** automatically after each epoch
- **Resumes training** from the last checkpoint if interrupted

### Training Stages

The training is configured in `TRAINING_STAGES` in `train.py`:

```python
TRAINING_STAGES = [
    {"games_per_file": 1000, "max_files": 2, "max_samples": 50000, "epochs": 10, "name": "stage1_8k"},
    {"games_per_file": 2000, "max_files": 2, "max_samples": 100000, "epochs": 10, "name": "stage2_16k"},
    {"games_per_file": 2000, "max_files": 4, "max_samples": 200000, "epochs": 10, "name": "stage3_32k"},
    {"games_per_file": 2000, "max_files": 4, "max_samples": 500000, "epochs": 20, "name": "stage4_80k"},
]
```

Each stage progressively increases:
- Number of games per file
- Number of files processed
- Maximum training samples
- Number of training epochs

### Usage

1. **Start training:**
   ```bash
   python3 train.py
   ```

2. **Resume from checkpoint:**
   - If a checkpoint is found, you'll be prompted to continue
   - Type `y` to resume from the last checkpoint

3. **Checkpoints:**
   - Saved automatically after each epoch in the `checkpoints/` directory
   - Includes model weights, optimizer state, vocabulary, and training metadata
   - Latest checkpoint is automatically detected on restart

### Customizing Training Stages

Edit the `TRAINING_STAGES` list in `train.py` to customize:
- Add more stages for even larger datasets
- Adjust epochs per stage
- Modify dataset sizes based on your system's memory

### Files

- `train.py` - Progressive training script with checkpointing
- `model.py` - Neural network architecture
- `dataset.py` - PyTorch dataset wrapper
- `aux.py` - Helper functions for board encoding and move processing
- `checkpoints/` - Directory containing saved model checkpoints

## Web UI - Play Against the Bot

A web interface is available to play against the trained chess bot in your browser.

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model** (if you haven't already):
   ```bash
   python3 train.py
   ```
   The web UI will automatically find and load the most recent checkpoint.

### Running the Web Server

1. **Start the Flask server:**
   ```bash
   python3 app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5000`

3. **Play chess:**
   - You play as **White**
   - Click a piece, then click a square to move
   - The bot plays as **Black** automatically
   - Use "New Game" to start over
   - Use "Flip Board" to rotate the board

### Features

- **Interactive chess board** with drag-and-drop piece movement
- **Real-time move validation** - only legal moves are allowed
- **Move history** display showing all moves in the game
- **Game status** indicators (check, checkmate, draw)
- **Responsive design** that works on desktop and mobile
- **Beautiful UI** with modern styling

### Troubleshooting

- **"No trained model found"**: Make sure you've trained a model first using `train.py`
- **Bot makes illegal moves**: This shouldn't happen, but if it does, the game will handle it gracefully
- **Server won't start**: Check that Flask is installed (`pip install flask`)
