"""
Train NNUE Chess Bot on Modal with GPU

Modal Setup:
1. Install: pip install modal
2. Setup: modal setup
3. Run: modal run train_modal.py
"""

import modal

# Create Modal app
app = modal.App("nnue-chess-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "chess==1.10.0",
        "numpy==1.26.0",
        "tqdm==4.66.0",
    )
)

# Create a volume to store the trained model
volume = modal.Volume.from_name("chess-models", create_if_missing=True)

# Training function that runs on Modal
@app.function(
    image=image,
    gpu="T4",  # NVIDIA T4 GPU (free tier)
    timeout=3600 * 4,  # 4 hour timeout
    volumes={"/models": volume},
)
def train_on_modal(
    max_games_per_file=10000,
    max_files=4,
    positions_per_game=25,
    min_elo=2200,
    batch_size=1024,
    num_epochs=50,
):
    """
    Train NNUE model on Modal with GPU

    This function runs in the cloud with GPU acceleration
    """
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from tqdm import tqdm
    import chess
    import chess.pgn
    import numpy as np

    print("=" * 60)
    print("NNUE TRAINING ON MODAL (GPU)")
    print("=" * 60)
    print()

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # NNUE Model (inline to avoid imports)
    class NNUE(nn.Module):
        def __init__(self, input_size=640, hidden1_size=1408, hidden2_size=32):
            super(NNUE, self).__init__()
            self.ft_white = nn.Linear(input_size, hidden1_size)
            self.ft_black = nn.Linear(input_size, hidden1_size)
            self.fc1 = nn.Linear(hidden1_size * 2, hidden2_size)
            self.fc2 = nn.Linear(hidden2_size, 1)
            self.clipped_relu = lambda x: torch.clamp(torch.relu(x), 0, 1)

        def forward(self, white_features, black_features):
            white_hidden = self.clipped_relu(self.ft_white(white_features))
            black_hidden = self.clipped_relu(self.ft_black(black_features))
            combined = torch.cat([white_hidden, black_hidden], dim=1)
            x = self.clipped_relu(self.fc1(combined))
            x = self.fc2(x)
            return x

    # Dataset class
    class NNUEDataset(Dataset):
        def __init__(self, white_features, black_features, results):
            self.white_features = torch.tensor(white_features, dtype=torch.float32)
            self.black_features = torch.tensor(black_features, dtype=torch.float32)
            self.results = torch.tensor(results, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.white_features)

        def __getitem__(self, idx):
            return self.white_features[idx], self.black_features[idx], self.results[idx]

    # HalfKP encoding
    def board_to_halfkp_features(board):
        white_features = np.zeros(640, dtype=np.float32)
        black_features = np.zeros(640, dtype=np.float32)

        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        if white_king_square is None or black_king_square is None:
            return white_features, black_features

        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue

            piece_idx = piece.piece_type - 1
            color_offset = 0 if piece.color == chess.WHITE else 5

            # White perspective
            idx = (piece_idx + color_offset) * 64 + square
            white_features[idx] = 1.0

            # Black perspective (flipped)
            flipped_square = square ^ 56
            idx = (piece_idx + (0 if not piece.color else 5)) * 64 + flipped_square
            black_features[idx] = 1.0

        return white_features, black_features

    # Load and process data
    print("LOADING DATA FROM MODAL STORAGE")
    print("-" * 60)

    # Note: Data should be uploaded to Modal storage first
    # For now, we'll use the data mounted from local
    data_dir = "/data"

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} not found!")
        print("Upload your PGN files to Modal storage first.")
        return None

    pgn_files = [f for f in os.listdir(data_dir) if f.endswith(".pgn")]
    print(f"Found {len(pgn_files)} PGN files")

    # Load games
    all_games = []
    for pgn_file in tqdm(pgn_files[:max_files], desc="Loading PGN files"):
        file_path = os.path.join(data_dir, pgn_file)
        games_loaded = 0

        with open(file_path) as f:
            while games_loaded < max_games_per_file:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Filter by ELO
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                    if white_elo >= min_elo and black_elo >= min_elo:
                        all_games.append(game)
                        games_loaded += 1
                except:
                    continue

    print(f"Loaded {len(all_games)} high-quality games (ELO >= {min_elo})\n")

    # Create training data
    print("Creating training data...")
    white_features_list = []
    black_features_list = []
    results_list = []

    for game in tqdm(all_games, desc="Processing games"):
        result_str = game.headers.get("Result", "*")
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = -1.0
        elif result_str == "1/2-1/2":
            result = 0.0
        else:
            continue

        board = game.board()
        moves = list(game.mainline_moves())

        if len(moves) < 15:
            continue

        # Sample positions
        if len(moves) <= positions_per_game:
            sample_indices = range(len(moves))
        else:
            step = len(moves) / positions_per_game
            sample_indices = [int(i * step) for i in range(positions_per_game)]

        board = game.board()
        for i, move in enumerate(moves):
            if i in sample_indices:
                white_feat, black_feat = board_to_halfkp_features(board)
                white_features_list.append(white_feat)
                black_features_list.append(black_feat)

                # Adjust result for side to move
                if board.turn == chess.WHITE:
                    results_list.append(result)
                else:
                    results_list.append(-result)

            board.push(move)

    X_white = np.array(white_features_list, dtype=np.float32)
    X_black = np.array(black_features_list, dtype=np.float32)
    y = np.array(results_list, dtype=np.float32)

    print(f"Created {len(X_white)} training positions")
    print(f"Result distribution - Win: {(y > 0).sum()}, Draw: {(y == 0).sum()}, Loss: {(y < 0).sum()}\n")

    # Create dataset
    dataset = NNUEDataset(X_white, X_black, y)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # Create model
    print("=" * 60)
    print("MODEL")
    print("=" * 60)
    model = NNUE(hidden1_size=1408, hidden2_size=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for white_feat, black_feat, result in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            white_feat = white_feat.to(device)
            black_feat = black_feat.to(device)
            result = result.to(device)

            optimizer.zero_grad()
            prediction = model(white_feat, black_feat)
            loss = criterion(prediction, result)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for white_feat, black_feat, result in val_loader:
                white_feat = white_feat.to(device)
                black_feat = black_feat.to(device)
                result = result.to(device)

                prediction = model(white_feat, black_feat)
                loss = criterion(prediction, result)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/models/nnue_best.pth")
            print(f"  ✓ Saved best model")

    # Save final model
    torch.save(model.state_dict(), "/models/nnue_final.pth")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Best model saved (val_loss: {best_val_loss:.6f})")
    print(f"✓ Models saved to Modal volume: /models/")
    print()
    print("Download with: modal volume get chess-models nnue_best.pth")

    # Commit volume changes
    volume.commit()

    return {
        "best_val_loss": best_val_loss,
        "total_positions": len(X_white),
        "epochs": num_epochs,
    }


# Local entrypoint
@app.local_entrypoint()
def main():
    """
    Main function that runs locally and triggers Modal training
    """
    import sys

    print("=" * 60)
    print("TRAINING NNUE CHESS BOT ON MODAL")
    print("=" * 60)
    print()
    print("This will upload your data and train on Modal's GPU")
    print()

    # Training configuration
    print("Training Configuration:")
    print("  Max games per file: 10,000")
    print("  Max files: 4")
    print("  Positions per game: 25")
    print("  Min ELO: 2200")
    print("  Batch size: 1024")
    print("  Epochs: 50")
    print()
    print("Estimated time with GPU: 30-60 minutes")
    print("Estimated cost: ~$0.50-1.00 (well within 30 free credits)")
    print()

    confirm = input("Start training? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    print()
    print("Starting training on Modal...")
    print()

    # Run training on Modal
    result = train_on_modal.remote(
        max_games_per_file=10000,
        max_files=4,
        positions_per_game=25,
        min_elo=2200,
        batch_size=1024,
        num_epochs=50,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print(f"Results:")
    print(f"  Best validation loss: {result['best_val_loss']:.6f}")
    print(f"  Total positions: {result['total_positions']:,}")
    print(f"  Epochs: {result['epochs']}")
    print()
    print("Next steps:")
    print("  1. Download model:")
    print("     modal volume get chess-models nnue_best.pth")
    print()
    print("  2. Play against it:")
    print("     python3 nnue_play.py")
    print()
