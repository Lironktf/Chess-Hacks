"""
Training script for NNUE model

This trains the NNUE to evaluate chess positions based on game results
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import chess
import chess.pgn
import numpy as np

from nnue_model import NNUE, board_to_halfkp_features


class NNUEDataset(Dataset):
    """
    Dataset for NNUE training

    Each sample contains:
    - white_features: HalfKP features from white's perspective
    - black_features: HalfKP features from black's perspective
    - result: Game result (-1, 0, 1) adjusted for side to move
    """

    def __init__(self, white_features, black_features, results):
        """
        Args:
            white_features: numpy array (N, 640)
            black_features: numpy array (N, 640)
            results: numpy array (N,) with values in [-1, 1]
        """
        self.white_features = torch.tensor(white_features, dtype=torch.float32)
        self.black_features = torch.tensor(black_features, dtype=torch.float32)
        self.results = torch.tensor(results, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.white_features)

    def __getitem__(self, idx):
        return self.white_features[idx], self.black_features[idx], self.results[idx]


def get_game_result(game) -> Optional[float]:
    """
    Get game result

    Returns:
        1.0 = White wins
        0.0 = Draw
        -1.0 = Black wins
        None = Unknown
    """
    result = game.headers.get("Result", "*")

    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result == "1/2-1/2":
        return 0.0
    else:
        return None


def create_training_data(games, positions_per_game=20):
    """
    Create training data from PGN games

    Args:
        games: List of chess.pgn.Game objects
        positions_per_game: Number of positions to sample per game

    Returns:
        white_features, black_features, results
    """
    white_features_list = []
    black_features_list = []
    results_list = []

    for game in tqdm(games, desc="Processing games"):
        result = get_game_result(game)
        if result is None:
            continue

        board = game.board()
        moves = list(game.mainline_moves())

        # Skip very short games
        if len(moves) < 15:
            continue

        # Sample positions evenly
        if len(moves) <= positions_per_game:
            sample_indices = range(len(moves))
        else:
            step = len(moves) / positions_per_game
            sample_indices = [int(i * step) for i in range(positions_per_game)]

        # Extract positions
        board = game.board()
        for i, move in enumerate(moves):
            if i in sample_indices:
                # Get features
                white_feat, black_feat = board_to_halfkp_features(board)

                white_features_list.append(white_feat)
                black_features_list.append(black_feat)

                # Result from current player's perspective
                if board.turn == chess.WHITE:
                    results_list.append(result)
                else:
                    results_list.append(-result)

            board.push(move)

    return (
        np.array(white_features_list, dtype=np.float32),
        np.array(black_features_list, dtype=np.float32),
        np.array(results_list, dtype=np.float32)
    )


def load_pgn_games(file_path, max_games=None, min_elo=2000):
    """
    Load games from PGN file

    Args:
        file_path: Path to PGN file
        max_games: Maximum games to load
        min_elo: Minimum ELO for both players

    Returns:
        List of games
    """
    games = []

    with open(file_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Filter by ELO
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                if white_elo < min_elo or black_elo < min_elo:
                    continue
            except:
                continue

            games.append(game)

            if max_games and len(games) >= max_games:
                break

    return games


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for white_feat, black_feat, result in tqdm(dataloader, desc="Training"):
        white_feat = white_feat.to(device)
        black_feat = black_feat.to(device)
        result = result.to(device)

        optimizer.zero_grad()

        # Forward pass
        prediction = model(white_feat, black_feat)

        # Loss
        loss = criterion(prediction, result)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for white_feat, black_feat, result in dataloader:
            white_feat = white_feat.to(device)
            black_feat = black_feat.to(device)
            result = result.to(device)

            prediction = model(white_feat, black_feat)
            loss = criterion(prediction, result)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # Configuration
    DATA_DIR = "../data"
    MAX_GAMES_PER_FILE = 2000  # Increase for better training
    MAX_FILES = 20
    POSITIONS_PER_GAME = 25
    MIN_ELO = 2200  # Only use high-quality games
    BATCH_SIZE = 1024  # Larger batch size for faster training
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.95

    # Model size - OPTIMIZED FOR MAX STRENGTH UNDER 10MB
    HIDDEN1_SIZE = 1408  # ~7.2MB model (optimal for 10MB budget)
    HIDDEN2_SIZE = 32

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        sys.exit(1)

    pgn_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pgn")]
    if not pgn_files:
        print(f"Error: No PGN files found in '{DATA_DIR}'!")
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN files")
    pgn_files = pgn_files[:MAX_FILES]

    # Load games
    all_games = []
    for pgn_file in tqdm(pgn_files, desc="Loading PGN files"):
        file_path = os.path.join(DATA_DIR, pgn_file)
        games = load_pgn_games(file_path, max_games=MAX_GAMES_PER_FILE, min_elo=MIN_ELO)
        all_games.extend(games)

    print(f"Loaded {len(all_games)} high-quality games (ELO >= {MIN_ELO})\n")

    if len(all_games) < 100:
        print("Warning: Very few games loaded. Training may not be effective.")
        print("Download more high-ELO games from Lichess database.")

    # Create training data
    print("Creating training data...")
    white_feat, black_feat, results = create_training_data(all_games, POSITIONS_PER_GAME)
    print(f"Created {len(white_feat)} training positions")
    print(f"Result distribution - Win: {(results > 0).sum()}, "
          f"Draw: {(results == 0).sum()}, Loss: {(results < 0).sum()}\n")

    # Create dataset
    dataset = NNUEDataset(white_feat, black_feat, results)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # Create model
    print("=" * 60)
    print("MODEL")
    print("=" * 60)
    model = NNUE(hidden1_size=HIDDEN1_SIZE, hidden2_size=HIDDEN2_SIZE).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_size_mb():.2f} MB\n")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "nnue_best.pth")
            print(f"  ✓ Saved best model")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"nnue_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "nnue_final.pth")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Best model: nnue_best.pth (val_loss: {best_val_loss:.6f})")
    print(f"✓ Final model: nnue_final.pth")
    print("\nUse 'nnue_play.py' to play against the bot!")


if __name__ == "__main__":
    main()
