"""
Training script for compact chess value network
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import CompactChessNet
from dataset import ChessValueDataset
from utils import load_pgn_games, create_training_data


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for boards, results in tqdm(dataloader, desc="Training"):
        boards = boards.to(device)
        results = results.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(boards)

        # Compute loss
        loss = criterion(predictions, results)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
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
        for boards, results in dataloader:
            boards = boards.to(device)
            results = results.to(device)

            predictions = model(boards)
            loss = criterion(predictions, results)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # Configuration
    DATA_DIR = "../data"
    MAX_GAMES_PER_FILE = 500  # Limit games per file
    MAX_FILES = 10  # Limit number of PGN files
    MAX_POSITIONS_PER_GAME = 15  # Sample positions per game
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    TRAIN_SPLIT = 0.9

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load PGN files
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please create a 'data' directory with PGN files.")
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
        games = load_pgn_games(file_path, max_games=MAX_GAMES_PER_FILE)
        all_games.extend(games)

    print(f"Loaded {len(all_games)} games total\n")

    # Create training data
    print("Creating training data from positions...")
    X, y = create_training_data(all_games, max_positions_per_game=MAX_POSITIONS_PER_GAME)
    print(f"Created {len(X)} training positions")
    print(f"Result distribution - White wins: {(y > 0).sum()}, "
          f"Draws: {(y == 0).sum()}, Black wins: {(y < 0).sum()}\n")

    # Create dataset and split
    dataset = ChessValueDataset(X, y)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # Create model
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model = CompactChessNet().to(device)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size (float32): {model.count_parameters() * 4 / 1024 / 1024:.2f} MB")
    print(f"Model size (int8): {model.count_parameters() / 1024 / 1024:.2f} MB\n")

    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Time tracking
        epoch_time = time.time() - start_time

        # Print progress
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print("✓ Saved final model to: final_model.pth")
    print("✓ Saved best model to: best_model.pth")

    # Quantize and save compressed model
    print("\nQuantizing model to INT8...")
    model.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    torch.save(model_int8.state_dict(), "model_int8.pth")
    print("✓ Saved quantized model to: model_int8.pth")

    # Check file sizes
    print("\n" + "=" * 60)
    print("MODEL SIZES")
    print("=" * 60)
    for filename in ["best_model.pth", "final_model.pth", "model_int8.pth"]:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / 1024 / 1024
            print(f"{filename}: {size_mb:.2f} MB")

    print("\n✅ Training complete! Use 'engine.py' to play against the bot.")


if __name__ == "__main__":
    main()
