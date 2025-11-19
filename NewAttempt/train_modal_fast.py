"""
Modal training script - FAST version with parallel Stockfish labeling.

Usage:
  modal run --detach train_modal_fast.py::generate_data
  modal run --detach train_modal_fast.py::train_model
"""
import modal
from pathlib import Path

app = modal.App("chessbot-nnue-training")

# Container image with local data directory mounted
LOCAL_DATA_PATH = Path(__file__).parent / "data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "python-chess", "tqdm")
    .apt_install("stockfish")
    .add_local_dir(LOCAL_DATA_PATH, remote_path="/pgn")
)

# Volume for persistent storage
volume = modal.Volume.from_name("chessbot-data", create_if_missing=True)
VOLUME_PATH = "/data"


# Parallel labeling function
@app.function(
    image=image,
    timeout=300,
    cpu=1.0,
)
def label_batch(fens: list, stockfish_depth: int = 18):
    """Label a batch of positions in parallel."""
    import chess
    import chess.engine

    labeled_data = []

    with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
        for fen in fens:
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=stockfish_depth))
                score = info.get("score")

                if score is None:
                    continue

                if score.is_mate():
                    mate_in = score.relative.moves
                    cp_score = 10000 - mate_in * 10 if mate_in > 0 else -10000 - mate_in * 10
                else:
                    cp_score = score.relative.score()

                if not board.turn:
                    cp_score = -cp_score

                cp_score = max(-10000, min(10000, cp_score))

                labeled_data.append({
                    "fen": fen,
                    "score": cp_score,
                    "depth": stockfish_depth,
                })

            except Exception as e:
                continue

    return labeled_data


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    cpu=2.0,
    memory=4096,
)
def generate_data(
    pgn_filename: str = "lichess_elite_2021-01.pgn",
    num_positions: int = 50000,
    stockfish_depth: int = 18,
    batch_size: int = 100,  # Process 100 positions in parallel
):
    """Generate training data with PARALLEL Stockfish labeling."""
    import chess.pgn
    import json
    import random
    from pathlib import Path

    print("=" * 70)
    print("DATA GENERATION (PARALLEL)")
    print("=" * 70)
    print(f"PGN: {pgn_filename}")
    print(f"Positions: {num_positions}")
    print(f"Depth: {stockfish_depth}")
    print(f"Batch size: {batch_size} (parallel workers)")
    print()

    # PGN file is mounted at /pgn
    pgn_path = Path("/pgn") / pgn_filename
    if not pgn_path.exists():
        print(f"ERROR: PGN not found: {pgn_path}")
        return 0

    print(f"Using PGN: {pgn_path}")

    # Extract positions (same as before)
    print("\nExtracting positions from PGN...")
    positions = []
    games_processed = 0

    with open(pgn_path) as f:
        while len(positions) < num_positions * 2:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_processed += 1
            if games_processed % 1000 == 0:
                print(f"  Processed {games_processed} games, extracted {len(positions)} positions")

            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                board.push(move)
                ply += 1

                if ply < 10 or len(board.piece_map()) < 8:
                    continue
                if board.is_game_over():
                    break
                if random.random() < 0.2:
                    positions.append(board.fen())

    random.shuffle(positions)
    positions = positions[:num_positions]
    print(f"\n✓ Extracted {len(positions)} positions from {games_processed} games")

    # Label with Stockfish IN PARALLEL
    print(f"\nLabeling with Stockfish (depth {stockfish_depth}, {batch_size} parallel workers)...")

    # Split positions into batches
    batches = []
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i + batch_size]
        batches.append(batch)

    print(f"Created {len(batches)} batches of ~{batch_size} positions each")

    # Process all batches in parallel using Modal's map
    labeled_batches = []
    for batch in batches:
        result = label_batch.remote(batch, stockfish_depth)
        labeled_batches.append(result)

    print("Waiting for all batches to complete...")

    # Collect results
    labeled_data = []
    for i, future in enumerate(labeled_batches):
        batch_result = future
        labeled_data.extend(batch_result)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(batches)} batches ({len(labeled_data)} positions labeled)")

    print(f"\n✓ Labeled {len(labeled_data)} positions")

    # Save
    output_path = Path(VOLUME_PATH) / "training_data.json"
    with open(output_path, 'w') as f:
        json.dump(labeled_data, f)

    volume.commit()

    print(f"\n✓ COMPLETE! Saved to training_data.json")
    print(f"✓ Generated {len(labeled_data)} labeled positions")

    return len(labeled_data)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="T4",
    timeout=14400,
    memory=16384,
)
def train_model(
    data_file: str = "training_data.json",
    epochs: int = 5,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
):
    """Train NNUE model (same as before)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import json
    from tqdm import tqdm
    import chess
    from pathlib import Path

    print("=" * 70)
    print("NNUE TRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    data_path = Path(VOLUME_PATH) / data_file
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} positions")

    # [Rest of training code same as train_modal_simple.py]
    # ... (copying from previous version)

    NUM_PIECES = 10
    NUM_SQUARES = 64
    FEATURE_SIZE = NUM_PIECES * NUM_SQUARES * 64
    HIDDEN1 = 512
    HIDDEN2 = 256

    def piece_to_index(piece):
        if piece.piece_type == chess.KING:
            return -1
        idx = piece.piece_type - 1
        if not piece.color:
            idx += 5
        return idx

    def get_feature_index(piece_idx, square, king_sq, flip):
        if flip:
            square = square ^ 56
            king_sq = king_sq ^ 56
            piece_idx = (piece_idx + 5) % 10
        return piece_idx * NUM_SQUARES * 64 + square * 64 + king_sq

    class NNUENetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = nn.Linear(FEATURE_SIZE, HIDDEN1)
            self.fc1 = nn.Linear(HIDDEN1 * 2, HIDDEN2)
            self.fc2 = nn.Linear(HIDDEN2, 1)

        def forward(self, white_features, black_features):
            white_ft = torch.clamp(self.ft(white_features), 0, 1)
            black_ft = torch.clamp(self.ft(black_features), 0, 1)
            combined = torch.cat([white_ft, black_ft], dim=1)
            x = torch.relu(self.fc1(combined))
            return self.fc2(x).squeeze(-1)

    class ChessDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            board = chess.Board(item['fen'])

            white_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
            black_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)

            for square in range(64):
                piece = board.piece_at(square)
                if piece and piece.piece_type != chess.KING:
                    piece_idx = piece_to_index(piece)
                    feat_idx_w = get_feature_index(piece_idx, square, white_king_sq, False)
                    white_features[feat_idx_w] = 1.0
                    feat_idx_b = get_feature_index(piece_idx, square, black_king_sq, True)
                    black_features[feat_idx_b] = 1.0

            target = np.clip(item['score'] / 100.0, -100, 100)
            return white_features, black_features, target

    dataset = ChessDataset(data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {train_size:,}, Val: {val_size:,}\n")

    model = NNUENetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for white_feat, black_feat, target in pbar:
            white_feat = white_feat.to(device)
            black_feat = black_feat.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(white_feat, black_feat)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= train_batches

        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for white_feat, black_feat, target in val_loader:
                white_feat = white_feat.to(device)
                black_feat = black_feat.to(device)
                target = target.to(device)
                output = model(white_feat, black_feat)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        print(f"\nEpoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(VOLUME_PATH) / "model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_path)
            print(f"  ✓ Saved best model")

        scheduler.step()
        volume.commit()

    print(f"\n✓ TRAINING COMPLETE!")
    print(f"✓ Best val loss: {best_val_loss:.4f}")

    return best_val_loss
