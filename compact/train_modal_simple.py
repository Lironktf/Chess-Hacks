"""
Simple Modal Training - Mounts local data directly (NO UPLOAD NEEDED!)

Usage:
1. Install: pip install modal
2. Setup: modal setup
3. Run: modal run train_modal_simple.py
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("nnue-chess-training-simple")

# Define image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "chess==1.10.0",
        "numpy==1.26.0",
        "tqdm==4.66.0",
    )
)

# Mount local data directory
# This syncs your local ../data/ folder to Modal
data_mount = modal.Mount.from_local_dir(
    local_path=Path(__file__).parent.parent / "data",
    remote_path="/data",
)

# Volume to save trained models
volume = modal.Volume.from_name("chess-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Free tier GPU
    timeout=3600 * 3,  # 3 hours
    mounts=[data_mount],  # Mount local data
    volumes={"/models": volume},  # Save models here
)
def train(
    max_games_per_file=10000,
    max_files=4,
    num_epochs=50,
):
    """Train NNUE on Modal GPU with local data"""

    # Import everything inside the function
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

    print("=" * 70)
    print(" NNUE CHESS BOT TRAINING ON MODAL GPU")
    print("=" * 70)
    print()

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_mem:.1f} GB")
    print()

    # ==================== NNUE Model ====================
    class NNUE(nn.Module):
        def __init__(self, input_size=640, hidden1_size=1408, hidden2_size=32):
            super(NNUE, self).__init__()
            self.ft_white = nn.Linear(input_size, hidden1_size)
            self.ft_black = nn.Linear(input_size, hidden1_size)
            self.fc1 = nn.Linear(hidden1_size * 2, hidden2_size)
            self.fc2 = nn.Linear(hidden2_size, 1)

        def forward(self, white_features, black_features):
            clipped_relu = lambda x: torch.clamp(torch.relu(x), 0, 1)
            white_h = clipped_relu(self.ft_white(white_features))
            black_h = clipped_relu(self.ft_black(black_features))
            x = torch.cat([white_h, black_h], dim=1)
            x = clipped_relu(self.fc1(x))
            return self.fc2(x)

    # ==================== Dataset ====================
    class NNUEDataset(Dataset):
        def __init__(self, white_feat, black_feat, results):
            self.white_feat = torch.tensor(white_feat, dtype=torch.float32)
            self.black_feat = torch.tensor(black_feat, dtype=torch.float32)
            self.results = torch.tensor(results, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.white_feat)

        def __getitem__(self, idx):
            return self.white_feat[idx], self.black_feat[idx], self.results[idx]

    # ==================== HalfKP Features ====================
    def board_to_halfkp(board):
        white_feat = np.zeros(640, dtype=np.float32)
        black_feat = np.zeros(640, dtype=np.float32)

        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)

        if wk is None or bk is None:
            return white_feat, black_feat

        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.KING:
                continue

            pidx = pc.piece_type - 1
            coffset = 0 if pc.color == chess.WHITE else 5

            # White perspective
            white_feat[(pidx + coffset) * 64 + sq] = 1.0

            # Black perspective (flipped)
            black_feat[(pidx + (0 if not pc.color else 5)) * 64 + (sq ^ 56)] = 1.0

        return white_feat, black_feat

    # ==================== Load Data ====================
    print("=" * 70)
    print(" LOADING DATA")
    print("=" * 70)

    pgn_files = sorted([f for f in os.listdir("/data") if f.endswith(".pgn")])
    print(f"✓ Found {len(pgn_files)} PGN files")

    for f in pgn_files:
        size = os.path.getsize(f"/data/{f}") / 1024 / 1024
        print(f"  • {f}: {size:.0f} MB")
    print()

    # Load games
    all_games = []
    for pgn_file in pgn_files[:max_files]:
        print(f"Loading {pgn_file}...")
        count = 0
        with open(f"/data/{pgn_file}") as f:
            with tqdm(total=max_games_per_file, desc="  Games") as pbar:
                while count < max_games_per_file:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    try:
                        we = int(game.headers.get("WhiteElo", 0))
                        be = int(game.headers.get("BlackElo", 0))
                        if we >= 2200 and be >= 2200:
                            all_games.append(game)
                            count += 1
                            pbar.update(1)
                    except:
                        pass

    print()
    print(f"✓ Loaded {len(all_games):,} high-quality games (2200+ ELO)")
    print()

    # ==================== Create Training Data ====================
    print("Creating training positions...")

    white_feats = []
    black_feats = []
    results = []

    for game in tqdm(all_games, desc="Processing"):
        res = game.headers.get("Result", "*")
        if res == "1-0":
            r = 1.0
        elif res == "0-1":
            r = -1.0
        elif res == "1/2-1/2":
            r = 0.0
        else:
            continue

        moves = list(game.mainline_moves())
        if len(moves) < 15:
            continue

        # Sample 25 positions per game
        step = len(moves) / 25 if len(moves) > 25 else 1
        samples = [int(i * step) for i in range(min(25, len(moves)))]

        board = game.board()
        for i, move in enumerate(moves):
            if i in samples:
                wf, bf = board_to_halfkp(board)
                white_feats.append(wf)
                black_feats.append(bf)
                results.append(r if board.turn == chess.WHITE else -r)
            board.push(move)

    X_w = np.array(white_feats, dtype=np.float32)
    X_b = np.array(black_feats, dtype=np.float32)
    y = np.array(results, dtype=np.float32)

    print(f"✓ Created {len(X_w):,} positions")
    print(f"  • Wins: {(y > 0).sum():,}")
    print(f"  • Draws: {(y == 0).sum():,}")
    print(f"  • Losses: {(y < 0).sum():,}")
    print()

    # ==================== Create Datasets ====================
    dataset = NNUEDataset(X_w, X_b, y)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    print(f"✓ Train: {len(train_ds):,} positions")
    print(f"✓ Val: {len(val_ds):,} positions")
    print()

    # ==================== Create Model ====================
    print("=" * 70)
    print(" MODEL")
    print("=" * 70)

    model = NNUE().to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"✓ Parameters: {params:,}")
    print(f"✓ Size: {params * 4 / 1024 / 1024:.2f} MB")
    print()

    # ==================== Training Setup ====================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5)

    # ==================== Training Loop ====================
    print("=" * 70)
    print(" TRAINING")
    print("=" * 70)
    print()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start = time.time()

        # Train
        model.train()
        train_loss = 0
        for wf, bf, r in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            wf, bf, r = wf.to(device), bf.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(wf, bf)
            loss = criterion(pred, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for wf, bf, r in val_loader:
                wf, bf, r = wf.to(device), bf.to(device), r.to(device)
                pred = model(wf, bf)
                loss = criterion(pred, r)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train: {train_loss:.6f}")
        print(f"  Val:   {val_loss:.6f}")
        print(f"  Time:  {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/models/nnue_best.pth")
            print(f"  ✓ Saved best model!")

    # Save final
    torch.save(model.state_dict(), "/models/nnue_final.pth")
    volume.commit()

    print()
    print("=" * 70)
    print(" COMPLETE!")
    print("=" * 70)
    print(f"✓ Best val loss: {best_val_loss:.6f}")
    print(f"✓ Total positions: {len(X_w):,}")
    print()

    return {"val_loss": best_val_loss, "positions": len(X_w)}


@app.local_entrypoint()
def main():
    print()
    print("=" * 70)
    print(" NNUE CHESS BOT - MODAL GPU TRAINING")
    print("=" * 70)
    print()
    print("This will:")
    print("  • Use your local data (no upload needed!)")
    print("  • Train on Modal's T4 GPU")
    print("  • Take ~30-60 minutes")
    print("  • Cost ~$0.50-1.00 (you have 30 free credits)")
    print()

    resp = input("Start training? (y/n): ")
    if resp.lower() != 'y':
        print("Cancelled")
        return

    print()
    print("🚀 Starting training on Modal GPU...")
    print()

    result = train.remote(
        max_games_per_file=10000,
        max_files=4,
        num_epochs=50,
    )

    print()
    print("=" * 70)
    print(" SUCCESS!")
    print("=" * 70)
    print(f"✓ Val loss: {result['val_loss']:.6f}")
    print(f"✓ Positions: {result['positions']:,}")
    print()
    print("Download your model:")
    print("  modal volume get chess-models nnue_best.pth .")
    print()
    print("Then play:")
    print("  python3 nnue_play.py")
    print()
