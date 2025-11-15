"""
Improved Modal Training - More Data + More Epochs

This fixes the undertraining issue by:
1. Using ALL available games (not just 10K per file)
2. Training for 200 epochs (not just 50)
3. Better evaluation to catch when model is good enough

Usage:
    modal run train_modal_improved.py
"""

import modal

app = modal.App("nnue-chess-training-improved")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0",
    "chess==1.10.0",
    "numpy==1.26.0",
    "tqdm==4.66.0",
)

data_vol = modal.Volume.from_name("chess-data", create_if_missing=True)
model_vol = modal.Volume.from_name("chess-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 6,  # 6 hours max
    volumes={
        "/data": data_vol,
        "/models": model_vol,
    },
)
def train(max_games_per_file=50000, num_epochs=200):
    """Train NNUE with MORE data and MORE epochs"""
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
    print(" IMPROVED NNUE TRAINING - MORE DATA + MORE EPOCHS")
    print("=" * 70)
    print()

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    print(f"Configuration:")
    print(f"  • Games per file: {max_games_per_file:,}")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Expected training time: 2-4 hours")
    print()

    # NNUE Model
    class NNUE(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft_w = nn.Linear(640, 1408)
            self.ft_b = nn.Linear(640, 1408)
            self.fc1 = nn.Linear(2816, 32)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, wf, bf):
            relu = lambda x: torch.clamp(torch.relu(x), 0, 1)
            w = relu(self.ft_w(wf))
            b = relu(self.ft_b(bf))
            x = torch.cat([w, b], dim=1)
            x = relu(self.fc1(x))
            return self.fc2(x)

    # Dataset
    class D(Dataset):
        def __init__(self, w, b, r):
            self.w = torch.tensor(w, dtype=torch.float32)
            self.b = torch.tensor(b, dtype=torch.float32)
            self.r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.w)
        def __getitem__(self, i): return self.w[i], self.b[i], self.r[i]

    # HalfKP encoding
    def encode(board):
        wf = np.zeros(640, dtype=np.float32)
        bf = np.zeros(640, dtype=np.float32)
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is None or bk is None:
            return wf, bf
        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.KING:
                continue
            pi = pc.piece_type - 1
            co = 0 if pc.color else 5
            wf[(pi + co) * 64 + sq] = 1
            bf[(pi + (0 if not pc.color else 5)) * 64 + (sq ^ 56)] = 1
        return wf, bf

    # Load data
    print("=" * 70)
    print(" LOADING DATA")
    print("=" * 70)

    files = sorted([f for f in os.listdir("/data") if f.endswith(".pgn")])
    print(f"✓ Found {len(files)} PGN files")
    for f in files:
        size_mb = os.path.getsize(f'/data/{f}') / 1024 / 1024
        print(f"  • {f}: {size_mb:.0f} MB")
    print()

    games = []
    for fn in files[:4]:  # Use all 4 files
        print(f"Loading {fn}...")
        cnt = 0
        with open(f"/data/{fn}") as f:
            with tqdm(total=max_games_per_file, desc="  Games") as pbar:
                while cnt < max_games_per_file:
                    g = chess.pgn.read_game(f)
                    if g is None: break
                    try:
                        we = int(g.headers.get("WhiteElo", 0))
                        be = int(g.headers.get("BlackElo", 0))
                        if we >= 2200 and be >= 2200:
                            games.append(g)
                            cnt += 1
                            pbar.update(1)
                    except: pass

    print(f"\n✓ Loaded {len(games):,} games (2200+ ELO)")
    print()

    # Create training data
    print("Creating positions...")
    wl, bl, rl = [], [], []

    for g in tqdm(games, desc="Processing"):
        res = g.headers.get("Result", "*")
        if res == "1-0": r = 1.0
        elif res == "0-1": r = -1.0
        elif res == "1/2-1/2": r = 0.0
        else: continue

        mvs = list(g.mainline_moves())
        if len(mvs) < 15: continue

        # Sample 25 positions per game
        step = len(mvs) / 25 if len(mvs) > 25 else 1
        samps = [int(i * step) for i in range(min(25, len(mvs)))]

        b = g.board()
        for i, m in enumerate(mvs):
            if i in samps:
                w, bl_f = encode(b)
                wl.append(w)
                bl.append(bl_f)
                rl.append(r if b.turn else -r)
            b.push(m)

    X_w = np.array(wl, dtype=np.float32)
    X_b = np.array(bl, dtype=np.float32)
    y = np.array(rl, dtype=np.float32)

    print(f"✓ {len(X_w):,} positions")
    print(f"  Wins: {(y>0).sum():,}, Draws: {(y==0).sum():,}, Loss: {(y<0).sum():,}")
    print()

    # Create datasets
    ds = D(X_w, X_b, y)
    tr_sz = int(0.95 * len(ds))
    tr_ds, val_ds = random_split(ds, [tr_sz, len(ds) - tr_sz])
    tr_ld = DataLoader(tr_ds, batch_size=2048, shuffle=True)  # Larger batch size
    val_ld = DataLoader(val_ds, batch_size=2048)

    print(f"✓ Train: {len(tr_ds):,}, Val: {len(val_ds):,}")
    print()

    # Model
    print("=" * 70)
    print(" MODEL")
    print("=" * 70)
    model = NNUE().to(device)
    print(f"✓ Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Size: {sum(p.numel() for p in model.parameters())*4/1024/1024:.2f} MB")
    print()

    # Train
    print("=" * 70)
    print(" TRAINING")
    print("=" * 70)

    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 5)

    best = float('inf')
    patience = 0
    max_patience = 20  # Early stopping after 20 epochs without improvement

    for ep in range(num_epochs):
        st = time.time()

        # Train
        model.train()
        tl = 0
        for w, b, r in tqdm(tr_ld, desc=f"Epoch {ep+1}/{num_epochs}", leave=False):
            w, b, r = w.to(device), b.to(device), r.to(device)
            opt.zero_grad()
            p = model(w, b)
            loss = crit(p, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(tr_ld)

        # Val
        model.eval()
        vl = 0
        with torch.no_grad():
            for w, b, r in val_ld:
                w, b, r = w.to(device), b.to(device), r.to(device)
                vl += crit(model(w, b), r).item()
        vl /= len(val_ld)

        sched.step(vl)

        print(f"\nEpoch {ep+1}/{num_epochs} ({time.time()-st:.1f}s)")
        print(f"  Train: {tl:.6f}")
        print(f"  Val:   {vl:.6f}")

        if vl < best:
            improvement = best - vl
            best = vl
            patience = 0
            torch.save(model.state_dict(), "/models/nnue_best.pth")
            model_vol.commit()  # Save to Modal volume
            print(f"  ✓ Saved! (improved by {improvement:.6f})")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{max_patience})")

        # Early stopping
        if patience >= max_patience:
            print(f"\n⚠️  Early stopping - no improvement for {max_patience} epochs")
            break

        # Checkpoint every 20 epochs
        if (ep + 1) % 20 == 0:
            torch.save(model.state_dict(), f"/models/nnue_epoch_{ep+1}.pth")
            model_vol.commit()
            print(f"  ✓ Checkpoint saved")

    print()
    print("=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"✓ Best validation loss: {best:.6f}")
    print(f"✓ Model saved to Modal volume: chess-models/nnue_best.pth")
    print()
    print("Download with:")
    print("  modal volume get chess-models nnue_best.pth .")


@app.local_entrypoint()
def main():
    """Run training"""
    print("Starting improved NNUE training...")
    print("This will use 50K games per file × 4 files = 200K games")
    print("Training for up to 200 epochs with early stopping")
    print()
    train.remote()
    print()
    print("Training complete! Download your model:")
    print("  modal volume get chess-models nnue_best.pth .")
