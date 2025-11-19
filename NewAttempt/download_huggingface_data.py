#!/usr/bin/env python3
"""
Download and convert Lichess chess position evaluations from Hugging Face.

This dataset has 316 million Stockfish-evaluated positions with depth information.
Much better than puzzle ratings!
"""
import json
from pathlib import Path
from tqdm import tqdm


def download_and_convert(
    output_file: str = "data/training_data_hf.json",
    num_positions: int = 100000,
    min_depth: int = 15,
    max_depth: int = 50,
):
    """
    Download positions from Hugging Face and convert to training format.

    Args:
        output_file: Where to save the converted data
        num_positions: How many positions to use (default 100k)
        min_depth: Minimum Stockfish depth (default 15 for quality)
        max_depth: Maximum depth to consider (avoid extremely deep positions)
    """
    print("=" * 70)
    print("Downloading Lichess Position Evaluations from Hugging Face")
    print("=" * 70)
    print(f"Target positions: {num_positions:,}")
    print(f"Depth range: {min_depth}-{max_depth}")
    print()

    # Import datasets library (user needs to install)
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed!")
        print("\nInstall with:")
        print("  pip install datasets")
        print("\nOr if that's slow, use:")
        print("  pip install datasets[parquet]")
        return

    print("Loading dataset from Hugging Face...")
    print("(This will download ~37.7 GB, but it's cached after first download)")
    print()

    # Stream the dataset to avoid loading all 316M positions into memory
    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True  # Important: don't load all 316M into RAM!
    )

    training_data = []
    positions_checked = 0
    positions_kept = 0

    print(f"Processing positions (filtering for depth {min_depth}-{max_depth})...")
    print()

    # Iterate through dataset
    for item in tqdm(dataset, total=num_positions, desc="Extracting"):
        positions_checked += 1

        # Get fields
        fen = item.get('Fen')
        depth = item.get('Depth')
        cp = item.get('CP')  # Centipawn score
        mate = item.get('Mate')  # Mate in X moves

        # Skip if missing data
        if not fen or depth is None:
            continue

        # Filter by depth (quality control)
        if depth < min_depth or depth > max_depth:
            continue

        # Convert evaluation to centipawns
        if mate is not None:
            # Mate score: convert to large centipawn value
            if mate > 0:
                score = 10000 - mate * 10  # Mate for white
            else:
                score = -10000 - mate * 10  # Mate for black
        elif cp is not None:
            score = cp
        else:
            # No evaluation available
            continue

        # Clamp to reasonable range
        score = max(-10000, min(10000, score))

        training_data.append({
            'fen': fen,
            'score': score,
            'depth': depth,
            'source': 'lichess_hf',
        })

        positions_kept += 1

        # Stop when we have enough
        if positions_kept >= num_positions:
            break

        # Progress update every 10k positions checked
        if positions_checked % 10000 == 0:
            keep_rate = (positions_kept / positions_checked) * 100
            print(f"  Checked: {positions_checked:,} | Kept: {positions_kept:,} ({keep_rate:.1f}%)")

    print()
    print(f"✓ Collected {len(training_data):,} positions")
    print(f"  Checked {positions_checked:,} total positions")
    print(f"  Keep rate: {(positions_kept/positions_checked)*100:.1f}%")
    print()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(training_data, f)

    print(f"✓ Saved to {output_file}")
    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("Upload to Modal volume:")
    print(f"  modal volume put chessbot-data {output_file} training_data.json")
    print()
    print("Train the model:")
    print("  modal run --detach train_modal_simple.py::train_model \\")
    print("    --data-file training_data.json \\")
    print("    --epochs 10 \\")
    print("    --batch-size 512 \\")
    print("    --learning-rate 0.001")
    print()
    print(f"With {len(training_data):,} positions, this should train a much better model!")
    print()


if __name__ == '__main__':
    import sys

    # Parse arguments
    num_positions = 100000  # Default
    min_depth = 15

    if len(sys.argv) > 1:
        num_positions = int(sys.argv[1])
    if len(sys.argv) > 2:
        min_depth = int(sys.argv[2])

    print()
    print("Usage: python3 download_huggingface_data.py [num_positions] [min_depth]")
    print(f"Using: {num_positions:,} positions, min depth {min_depth}")
    print()

    download_and_convert(
        num_positions=num_positions,
        min_depth=min_depth,
    )
