#!/usr/bin/env python3
"""
Download and prepare pre-labeled chess data for training.
Much faster than generating data yourself!
"""
import requests
import json
import csv
from pathlib import Path
from tqdm import tqdm


def download_lichess_puzzles(output_file: str = "data/lichess_puzzles.csv", max_positions: int = 50000):
    """
    Download Lichess puzzle database.
    Contains millions of positions with evaluations.
    """
    print("Downloading Lichess puzzle database...")
    print("This contains pre-labeled positions (no Stockfish needed!)")

    # Lichess puzzle database URL
    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

    print(f"\nDownload URL: {url}")
    print("\nNote: This is a .zst compressed file (~500MB)")
    print("You'll need to decompress it:")
    print("  brew install zstd  # or apt-get install zstd")
    print("  zstd -d lichess_db_puzzle.csv.zst")
    print("\nThen run this script again to convert to training format.")

    return url


def convert_lichess_puzzles_to_training_data(
    puzzle_file: str = "data/lichess_db_puzzle.csv",
    output_file: str = "data/training_data.json",
    max_positions: int = 50000,
):
    """
    Convert Lichess puzzle CSV to training format.

    CSV format:
    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
    """
    print(f"Converting Lichess puzzles to training format...")
    print(f"Input: {puzzle_file}")
    print(f"Output: {output_file}")
    print(f"Max positions: {max_positions}")

    if not Path(puzzle_file).exists():
        print(f"\nERROR: File not found: {puzzle_file}")
        print("\nDownload it first:")
        print("  wget https://database.lichess.org/lichess_db_puzzle.csv.zst")
        print("  zstd -d lichess_db_puzzle.csv.zst")
        return

    training_data = []

    with open(puzzle_file, 'r') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(tqdm(reader, total=max_positions)):
            if i >= max_positions:
                break

            fen = row['FEN']
            rating = int(row['Rating'])

            # Convert puzzle rating to centipawns (rough approximation)
            # Rating 1500 = beginner puzzle = roughly equal
            # Rating 2500 = master puzzle = roughly +3 pawns advantage
            # This is VERY rough but gives us training signal
            cp_score = (rating - 1500) * 2  # 100 rating points = ~200 cp

            # Clamp to reasonable range
            cp_score = max(-2000, min(2000, cp_score))

            training_data.append({
                'fen': fen,
                'score': cp_score,
                'source': 'lichess_puzzle',
            })

    print(f"\n✓ Converted {len(training_data)} positions")

    # Save
    with open(output_file, 'w') as f:
        json.dump(training_data, f)

    print(f"✓ Saved to {output_file}")
    print(f"\nYou can now train with:")
    print(f"  modal run --detach train_modal_simple.py::train_model")


def download_ccrl_evals():
    """
    Alternative: CCRL evaluation database
    """
    print("CCRL Option:")
    print("Visit: http://www.computerchess.org.uk/ccrl/")
    print("Download their opening book or evaluation files")


def use_existing_nnue_net():
    """
    Option: Download an existing NNUE network
    """
    print("\n" + "="*70)
    print("Alternative: Use Pre-Trained NNUE Networks")
    print("="*70)
    print("\nYou can download already-trained networks:")
    print("\n1. Stockfish NNUE nets:")
    print("   https://tests.stockfishchess.org/nns")
    print("   These are for Stockfish's architecture, not directly compatible")
    print("\n2. Lc0 (Leela Chess Zero) nets:")
    print("   https://lczero.org/play/networks/bestnets/")
    print("   Different architecture, but you could study the approach")
    print("\n3. Train your own (what you're doing!):")
    print("   More control, learns from your choice of data")


if __name__ == '__main__':
    print("="*70)
    print("Chess Training Data Options")
    print("="*70)

    print("\nOption 1: Lichess Puzzles (RECOMMENDED)")
    print("-" * 70)
    url = download_lichess_puzzles()
    print(f"\nDownload with:")
    print(f"  wget {url}")
    print(f"  zstd -d lichess_db_puzzle.csv.zst")
    print(f"  python3 download_data.py convert")

    print("\n\nOption 2: Generate from PGN (what you're doing now)")
    print("-" * 70)
    print("Pros: Exact Stockfish labels, custom depth")
    print("Cons: Slow, expensive on Modal")

    print("\n\nOption 3: Use existing NNUE")
    print("-" * 70)
    use_existing_nnue_net()

    # Check if user wants to convert
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        convert_lichess_puzzles_to_training_data()
