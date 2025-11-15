#!/usr/bin/env python3
"""
Analyze total size of the chess bot (RAM + VRAM)

This script calculates the total memory footprint to verify it's under 10MB
"""

import os
import sys
import torch
from nnue_model import NNUE
from opening_book import OpeningBook


def analyze_size():
    """Analyze total size of chess bot"""
    print("=" * 60)
    print("CHESS BOT SIZE ANALYSIS")
    print("=" * 60)
    print()

    total_size = 0

    # 1. Model file size
    print("1. MODEL FILE")
    print("-" * 60)

    model_files = ["nnue_best.pth", "nnue_final.pth", "nnue_model.pth", "nnue_random.pth"]
    model_path = None

    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break

    if model_path:
        model_file_size = os.path.getsize(model_path) / 1024 / 1024
        print(f"Model file: {model_path}")
        print(f"Size: {model_file_size:.2f} MB")
        total_size += model_file_size
    else:
        print("No model file found (creating estimate)")
        model = NNUE(hidden1_size=512, hidden2_size=32)
        model_file_size = model.get_size_mb()
        print(f"Estimated size: {model_file_size:.2f} MB")
        total_size += model_file_size

    print()

    # 2. Runtime memory
    print("2. RUNTIME MEMORY")
    print("-" * 60)

    # Model parameters in memory (same as file for float32)
    print(f"Model parameters: {model_file_size:.2f} MB")

    # Activations (batch_size=1)
    # Feature transformer: 2 * 512 = 1024 floats
    # FC1: 32 floats
    # FC2: 1 float
    # Total: ~1057 floats * 4 bytes = ~4KB
    activation_size = 1057 * 4 / 1024 / 1024
    print(f"Activations: {activation_size:.3f} MB")

    # Board representation (minimal)
    board_size = 0.001  # Negligible
    print(f"Board state: {board_size:.3f} MB")

    runtime_size = model_file_size + activation_size + board_size
    print(f"\nSubtotal (runtime): {runtime_size:.2f} MB")
    print()

    # 3. Transposition table
    print("3. TRANSPOSITION TABLE")
    print("-" * 60)
    tt_size = 2.2  # Configured to 2.2MB (tuned for 10MB limit)
    print(f"TT size: {tt_size:.2f} MB")
    print()

    # 4. Opening book
    print("4. OPENING BOOK")
    print("-" * 60)
    book = OpeningBook()
    book_size = book.size_mb()
    print(f"Positions: {len(book.book)}")
    print(f"Size: {book_size:.3f} MB")
    print()

    # 5. Search overhead
    print("5. SEARCH OVERHEAD")
    print("-" * 60)
    # Killer moves, history table, etc.
    search_overhead = 0.5
    print(f"Killer moves, history, etc: {search_overhead:.2f} MB")
    print()

    # Total
    print("=" * 60)
    print("TOTAL SIZE (RAM + VRAM)")
    print("=" * 60)

    total = runtime_size + tt_size + book_size + search_overhead
    print(f"\nBreakdown:")
    print(f"  Model (file + runtime):  {runtime_size:.2f} MB")
    print(f"  Transposition table:     {tt_size:.2f} MB")
    print(f"  Opening book:            {book_size:.3f} MB")
    print(f"  Search overhead:         {search_overhead:.2f} MB")
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total:.2f} MB")
    print(f"{'=' * 60}")
    print()

    if total < 10.0:
        margin = 10.0 - total
        print(f"✅ UNDER 10MB LIMIT!")
        print(f"   Margin: {margin:.2f} MB ({margin/10*100:.1f}% remaining)")
    else:
        over = total - 10.0
        print(f"❌ OVER 10MB LIMIT!")
        print(f"   Excess: {over:.2f} MB ({over/10*100:.1f}% over)")

    print()

    # Recommendations
    print("=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    if total < 10.0:
        # Can increase size for better strength
        margin = 10.0 - total
        print(f"\nYou have {margin:.2f} MB available.")
        print("To improve strength, you could:")
        print(f"  • Increase NNUE hidden layer to {int(512 + margin * 200)} neurons")
        print(f"  • Increase transposition table to {tt_size + margin/2:.1f} MB")
        print(f"  • Expand opening book with more positions")
    else:
        over = total - 10.0
        print(f"\nYou need to reduce size by {over:.2f} MB.")
        print("Options:")
        print(f"  • Reduce TT size to {max(0.5, tt_size - over):.1f} MB")
        print(f"  • Use smaller NNUE (hidden1={int(512 - over*150)})")
        print(f"  • Reduce opening book size")

    print()

    return total


def optimize_for_10mb():
    """Calculate optimal configuration for ~9.5MB usage"""
    print("=" * 60)
    print("OPTIMAL CONFIGURATION FOR 10MB BUDGET")
    print("=" * 60)
    print()

    target_size = 9.5  # Leave 0.5MB margin

    # Fixed costs
    book_size = 0.005  # Opening book
    search_overhead = 0.5
    activation_size = 0.01

    # Variable allocations
    tt_size = 2.0  # Transposition table

    remaining = target_size - (book_size + search_overhead + activation_size + tt_size)
    print(f"Budget for model: {remaining:.2f} MB")

    # Calculate optimal hidden layer size
    # Model size ≈ (640 * hidden1 * 2 + hidden1 * 2 * hidden2 + hidden2) * 4 bytes
    # For hidden2=32: size ≈ (1280 * hidden1 + 2080) * 4 / 1024 / 1024

    hidden1_optimal = int((remaining * 1024 * 1024 / 4 - 2080) / 1280)
    hidden1_optimal = (hidden1_optimal // 64) * 64  # Round to multiple of 64

    model = NNUE(hidden1_size=hidden1_optimal, hidden2_size=32)
    model_size = model.get_size_mb()

    print(f"\nOptimal configuration:")
    print(f"  NNUE hidden1: {hidden1_optimal}")
    print(f"  NNUE hidden2: 32")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  TT size: {tt_size:.2f} MB")
    print(f"  Book size: {book_size:.3f} MB")
    print(f"  Search overhead: {search_overhead:.2f} MB")
    print(f"  Activations: {activation_size:.2f} MB")
    print(f"\nTotal: {model_size + tt_size + book_size + search_overhead + activation_size:.2f} MB")
    print()

    if hidden1_optimal < 512:
        print("⚠️  Current model (512 hidden) may be too large!")
        print(f"   Recommended: {hidden1_optimal} hidden neurons")
    else:
        print("✅ Current model (512 hidden) fits within budget")
        if hidden1_optimal > 512:
            print(f"   You could increase to {hidden1_optimal} for better strength")

    print()


if __name__ == "__main__":
    total = analyze_size()
    print()
    optimize_for_10mb()
