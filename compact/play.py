#!/usr/bin/env python3
"""
Simple script to play against the chess bot
"""

import os
import sys
from engine import ChessEngine, play_game


def main():
    print("=" * 60)
    print("COMPACT CHESS BOT - Under 10MB Challenge")
    print("=" * 60)
    print()

    # Find the best available model
    model_files = ["model_int8.pth", "best_model.pth", "final_model.pth"]
    model_path = None

    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break

    if model_path is None:
        print("❌ No trained model found!")
        print("\nPlease train a model first:")
        print("  python train.py")
        print()
        print("Then optionally quantize it:")
        print("  python quantize.py")
        sys.exit(1)

    # Get file size
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Loading model: {model_path} ({model_size:.2f} MB)")

    # Load engine
    try:
        engine = ChessEngine(model_path)
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Get search depth
    print("Select difficulty:")
    print("  1. Easy (depth=2)")
    print("  2. Medium (depth=3)")
    print("  3. Hard (depth=4)")
    print("  4. Very Hard (depth=5)")
    print()

    while True:
        try:
            difficulty = input("Enter difficulty (1-4) [default: 2]: ").strip()
            if difficulty == "":
                depth = 3
                break
            difficulty = int(difficulty)
            if 1 <= difficulty <= 4:
                depth = difficulty + 1
                break
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")

    print(f"\nSearch depth: {depth}")
    print()

    # Play the game
    play_game(engine, search_depth=depth)


if __name__ == "__main__":
    main()
