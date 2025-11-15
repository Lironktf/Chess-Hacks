"""
Test NNUE Model Quality

This script tests if your NNUE model can properly evaluate positions.
A good model should:
- Prefer winning positions (up material)
- Avoid losing positions (down material)
- Distinguish between good and bad moves
"""

import torch
import chess
import numpy as np
from nnue_model import NNUE, board_to_halfkp_features


def evaluate_position(model, board, device="cpu"):
    """Evaluate a position with the NNUE model"""
    white_feat, black_feat = board_to_halfkp_features(board)

    white_tensor = torch.tensor(white_feat, dtype=torch.float32).unsqueeze(0).to(device)
    black_tensor = torch.tensor(black_feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(white_tensor, black_tensor).item()

    return score


def test_model(model_path):
    """Test model quality on known positions"""

    print("=" * 70)
    print(f" TESTING MODEL: {model_path}")
    print("=" * 70)
    print()

    # Load model
    model = NNUE(hidden1_size=1408, hidden2_size=32)
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle different naming conventions
    if "ft_w.weight" in state_dict and "ft_white.weight" not in state_dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == "ft_w.weight":
                new_state_dict["ft_white.weight"] = value
            elif key == "ft_w.bias":
                new_state_dict["ft_white.bias"] = value
            elif key == "ft_b.weight":
                new_state_dict["ft_black.weight"] = value
            elif key == "ft_b.bias":
                new_state_dict["ft_black.bias"] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    print("✓ Model loaded\n")

    # Test positions
    test_cases = [
        ("Equal (starting position)",
         chess.Board(),
         0.0, 0.1),  # Should be near 0

        ("White up a pawn",
         chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1"),
         0.2, 0.5),  # Should be positive

        ("White up a queen (~9 pawns)",
         chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
         0.5, 1.0),  # Should be strongly positive

        ("Black up a queen (~9 pawns)",
         chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
         -1.0, -0.5),  # Should be strongly negative

        ("White has checkmate in 1",
         chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
         0.8, 1.0),  # Should be very positive

        ("Endgame: White Queen vs Black Pawn",
         chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"),
         0.7, 1.0),  # White winning
    ]

    print("Testing on standard positions:")
    print("-" * 70)

    results = []
    for desc, board, expected_min, expected_max in test_cases:
        score = evaluate_position(model, board)

        # Check if in expected range
        in_range = expected_min <= score <= expected_max
        status = "✓" if in_range else "✗"

        results.append(in_range)

        print(f"{status} {desc}")
        print(f"  Score: {score:.4f} (expected: {expected_min:.2f} to {expected_max:.2f})")
        print()

    # Summary
    print("=" * 70)
    print(" RESULTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)
    percentage = 100 * passed / total

    print(f"Passed: {passed}/{total} ({percentage:.0f}%)")
    print()

    if percentage >= 80:
        print("✓ GOOD MODEL - Ready for play!")
        print("  The model understands material and position well.")
    elif percentage >= 50:
        print("⚠️  MEDIOCRE MODEL - Needs more training")
        print("  The model partially understands chess but needs improvement.")
    else:
        print("✗ POOR MODEL - Needs significant training")
        print("  The model doesn't understand chess well yet.")
        print("  Recommendation: Train for more epochs with more data.")

    return percentage >= 80


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "nnue_best.pth"
    test_model(model_path)
