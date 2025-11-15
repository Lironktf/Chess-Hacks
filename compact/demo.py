#!/usr/bin/env python3
"""
Demo script - Test the engine with a randomly initialized model
This demonstrates the engine works even before training
(though it won't play well until trained!)
"""

import torch
import chess
from model import CompactChessNet
from engine import ChessEngine


def create_random_model():
    """Create a randomly initialized model for testing"""
    model = CompactChessNet()
    torch.save(model.state_dict(), "random_model.pth")
    return "random_model.pth"


def demo_position_evaluation():
    """Demo: Evaluate some positions"""
    print("=" * 60)
    print("DEMO: Position Evaluation")
    print("=" * 60)
    print()

    model_path = create_random_model()
    engine = ChessEngine(model_path)

    # Starting position
    board = chess.Board()
    eval_start = engine.evaluate_position(board)
    print(f"Starting position:")
    print(board)
    print(f"Evaluation: {eval_start:.4f}")
    print()

    # After 1. e4
    board.push_uci("e2e4")
    eval_e4 = engine.evaluate_position(board)
    print(f"After 1. e4:")
    print(board)
    print(f"Evaluation: {eval_e4:.4f}")
    print()

    # Scholar's mate position (White winning)
    board = chess.Board()
    for move in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
        board.push_uci(move)

    eval_checkmate = engine.evaluate_position(board)
    print(f"Checkmate position (Scholar's mate):")
    print(board)
    print(f"Evaluation: {eval_checkmate:.4f}")
    print(f"Game over: {board.is_checkmate()}")
    print()


def demo_move_search():
    """Demo: Search for best move"""
    print("=" * 60)
    print("DEMO: Move Search")
    print("=" * 60)
    print()

    model_path = create_random_model()
    engine = ChessEngine(model_path)

    board = chess.Board()
    print("Finding best move from starting position...")
    print(board)
    print()

    move, eval_score = engine.search(board, depth=2)
    print(f"Best move found: {move}")
    print(f"Evaluation: {eval_score:.4f}")
    print(f"Nodes searched: {engine.nodes_searched:,}")
    print()


def demo_short_game():
    """Demo: Play a short game between two random models"""
    print("=" * 60)
    print("DEMO: Short Game (Random Model)")
    print("=" * 60)
    print()

    model_path = create_random_model()
    engine = ChessEngine(model_path)

    board = chess.Board()

    for move_num in range(5):  # Play 5 moves each
        print(f"Move {move_num + 1}:")
        print(board)
        print()

        if board.is_game_over():
            break

        # White's turn
        move, eval_score = engine.search(board, depth=2)
        print(f"White plays: {move} (eval: {eval_score:.3f})")
        board.push(move)

        if board.is_game_over():
            break

        # Black's turn
        move, eval_score = engine.search(board, depth=2)
        print(f"Black plays: {move} (eval: {eval_score:.3f})")
        board.push(move)
        print()

    print("Final position:")
    print(board)
    print()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("COMPACT CHESS BOT - DEMO")
    print("=" * 60)
    print()
    print("This demo uses a RANDOMLY INITIALIZED model.")
    print("The moves won't be good until you train the model!")
    print()
    print("To train a real model:")
    print("  1. Put PGN files in ../data/")
    print("  2. Run: python3 train.py")
    print()

    try:
        demo_position_evaluation()
        demo_move_search()
        demo_short_game()

        print("=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print()
        print("The engine is working correctly!")
        print()
        print("Next steps:")
        print("  1. Train the model:   python3 train.py")
        print("  2. Quantize it:       python3 quantize.py")
        print("  3. Play against it:   python3 play.py")
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
