#!/usr/bin/env python3
"""
Play against the NNUE chess engine

Combines:
- NNUE evaluation
- Advanced search (alpha-beta, quiescence, transposition table)
- Opening book
"""

import os
import sys
import chess
from nnue_engine import NNUEEngine
from opening_book import OpeningBook


def print_board(board: chess.Board, flip=False):
    """Print board nicely"""
    board_str = str(board)
    if flip:
        lines = board_str.split('\n')
        board_str = '\n'.join(reversed(lines))
    print(board_str)


def play_game(engine: NNUEEngine, book: OpeningBook, human_color: chess.Color, depth: int):
    """
    Play a game against the engine

    Args:
        engine: NNUE engine
        book: Opening book
        human_color: Color for human (chess.WHITE or chess.BLACK)
        depth: Search depth
    """
    board = chess.Board()
    moves_from_book = 0

    print("\n" + "=" * 60)
    print("NNUE CHESS ENGINE - ADVANCED")
    print("=" * 60)
    print(f"You are playing as {'White' if human_color == chess.WHITE else 'Black'}")
    print(f"Engine depth: {depth}")
    print("Enter moves in UCI format (e.g., 'e2e4') or type 'quit' to exit")
    print("=" * 60)
    print()

    while not board.is_game_over():
        print_board(board, flip=(human_color == chess.BLACK))
        print()

        if board.turn == human_color:
            # Human's turn
            while True:
                move_str = input("Your move: ").strip().lower()

                if move_str == 'quit':
                    print("Thanks for playing!")
                    return

                if move_str == 'help':
                    print("Legal moves:", ", ".join([m.uci() for m in board.legal_moves]))
                    continue

                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again (or type 'help' for legal moves)")
                except ValueError:
                    print("Invalid format! Use UCI notation (e.g., 'e2e4')")
        else:
            # Engine's turn
            print("Engine thinking...")

            # Try opening book first
            book_move = None
            if book.is_in_book(board):
                book_move = book.get_book_move(board)
                if book_move:
                    print(f"[Using opening book]")
                    move = book_move
                    moves_from_book += 1
                else:
                    move = engine.get_move(board, depth=depth)
            else:
                move = engine.get_move(board, depth=depth)

            print(f"Engine plays: {move}\n")
            board.push(move)

    # Game over
    print_board(board, flip=(human_color == chess.BLACK))
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    print(f"Result: {board.result()}")

    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material!")
    elif board.is_fifty_moves():
        print("Draw by fifty-move rule!")
    elif board.is_repetition():
        print("Draw by repetition!")

    print(f"\nBook moves used: {moves_from_book}")


def main():
    print()
    print("=" * 60)
    print("NNUE CHESS ENGINE - ADVANCED")
    print("=" * 60)
    print()

    # Find model
    model_files = ["nnue_best.pth", "nnue_final.pth", "nnue_model.pth"]
    model_path = None

    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break

    if model_path is None:
        print("❌ No trained NNUE model found!")
        print("\nPlease train a model first:")
        print("  python3 nnue_train.py")
        print()
        print("Or test with a random model:")
        print("  python3 nnue_model.py")
        sys.exit(1)

    # Get file size
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Loading model: {model_path} ({model_size:.2f} MB)")

    # Load engine (with tuned TT to stay under 10MB)
    try:
        engine = NNUEEngine(model_path, tt_size_mb=2.2)
        print("✓ Engine loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Load opening book
    book = OpeningBook()
    print(f"✓ Opening book loaded ({len(book.book)} positions)")
    print()

    # Select color
    print("Choose your color:")
    print("  1. White")
    print("  2. Black")
    while True:
        try:
            choice = input("Enter 1 or 2 [default: 1]: ").strip()
            if choice == "" or choice == "1":
                human_color = chess.WHITE
                break
            elif choice == "2":
                human_color = chess.BLACK
                break
            else:
                print("Please enter 1 or 2")
        except:
            print("Please enter 1 or 2")

    # Select difficulty
    print()
    print("Select difficulty:")
    print("  1. Easy (depth=3)")
    print("  2. Medium (depth=4)")
    print("  3. Hard (depth=5)")
    print("  4. Very Hard (depth=6)")
    print("  5. Expert (depth=7)")
    while True:
        try:
            choice = input("Enter 1-5 [default: 3]: ").strip()
            if choice == "" or choice == "3":
                depth = 5
                break
            elif choice in ["1", "2", "3", "4", "5"]:
                depth = int(choice) + 2
                break
            else:
                print("Please enter 1-5")
        except:
            print("Please enter 1-5")

    # Play
    play_game(engine, book, human_color, depth)


if __name__ == "__main__":
    main()
