"""
Utility functions for the chess engine.
Includes model conversion, benchmarking, and debugging tools.
"""
import torch
import numpy as np
import chess
from pathlib import Path
from typing import Optional
from nnue import NNUENetwork, NNUEEvaluator


def convert_torch_to_numpy(torch_path: str, numpy_path: str):
    """
    Convert PyTorch model to NumPy weights for fast inference.

    Args:
        torch_path: Path to PyTorch checkpoint (.pt/.pth)
        numpy_path: Output path for NumPy weights (.npz)
    """
    print(f"Converting {torch_path} to {numpy_path}...")

    # Load PyTorch model
    checkpoint = torch.load(torch_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Extract weights
    ft_weights = state_dict['ft.weight'].t().numpy()
    ft_bias = state_dict['ft.bias'].numpy()
    fc1_weights = state_dict['fc1.weight'].t().numpy()
    fc1_bias = state_dict['fc1.bias'].numpy()
    fc2_weights = state_dict['fc2.weight'].t().numpy()
    fc2_bias = state_dict['fc2.bias'].numpy()[0]

    # Save as NumPy
    np.savez(
        numpy_path,
        ft_weights=ft_weights,
        ft_bias=ft_bias,
        fc1_weights=fc1_weights,
        fc1_bias=fc1_bias,
        fc2_weights=fc2_weights,
        fc2_bias=fc2_bias,
    )

    print(f"Saved NumPy weights to {numpy_path}")

    # Print model info
    total_params = sum(w.size for w in [ft_weights, ft_bias, fc1_weights, fc1_bias, fc2_weights])
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {Path(numpy_path).stat().st_size / 1024 / 1024:.2f} MB")


def benchmark_engine(evaluator: NNUEEvaluator, num_positions: int = 1000):
    """
    Benchmark NNUE evaluator speed.

    Args:
        evaluator: NNUE evaluator to benchmark
        num_positions: Number of positions to evaluate
    """
    import time
    import random

    print(f"Benchmarking NNUE evaluator ({num_positions} positions)...")

    # Generate random positions
    boards = []
    for _ in range(num_positions):
        board = chess.Board()

        # Make random moves
        for _ in range(random.randint(10, 40)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)

        boards.append(board)

    # Benchmark
    start = time.time()
    for board in boards:
        evaluator.reset_accumulators(board)
        evaluator.evaluate(board)

    elapsed = time.time() - start

    evals_per_sec = num_positions / elapsed

    print(f"Time: {elapsed:.2f}s")
    print(f"Evaluations per second: {evals_per_sec:,.0f}")


def perft_test(depth: int = 5):
    """
    Run perft test to verify move generation correctness.

    Args:
        depth: Perft depth

    Returns:
        Dictionary of results
    """
    from engine import ChessEngine

    print(f"Running perft test (depth {depth})...")

    engine = ChessEngine()

    # Known perft results for starting position
    expected = {
        1: 20,
        2: 400,
        3: 8902,
        4: 197281,
        5: 4865609,
        6: 119060324,
    }

    results = {}
    for d in range(1, depth + 1):
        count = engine.perft(d)
        results[d] = count

        expected_count = expected.get(d, None)
        if expected_count:
            status = "✓" if count == expected_count else "✗"
            print(f"Depth {d}: {count:,} nodes {status}")
        else:
            print(f"Depth {d}: {count:,} nodes")

    return results


def test_incremental_updates(num_positions: int = 100):
    """
    Test that incremental NNUE updates match full recalculation.

    Args:
        num_positions: Number of positions to test
    """
    import random

    print(f"Testing incremental NNUE updates ({num_positions} positions)...")

    evaluator = NNUEEvaluator()
    errors = 0

    for i in range(num_positions):
        board = chess.Board()

        # Make random moves
        for _ in range(random.randint(5, 30)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            move = random.choice(legal_moves)

            # Get eval with incremental update
            evaluator.reset_accumulators(board)
            eval_before = evaluator.evaluate(board)

            evaluator.push(board, move)
            board.push(move)
            eval_incremental = evaluator.evaluate(board)

            # Get eval with full reset
            evaluator.reset_accumulators(board)
            eval_full = evaluator.evaluate(board)

            # Compare
            if eval_incremental != eval_full:
                errors += 1
                print(f"Error at position {i}: incremental={eval_incremental}, full={eval_full}")
                print(f"FEN: {board.fen()}")
                print(f"Move: {move}")
                break

    if errors == 0:
        print(f"✓ All {num_positions} positions passed!")
    else:
        print(f"✗ {errors} errors found")


def analyze_game(pgn_path: str, engine_depth: int = 8):
    """
    Analyze a game from PGN and print evaluations.

    Args:
        pgn_path: Path to PGN file
        engine_depth: Search depth for analysis
    """
    import chess.pgn
    from engine import ChessEngine

    print(f"Analyzing game from {pgn_path}...")

    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print("Could not read game")
        return

    engine = ChessEngine()

    print(f"Event: {game.headers.get('Event', 'Unknown')}")
    print(f"White: {game.headers.get('White', 'Unknown')}")
    print(f"Black: {game.headers.get('Black', 'Unknown')}")
    print(f"Result: {game.headers.get('Result', '*')}")
    print()

    board = game.board()
    move_num = 1

    for move in game.mainline_moves():
        # Analyze position before move
        engine.set_position(board.fen())
        best_move, score, info = engine.get_best_move(depth=engine_depth)

        move_san = board.san(move)
        best_move_san = board.san(best_move)

        # Show if played move differs from engine
        if move != best_move:
            inaccuracy = abs(score - info['score_cp'])
            marker = "??" if inaccuracy > 200 else "?" if inaccuracy > 100 else ""
        else:
            marker = ""

        side = "White" if board.turn else "Black"
        print(f"{move_num}. {side}: {move_san} {marker}")
        print(f"   Eval: {score:+d} cp, Engine: {best_move_san} ({info['score_cp']:+d} cp)")

        if info['pv']:
            print(f"   PV: {' '.join(info['pv'][:5])}")

        print()

        board.push(move)

        if not board.turn:
            move_num += 1


def create_opening_book(pgn_paths: list, output_path: str, min_count: int = 5):
    """
    Create simple opening book from PGN database.

    Args:
        pgn_paths: List of PGN file paths
        output_path: Output path for opening book (JSON)
        min_count: Minimum occurrence count for inclusion
    """
    import chess.pgn
    import json
    from collections import defaultdict

    print(f"Creating opening book from {len(pgn_paths)} PGN files...")

    positions = defaultdict(lambda: defaultdict(int))

    for pgn_path in pgn_paths:
        print(f"Processing {pgn_path}...")

        with open(pgn_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()

                # Only store first 15 plies
                for i, move in enumerate(game.mainline_moves()):
                    if i >= 15:
                        break

                    fen = board.fen()
                    move_uci = move.uci()

                    positions[fen][move_uci] += 1

                    board.push(move)

    # Filter by min count
    book = {}
    for fen, moves in positions.items():
        filtered_moves = {move: count for move, count in moves.items() if count >= min_count}
        if filtered_moves:
            book[fen] = filtered_moves

    # Save
    with open(output_path, 'w') as f:
        json.dump(book, f)

    print(f"Saved opening book with {len(book)} positions to {output_path}")


if __name__ == '__main__':
    # Run tests
    print("Running NNUE engine tests...")
    print("=" * 60)

    print("\n1. Perft test (move generation)")
    perft_test(depth=5)

    print("\n2. Incremental update test")
    test_incremental_updates(num_positions=50)

    print("\n3. Evaluation benchmark")
    from nnue import NNUEEvaluator
    evaluator = NNUEEvaluator()
    benchmark_engine(evaluator, num_positions=1000)

    print("\n✓ All tests complete!")
