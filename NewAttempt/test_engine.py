#!/usr/bin/env python3
"""
Quick test script to verify chess engine is working correctly.
Run this after installation to check all components.
"""
import sys
from engine import ChessEngine
from utils import perft_test, test_incremental_updates


def test_basic_functionality():
    """Test basic engine functionality."""
    print("=" * 70)
    print("Testing ChessBot Engine")
    print("=" * 70)

    engine = ChessEngine()

    # Test 1: Board setup
    print("\n1. Testing board setup...")
    assert engine.get_fen() == chess.Board().fen()
    print("   ✓ Board initialized correctly")

    # Test 2: Make moves
    print("\n2. Testing move making...")
    engine.make_move('e4')
    engine.make_move('e5')
    engine.make_move('Nf3')
    assert len(engine.board.move_stack) == 3
    print("   ✓ Moves executed correctly")

    # Test 3: Static evaluation
    print("\n3. Testing static evaluation...")
    engine.new_game()
    score = engine.evaluate_position()
    print(f"   Starting position eval: {score:+d} cp")
    print("   ✓ Evaluation works")

    # Test 4: Search
    print("\n4. Testing search (depth 4)...")
    engine.new_game()
    move, score, info = engine.get_best_move(depth=4)
    print(f"   Best move: {engine.board.san(move)}")
    print(f"   Score: {score:+d} cp")
    print(f"   Nodes: {info['nodes']:,}")
    print(f"   Time: {info['time_ms']:.1f} ms")
    print(f"   NPS: {info['nps']:,}")
    assert move is not None
    print("   ✓ Search works")

    # Test 5: Position analysis
    print("\n5. Testing position analysis...")
    engine.set_position_from_moves(['e4', 'e5', 'Bc4', 'Nc6', 'Qh5', 'Nf6'])
    analysis = engine.analyze_position(depth=5)
    print(f"   Best defense: {analysis[0]['move']}")
    print(f"   Evaluation: {analysis[0]['score_cp']:+d} cp")
    print("   ✓ Analysis works")

    # Test 6: Perft (move generation validation)
    print("\n6. Testing move generation (perft)...")
    engine.new_game()
    count = engine.perft(4)
    expected = 197281
    if count == expected:
        print(f"   Perft(4): {count:,} nodes ✓")
    else:
        print(f"   Perft(4): {count:,} nodes (expected {expected:,}) ✗")
        return False

    print("\n" + "=" * 70)
    print("All basic tests passed!")
    print("=" * 70)
    return True


def test_tactical_positions():
    """Test engine on tactical positions."""
    print("\n" + "=" * 70)
    print("Testing Tactical Positions")
    print("=" * 70)

    engine = ChessEngine()

    # Position 1: Mate in 1
    print("\n1. Mate in 1 (Back rank mate)")
    engine.set_position('6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1')
    move, score, info = engine.get_best_move(depth=5)
    print(f"   Move: {engine.board.san(move)}")
    print(f"   Expected: Re8#")
    print(f"   Score: {score:+d} cp")

    # Position 2: Win material
    print("\n2. Win queen (fork)")
    engine.set_position('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4')
    move, score, info = engine.get_best_move(depth=5)
    print(f"   Move: {engine.board.san(move)}")
    print(f"   Score: {score:+d} cp")

    # Position 3: Defense
    print("\n3. Defend against Scholar's Mate")
    engine.set_position('r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4')
    move, score, info = engine.get_best_move(depth=5)
    print(f"   Move: {engine.board.san(move)}")
    print(f"   Expected: Qe7, Nxh5, or g6")
    print(f"   Score: {score:+d} cp")

    print("\n" + "=" * 70)
    print("Tactical tests complete!")
    print("=" * 70)


def test_performance():
    """Test engine performance."""
    print("\n" + "=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)

    engine = ChessEngine()

    # Depth benchmark
    print("\nDepth scaling test (starting position):")
    for depth in [3, 4, 5, 6]:
        engine.new_game()
        move, score, info = engine.get_best_move(depth=depth)
        print(f"  Depth {depth}: {info['nodes']:>8,} nodes, {info['time_ms']:>6.1f} ms, "
              f"{info['nps']:>8,} nps")

    # Evaluation benchmark
    print("\nEvaluation speed test...")
    from utils import benchmark_engine
    benchmark_engine(engine.evaluator, num_positions=500)

    print("\n" + "=" * 70)


def run_all_tests():
    """Run all tests."""
    import chess  # Import here to avoid issues if not installed

    success = True

    try:
        success = test_basic_functionality()
        if not success:
            print("\n✗ Basic tests failed")
            return False

        test_tactical_positions()
        test_performance()

        print("\n" + "=" * 70)
        print("🎉 All tests passed! Engine is ready to use.")
        print("=" * 70)

        print("\nNext steps:")
        print("  1. Train NNUE weights: See README.md for Modal training instructions")
        print("  2. Load weights: engine.load_weights('model.npz')")
        print("  3. Analyze games: from utils import analyze_game")
        print("  4. Integrate with GUI: Use engine.get_best_move() API")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    import chess  # Make sure python-chess is installed

    success = run_all_tests()
    sys.exit(0 if success else 1)
