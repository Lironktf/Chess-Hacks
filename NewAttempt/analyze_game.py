#!/usr/bin/env python3
"""
Simple game analysis script.
Analyzes your chess games and highlights mistakes.

Usage:
    python3 analyze_game.py
"""
from engine import ChessEngine


def analyze_my_game():
    """Analyze a sample game move by move."""

    # Your game moves (replace with your actual game)
    game_moves = [
        'e4', 'e5',      # 1.
        'Nf3', 'Nc6',    # 2.
        'Bc4', 'Bc5',    # 3.
        'c3', 'Nf6',     # 4.
        'd4', 'exd4',    # 5.
        'cxd4', 'Bb4+',  # 6.
        'Nc3', 'Nxe4',   # 7.
        'O-O', 'Bxc3',   # 8.
        'bxc3', 'O-O',   # 9.
    ]

    print("=" * 70)
    print("GAME ANALYSIS")
    print("=" * 70)

    engine = ChessEngine()
    depth = 6  # Adjust for speed/accuracy tradeoff

    mistakes = []

    for move_num in range(0, len(game_moves), 2):
        # White's move
        if move_num < len(game_moves):
            engine.new_game()
            engine.set_position_from_moves(game_moves[:move_num])

            # Get engine's recommendation
            best_move, best_score, _ = engine.get_best_move(depth=depth)
            played_move = game_moves[move_num]

            # Evaluate played move
            engine.make_move(played_move)
            _, score_after, _ = engine.get_best_move(depth=depth)
            score_loss = abs(best_score + score_after)

            # Classify mistake
            marker = ""
            if score_loss > 200:
                marker = "?? BLUNDER"
                mistakes.append((move_num // 2 + 1, "White", played_move, score_loss))
            elif score_loss > 100:
                marker = "? Mistake"
                mistakes.append((move_num // 2 + 1, "White", played_move, score_loss))
            elif score_loss > 50:
                marker = "?! Inaccuracy"

            print(f"\n{move_num // 2 + 1}. White: {played_move} {marker}")
            if marker:
                print(f"   Engine suggests: {engine.board.san(best_move)}")
                print(f"   Evaluation loss: {score_loss:.0f} cp")

        # Black's move
        if move_num + 1 < len(game_moves):
            engine.new_game()
            engine.set_position_from_moves(game_moves[:move_num + 1])

            best_move, best_score, _ = engine.get_best_move(depth=depth)
            played_move = game_moves[move_num + 1]

            engine.make_move(played_move)
            _, score_after, _ = engine.get_best_move(depth=depth)
            score_loss = abs(best_score + score_after)

            marker = ""
            if score_loss > 200:
                marker = "?? BLUNDER"
                mistakes.append((move_num // 2 + 1, "Black", played_move, score_loss))
            elif score_loss > 100:
                marker = "? Mistake"
                mistakes.append((move_num // 2 + 1, "Black", played_move, score_loss))
            elif score_loss > 50:
                marker = "?! Inaccuracy"

            print(f"   Black: {played_move} {marker}")
            if marker:
                engine.new_game()
                engine.set_position_from_moves(game_moves[:move_num + 1])
                best_move, _, _ = engine.get_best_move(depth=depth)
                print(f"   Engine suggests: {engine.board.san(best_move)}")
                print(f"   Evaluation loss: {score_loss:.0f} cp")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if mistakes:
        print(f"\nFound {len(mistakes)} significant mistakes:\n")
        for move_num, side, move, loss in mistakes:
            print(f"  Move {move_num} ({side}): {move} - Lost {loss:.0f} cp")
    else:
        print("\nNo significant mistakes found! Great game!")

    # Final position analysis
    engine.new_game()
    engine.set_position_from_moves(game_moves)

    print("\nFinal Position:")
    engine.print_board()

    analysis = engine.analyze_position(depth=depth + 2)
    print(f"\nFinal evaluation: {analysis[0]['score_cp']:+d} cp")
    print(f"Best continuation: {' '.join(analysis[0]['pv'][:5])}")


def interactive_analysis():
    """Interactive position analysis."""

    print("=" * 70)
    print("INTERACTIVE ANALYSIS")
    print("=" * 70)
    print("Enter moves in SAN notation (e.g., e4, Nf3, O-O)")
    print("Type 'analyze' to get engine analysis")
    print("Type 'board' to see the board")
    print("Type 'back' to undo last move")
    print("Type 'quit' to exit")
    print("=" * 70)

    engine = ChessEngine()
    depth = 6

    while True:
        try:
            cmd = input("\n> ").strip()

            if cmd == 'quit':
                break

            elif cmd == 'board':
                engine.print_board()

            elif cmd == 'analyze':
                print("Analyzing...")
                analysis = engine.analyze_position(depth=depth)
                print(f"\nBest move: {analysis[0]['move']}")
                print(f"Evaluation: {analysis[0]['score_cp']:+d} cp")
                print(f"Principal variation: {' '.join(analysis[0]['pv'][:8])}")
                print(f"Depth: {analysis[0]['depth']}, Nodes: {analysis[0]['nodes']:,}")

            elif cmd == 'back':
                if len(engine.board.move_stack) > 0:
                    engine.board.pop()
                    print("Move undone")
                else:
                    print("No moves to undo")

            elif cmd == 'reset':
                engine.new_game()
                print("Board reset")

            elif cmd == 'help':
                print("Commands: analyze, board, back, reset, quit")
                print("Or enter a move in SAN notation")

            else:
                # Try to make the move
                try:
                    move = engine.make_move(cmd)
                    print(f"Played: {engine.board.san(move)}")

                    # Quick evaluation
                    score = engine.evaluate_position()
                    print(f"Static eval: {score:+d} cp")

                except ValueError as e:
                    print(f"Invalid move: {e}")
                    print("Legal moves:", ', '.join(engine.get_legal_moves()[:15]))

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def analyze_critical_position():
    """Analyze a specific critical position in depth."""

    print("=" * 70)
    print("DEEP POSITION ANALYSIS")
    print("=" * 70)

    # Example: Tricky middlegame position
    engine = ChessEngine()
    engine.set_position('r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9')

    print("\nPosition:")
    engine.print_board()

    print("\nRunning multi-depth analysis...")

    for depth in range(4, 9):
        move, score, info = engine.get_best_move(depth=depth)
        print(f"\nDepth {depth}:")
        print(f"  Best: {engine.board.san(move)} ({score:+d} cp)")
        print(f"  PV: {' '.join(info['pv'][:6])}")
        print(f"  Nodes: {info['nodes']:,}, Time: {info['time_ms']:.0f}ms")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_analysis()
    elif len(sys.argv) > 1 and sys.argv[1] == 'deep':
        analyze_critical_position()
    else:
        analyze_my_game()

    print("\nDone!")
