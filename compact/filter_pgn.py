#!/usr/bin/env python3
"""
Filter PGN files to only include high-quality games

Usage:
    python3 filter_pgn.py input.pgn [output.pgn] [--min-elo 2200] [--max-games 50000]

This filters for:
- High ELO players (default: 2200+)
- Classical time control (no bullet/blitz)
- Decisive games or draws (no aborted)
- Reasonable game length (20+ moves)
"""

import sys
import argparse
import chess.pgn
from tqdm import tqdm


def filter_pgn(input_file, output_file, min_elo=2200, max_games=None, time_control="classical"):
    """
    Filter PGN file for high-quality games

    Args:
        input_file: Input PGN file path
        output_file: Output PGN file path
        min_elo: Minimum ELO for both players
        max_games: Maximum games to include (None = all)
        time_control: "classical", "rapid", or "all"
    """

    games_read = 0
    games_written = 0

    print(f"Filtering: {input_file}")
    print(f"Min ELO: {min_elo}")
    print(f"Max games: {max_games if max_games else 'unlimited'}")
    print(f"Time control: {time_control}")
    print()

    with open(input_file) as f_in, open(output_file, 'w') as f_out:

        with tqdm(desc="Processing games", unit=" games") as pbar:
            while True:
                game = chess.pgn.read_game(f_in)
                if game is None:
                    break

                games_read += 1
                pbar.update(1)

                # Check ELO
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))

                    if white_elo < min_elo or black_elo < min_elo:
                        continue
                except (ValueError, TypeError):
                    continue

                # Check time control
                if time_control != "all":
                    tc = game.headers.get("TimeControl", "")

                    if time_control == "classical":
                        # Classical: 1800+ seconds or unlimited
                        if tc and tc != "-":
                            try:
                                # Format: "initial+increment" or "initial"
                                initial = int(tc.split('+')[0])
                                if initial < 1800:  # Less than 30 minutes
                                    continue
                            except:
                                pass
                    elif time_control == "rapid":
                        # Rapid: 600-1800 seconds
                        if tc and tc != "-":
                            try:
                                initial = int(tc.split('+')[0])
                                if initial < 600 or initial >= 1800:
                                    continue
                            except:
                                pass

                # Check game result
                result = game.headers.get("Result", "*")
                if result not in ["1-0", "0-1", "1/2-1/2"]:
                    continue  # Skip unfinished/aborted games

                # Check game length
                moves = list(game.mainline_moves())
                if len(moves) < 20:  # Too short
                    continue

                # Check termination (avoid timeouts in rapid games)
                termination = game.headers.get("Termination", "").lower()
                if "time" in termination and time_control == "classical":
                    continue  # Skip timeout games in classical

                # Write the game
                print(game, file=f_out, end="\n\n")
                games_written += 1
                pbar.set_postfix(written=games_written, ratio=f"{games_written/games_read*100:.1f}%")

                if max_games and games_written >= max_games:
                    break

    print()
    print(f"✓ Filtering complete!")
    print(f"  Games read: {games_read:,}")
    print(f"  Games written: {games_written:,}")
    print(f"  Filter ratio: {games_written/games_read*100:.1f}%")
    print(f"  Output file: {output_file}")

    return games_written


def main():
    parser = argparse.ArgumentParser(description="Filter PGN files for high-quality games")
    parser.add_argument("input", help="Input PGN file")
    parser.add_argument("output", nargs='?', help="Output PGN file (default: input_filtered.pgn)")
    parser.add_argument("--min-elo", type=int, default=2200, help="Minimum ELO (default: 2200)")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum games to include")
    parser.add_argument("--time-control", choices=["classical", "rapid", "all"],
                        default="classical", help="Time control filter")

    args = parser.parse_args()

    # Default output filename
    if args.output is None:
        if args.input.endswith('.pgn'):
            args.output = args.input.replace('.pgn', '_filtered.pgn')
        else:
            args.output = args.input + '_filtered.pgn'

    # Filter
    try:
        filter_pgn(args.input, args.output,
                   min_elo=args.min_elo,
                   max_games=args.max_games,
                   time_control=args.time_control)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
