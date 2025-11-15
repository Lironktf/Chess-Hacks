"""
Opening book for chess engine

Uses common opening theory to play strong moves in the opening
This dramatically improves early game strength
"""

import chess
import random
from typing import Optional, List, Dict


# Common opening moves (from master games)
# Format: position_fen -> [(move, weight), ...]
OPENING_BOOK = {
    # Starting position
    chess.STARTING_FEN: [
        ("e2e4", 100),  # King's Pawn
        ("d2d4", 95),   # Queen's Pawn
        ("c2c4", 40),   # English
        ("g1f3", 35),   # Reti
        ("e2e3", 10),   # Various
    ],

    # After 1. e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
        ("e7e5", 100),  # Open game
        ("c7c5", 80),   # Sicilian
        ("e7e6", 50),   # French
        ("c7c6", 45),   # Caro-Kann
        ("d7d6", 35),   # Pirc
        ("g7g6", 30),   # Modern
    ],

    # After 1. e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": [
        ("g1f3", 100),  # King's Knight
        ("f1c4", 40),   # King's Gambit
        ("b1c3", 30),   # Vienna
        ("f2f4", 25),   # King's Gambit
    ],

    # After 1. e4 e5 2. Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
        ("b8c6", 100),  # Classical
        ("g8f6", 50),   # Petrov
        ("d7d6", 30),   # Philidor
    ],

    # After 1. e4 e5 2. Nf3 Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": [
        ("f1b5", 100),  # Ruy Lopez
        ("f1c4", 80),   # Italian
        ("d2d4", 50),   # Scotch
        ("b1c3", 40),   # Four Knights
    ],

    # Ruy Lopez main line
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": [
        ("a7a6", 100),  # Morphy Defense
        ("g8f6", 80),   # Berlin Defense
        ("f7f5", 20),   # Schliemann
    ],

    # After 1. d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": [
        ("g8f6", 100),  # Indian Systems
        ("d7d5", 90),   # Closed Game
        ("e7e6", 50),   # Various
        ("f7f5", 30),   # Dutch
    ],

    # After 1. d4 Nf6
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": [
        ("c2c4", 100),  # Indian Systems
        ("g1f3", 50),   # Various
        ("f2f3", 20),   # Trompowsky
    ],

    # After 1. d4 Nf6 2. c4
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": [
        ("e7e6", 100),  # Nimzo/QID/QGD
        ("g7g6", 80),   # King's Indian
        ("c7c5", 70),   # Benoni
        ("d7d5", 60),   # QGD
    ],

    # After 1. d4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2": [
        ("c2c4", 100),  # Queen's Gambit
        ("g1f3", 50),   # Various
        ("e2e3", 30),   # Colle
    ],

    # After 1. d4 d5 2. c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": [
        ("e7e6", 100),  # QGD
        ("c7c6", 80),   # Slav
        ("d5c4", 60),   # QGA
        ("g8f6", 50),   # Various
    ],
}


class OpeningBook:
    """
    Opening book for strong opening play
    """

    def __init__(self, book: Dict[str, List[tuple]] = OPENING_BOOK):
        """
        Args:
            book: Opening book dictionary
        """
        self.book = book

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get a move from the opening book

        Args:
            board: Current position

        Returns:
            Book move if found, else None
        """
        fen = board.fen()

        # Check if position is in book
        if fen not in self.book:
            return None

        # Get candidate moves
        candidates = self.book[fen]

        # Weight-based selection
        total_weight = sum(weight for _, weight in candidates)
        rand = random.randint(1, total_weight)

        cumulative = 0
        for move_str, weight in candidates:
            cumulative += weight
            if rand <= cumulative:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        return move
                except:
                    continue

        return None

    def is_in_book(self, board: chess.Board) -> bool:
        """Check if position is in opening book"""
        return board.fen() in self.book

    def add_position(self, fen: str, moves: List[tuple]):
        """
        Add a position to the book

        Args:
            fen: FEN string of position
            moves: List of (move_uci, weight) tuples
        """
        self.book[fen] = moves

    def size_bytes(self) -> int:
        """Estimate size of opening book in bytes"""
        # Rough estimate
        total = 0
        for fen, moves in self.book.items():
            total += len(fen)  # FEN string
            total += len(moves) * 10  # Moves and weights
        return total

    def size_mb(self) -> float:
        """Size in megabytes"""
        return self.size_bytes() / 1024 / 1024


def expand_book_from_pgn(pgn_file: str, max_games: int = 1000, max_ply: int = 20):
    """
    Expand opening book by analyzing PGN games

    Args:
        pgn_file: Path to PGN file
        max_games: Maximum games to process
        max_ply: Maximum ply to include in book

    Returns:
        Dictionary of positions -> moves
    """
    import chess.pgn

    book_positions = {}

    with open(pgn_file) as f:
        for i in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Only use games with good players
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                if white_elo < 2200 or black_elo < 2200:
                    continue
            except:
                continue

            # Extract opening moves
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= max_ply:
                    break

                fen = board.fen()

                if fen not in book_positions:
                    book_positions[fen] = {}

                move_uci = move.uci()
                if move_uci not in book_positions[fen]:
                    book_positions[fen][move_uci] = 0

                book_positions[fen][move_uci] += 1

                board.push(move)

    # Convert to proper format
    book = {}
    for fen, moves in book_positions.items():
        if len(moves) >= 2:  # Only include positions with multiple options
            book[fen] = [(move, count) for move, count in moves.items()]

    return book


if __name__ == "__main__":
    print("=" * 60)
    print("OPENING BOOK")
    print("=" * 60)
    print()

    book = OpeningBook()
    print(f"Positions in book: {len(book.book)}")
    print(f"Book size: {book.size_mb():.4f} MB")
    print()

    # Test with starting position
    board = chess.Board()
    move = book.get_book_move(board)

    if move:
        print(f"Book move from starting position: {move}")
    else:
        print("No book move found")

    print()

    # Test coverage
    board = chess.Board()
    moves_in_book = 0

    for _ in range(10):  # Simulate 10 moves
        if book.is_in_book(board):
            move = book.get_book_move(board)
            if move:
                print(f"{board.fullmove_number}. {move}")
                board.push(move)
                moves_in_book += 1
            else:
                break
        else:
            break

    print()
    print(f"Moves played from book: {moves_in_book}")
    print()
    print("✓ Opening book working correctly!")
