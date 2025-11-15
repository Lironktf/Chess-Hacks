"""
Utility functions for board representation and game processing
"""

import numpy as np
import chess


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a 13x8x8 tensor representation

    Channels:
    - 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - 12: Legal move destinations

    Args:
        board: python-chess Board object

    Returns:
        numpy array of shape (13, 8, 8)
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    # Encode piece positions (channels 0-11)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)

        # Piece type: 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
        piece_type = piece.piece_type - 1  # Convert to 0-indexed

        # Color offset: white=0, black=6
        color_offset = 0 if piece.color == chess.WHITE else 6

        tensor[piece_type + color_offset, row, col] = 1

    # Encode legal move destinations (channel 12)
    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        tensor[12, row, col] = 1

    return tensor


def get_game_result(game) -> float:
    """
    Extract the game result from a PGN game

    Returns:
        1.0 if White won
        0.0 if Draw
        -1.0 if Black won
        None if result is unknown
    """
    result = game.headers.get("Result", "*")

    if result == "1-0":
        return 1.0  # White wins
    elif result == "0-1":
        return -1.0  # Black wins
    elif result == "1/2-1/2":
        return 0.0  # Draw
    else:
        return None  # Unknown result


def create_training_data(games, max_positions_per_game=20):
    """
    Create training data from PGN games for value network

    For each game, we sample positions and label them with the game outcome.
    This teaches the network to evaluate positions.

    Args:
        games: List of python-chess Game objects
        max_positions_per_game: Maximum positions to sample per game

    Returns:
        X: numpy array of board states (N, 13, 8, 8)
        y: numpy array of game results (N,) in range [-1, 1]
    """
    X = []
    y = []

    for game in games:
        result = get_game_result(game)
        if result is None:
            continue  # Skip games without clear result

        board = game.board()
        moves = list(game.mainline_moves())

        # Skip very short games (likely invalid)
        if len(moves) < 10:
            continue

        # Sample positions evenly throughout the game
        if len(moves) <= max_positions_per_game:
            sample_indices = range(len(moves))
        else:
            # Sample evenly throughout the game
            step = len(moves) / max_positions_per_game
            sample_indices = [int(i * step) for i in range(max_positions_per_game)]

        # Extract positions
        board = game.board()
        for i, move in enumerate(moves):
            if i in sample_indices:
                # Encode the position
                X.append(board_to_tensor(board))

                # The label is the game result from current player's perspective
                # If it's white's turn, use result as-is
                # If it's black's turn, flip the sign
                if board.turn == chess.WHITE:
                    y.append(result)
                else:
                    y.append(-result)

            board.push(move)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_pgn_games(file_path, max_games=None):
    """
    Load games from a PGN file

    Args:
        file_path: Path to PGN file
        max_games: Maximum number of games to load (None for all)

    Returns:
        List of python-chess Game objects
    """
    games = []

    with open(file_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)

            if max_games is not None and len(games) >= max_games:
                break

    return games


if __name__ == "__main__":
    # Test board encoding
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"Board tensor shape: {tensor.shape}")
    print(f"Number of pieces on starting position: {tensor[:12].sum()}")
    print(f"Number of legal moves: {tensor[12].sum()}")

    # Test game result parsing
    from chess import pgn
    from io import StringIO

    sample_pgn = '''[Result "1-0"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
    '''

    game = pgn.read_game(StringIO(sample_pgn))
    result = get_game_result(game)
    print(f"\nGame result: {result}")

    # Test training data creation
    X, y = create_training_data([game], max_positions_per_game=5)
    print(f"\nTraining data shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Labels: {y}")
