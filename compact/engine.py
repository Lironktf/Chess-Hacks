"""
Compact Chess Engine - Combines neural network evaluation with alpha-beta search

This engine uses:
1. Neural network for position evaluation
2. Alpha-beta pruning for efficient move search
3. Move ordering for better pruning
"""

import os
import torch
import chess
import numpy as np
from typing import Optional, Tuple

from model import CompactChessNet
from utils import board_to_tensor


class ChessEngine:
    """
    Chess engine that combines neural network evaluation with alpha-beta search
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the chess engine

        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.model = CompactChessNet().to(self.device)

        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Statistics
        self.nodes_searched = 0

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a chess position using the neural network

        Args:
            board: Chess board to evaluate

        Returns:
            Evaluation score from the current player's perspective
            Positive = current player winning, Negative = current player losing
        """
        # Convert board to tensor
        board_tensor = board_to_tensor(board)
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
        board_tensor = board_tensor.to(self.device)

        # Get evaluation from network
        with torch.no_grad():
            evaluation = self.model(board_tensor).item()

        # The network outputs evaluation from white's perspective
        # Flip sign if it's black's turn
        if board.turn == chess.BLACK:
            evaluation = -evaluation

        return evaluation

    def order_moves(self, board: chess.Board):
        """
        Order moves for better alpha-beta pruning

        Move ordering heuristics:
        1. Captures (ordered by MVV-LVA)
        2. Checks
        3. Other moves

        Args:
            board: Current chess position

        Returns:
            Ordered list of moves
        """
        moves = list(board.legal_moves)

        def move_score(move):
            score = 0

            # Prioritize captures (Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # Piece values: P=1, N=3, B=3, R=5, Q=9, K=100
                    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}
                    score += 10 * piece_values[victim.piece_type]
                    score -= piece_values[attacker.piece_type]

            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()

            # Prioritize center moves in opening
            if board.fullmove_number < 10:
                to_square = move.to_square
                rank, file = divmod(to_square, 8)
                center_distance = abs(rank - 3.5) + abs(file - 3.5)
                score -= center_distance

            return score

        # Sort moves by score (descending)
        moves.sort(key=move_score, reverse=True)
        return moves

    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> float:
        """
        Alpha-beta pruning search

        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: Whether this is a maximizing node

        Returns:
            Best evaluation found
        """
        self.nodes_searched += 1

        # Base cases
        if depth == 0:
            return self.evaluate_position(board)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1000 if board.turn == chess.WHITE else -1000
            elif result == "0-1":
                return -1000 if board.turn == chess.WHITE else 1000
            else:
                return 0  # Draw

        # Get ordered moves for better pruning
        moves = self.order_moves(board)

        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                board.push(move)
                eval_score = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval

    def search(self, board: chess.Board, depth: int = 3) -> Tuple[chess.Move, float]:
        """
        Search for the best move using alpha-beta pruning

        Args:
            board: Current chess position
            depth: Search depth (3-5 recommended)

        Returns:
            Tuple of (best_move, evaluation)
        """
        self.nodes_searched = 0

        best_move = None
        best_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        # Search all legal moves
        moves = self.order_moves(board)

        for move in moves:
            board.push(move)
            eval_score = self.alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)

        return best_move, best_eval

    def get_move(self, board: chess.Board, search_depth: int = 3) -> chess.Move:
        """
        Get the best move for the current position

        Args:
            board: Current chess position
            search_depth: How deep to search (default 3)

        Returns:
            Best move to play
        """
        move, eval_score = self.search(board, depth=search_depth)
        print(f"Nodes searched: {self.nodes_searched}")
        print(f"Evaluation: {eval_score:.3f}")
        return move


def play_game(engine: ChessEngine, search_depth: int = 3):
    """
    Play a game against the engine

    Args:
        engine: Chess engine to play against
        search_depth: Search depth for the engine
    """
    board = chess.Board()

    print("=" * 60)
    print("COMPACT CHESS BOT")
    print("=" * 60)
    print("You are White. Enter moves in UCI format (e.g., 'e2e4')")
    print("Type 'quit' to exit\n")

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == chess.WHITE:
            # Human's turn
            while True:
                move_str = input("Your move: ").strip()

                if move_str.lower() == 'quit':
                    print("Thanks for playing!")
                    return

                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid format! Use UCI notation (e.g., 'e2e4')")
        else:
            # Engine's turn
            print("Engine thinking...")
            move = engine.get_move(board, search_depth=search_depth)
            print(f"Engine plays: {move}\n")
            board.push(move)

    # Game over
    print(board)
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


if __name__ == "__main__":
    import sys

    # Load the model
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        model_path = "final_model.pth"
        if not os.path.exists(model_path):
            print("Error: No trained model found!")
            print("Please run 'python train.py' first to train a model.")
            sys.exit(1)

    print(f"Loading model from: {model_path}")
    engine = ChessEngine(model_path)

    # Play a game
    play_game(engine, search_depth=3)
