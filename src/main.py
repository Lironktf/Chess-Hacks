"""
Compact Chess Engine - Level 4 (Very Hard)
Chess bot with neural network evaluation and alpha-beta search at depth 5
"""

import os
import sys
import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class CompactChessNet(nn.Module):
    """
    Ultra-compact neural network for chess position evaluation.

    Architecture:
    - Input: 13x8x8 (12 piece planes + 1 legal moves plane)
    - Conv1: 13 → 32 channels (3x3 kernel)
    - Conv2: 32 → 64 channels (3x3 kernel)
    - FC1: 64*8*8 → 128
    - FC2: 128 → 1 (value head)

    Total params: ~680K → ~2.7MB (float32) → ~680KB (int8 quantized)
    """

    def __init__(self):
        super(CompactChessNet, self).__init__()

        # Convolutional layers - extract spatial patterns
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)  # Keep 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Keep 8x8

        # Batch normalization for better training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers - combine features into evaluation
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)  # Single output: position evaluation

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Output in [-1, 1] range

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Board state tensor (batch_size, 13, 8, 8)
        Returns:
            Position evaluation in [-1, 1] range
            -1 = Black winning, 0 = Draw, +1 = White winning
        """
        # Convolutional layers with batch norm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # Squash to [-1, 1]

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# CHESS ENGINE - LEVEL 4 (VERY HARD)
# ============================================================================

class ChessEngine:
    """
    Chess engine that combines neural network evaluation with alpha-beta search
    Configured for Level 4: Very Hard (depth=5)
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

    def search(self, board: chess.Board, depth: int = 5) -> Tuple[chess.Move, float]:
        """
        Search for the best move using alpha-beta pruning
        Level 4 default: depth=5

        Args:
            board: Current chess position
            depth: Search depth (default 5 for Level 4)

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

    def get_move(self, board: chess.Board, search_depth: int = 5) -> chess.Move:
        """
        Get the best move for the current position
        Level 4: search_depth=5 (Very Hard)

        Args:
            board: Current chess position
            search_depth: How deep to search (default 5 for Level 4)

        Returns:
            Best move to play
        """
        move, eval_score = self.search(board, depth=search_depth)
        print(f"Nodes searched: {self.nodes_searched}")
        print(f"Evaluation: {eval_score:.3f}")
        return move


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to initialize and run the Level 4 chess bot
    """
    print("=" * 60)
    print("COMPACT CHESS BOT - LEVEL 4 (VERY HARD)")
    print("Search Depth: 5")
    print("=" * 60)
    print()

    # Find the best available model
    model_files = ["model_int8.pth", "best_model.pth", "final_model.pth"]
    model_path = None

    # Check in current directory and compact directory
    search_paths = [".", "../compact", "compact"]

    for search_path in search_paths:
        for model_file in model_files:
            full_path = os.path.join(search_path, model_file)
            if os.path.exists(full_path):
                model_path = full_path
                break
        if model_path:
            break

    if model_path is None:
        print("❌ No trained model found!")
        print("\nPlease train a model first or ensure model files are in:")
        print("  - Current directory")
        print("  - compact/ directory")
        print()
        print("Looking for: model_int8.pth, best_model.pth, or final_model.pth")
        sys.exit(1)

    # Get file size
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"Loading model: {model_path} ({model_size:.2f} MB)")

    # Load engine
    try:
        engine = ChessEngine(model_path)
        print("✓ Model loaded successfully!")
        print(f"✓ Level 4 configuration active (search depth: 5)")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    return engine


if __name__ == "__main__":
    # Initialize the Level 4 chess engine
    engine = main()

    # Engine is ready to use
    print("Chess engine initialized and ready!")
    print("Use engine.get_move(board) to get the best move for a position")
    print()

    # Example usage
    print("Example usage:")
    print("  board = chess.Board()")
    print("  move = engine.get_move(board, search_depth=5)")
    print("  board.push(move)")
