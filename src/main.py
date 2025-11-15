"""
NNUE Chess Bot for Chess-Hacks
Level 4 - Uses NNUE (Efficiently Updatable Neural Network) architecture
"""

import os
import torch
import torch.nn as nn
import chess
import chess.pgn
import chess.polyglot
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict
from collections import defaultdict


# ============================================================================
# NNUE MODEL ARCHITECTURE
# ============================================================================

# Feature set sizes
NUM_SQUARES = 64
NUM_PIECE_TYPES = 6  # P, N, B, R, Q, K
NUM_COLORS = 2

# HalfKP: For each king position, track all pieces (except kings)
HALFKP_INPUT_SIZE = 10 * 64  # 640 features per side


def halfkp_index(king_square, piece_square, piece_type, piece_color):
    """
    Compute HalfKP feature index

    Args:
        king_square: Square of the king (0-63)
        piece_square: Square of the piece (0-63)
        piece_type: Type of piece (1-6 for P,N,B,R,Q,K)
        piece_color: Color of piece (True=White, False=Black)

    Returns:
        Feature index (0-639) or None if piece is a king
    """
    # Don't include kings in features
    if piece_type == chess.KING:
        return None

    # Adjust piece type (remove king, so Q=4 -> 4)
    piece_idx = piece_type - 1  # 0-5 for P,N,B,R,Q,K

    # Color offset
    color_offset = 0 if piece_color == chess.WHITE else 5

    # Feature index: piece_type * 64 + piece_square
    feature_idx = (piece_idx + color_offset) * 64 + piece_square

    return feature_idx


def board_to_halfkp_features(board: chess.Board):
    """
    Convert board to HalfKP feature representation

    Returns:
        white_features: Features from white's perspective (640,)
        black_features: Features from black's perspective (640,)
    """
    white_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)
    black_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)

    # Find kings
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is None or black_king_square is None:
        # Invalid position (shouldn't happen in real games)
        return white_features, black_features

    # Populate features
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue  # Don't include kings in features

        # White's perspective (relative to white king)
        idx = halfkp_index(white_king_square, square, piece.piece_type, piece.color)
        if idx is not None:
            white_features[idx] = 1.0

        # Black's perspective (relative to black king, board flipped)
        flipped_square = square ^ 56  # Flip rank
        flipped_king_square = black_king_square ^ 56
        idx = halfkp_index(flipped_king_square, flipped_square, piece.piece_type, not piece.color)
        if idx is not None:
            black_features[idx] = 1.0

    return white_features, black_features


class NNUE(nn.Module):
    """
    NNUE Architecture optimized for chess evaluation
    Based on Stockfish NNUE architecture
    """

    def __init__(self,
                 input_size=HALFKP_INPUT_SIZE,
                 hidden1_size=1408,  # Optimized for ~10MB
                 hidden2_size=32):
        super(NNUE, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # Feature transformer (one for each side)
        self.ft_white = nn.Linear(input_size, hidden1_size)
        self.ft_black = nn.Linear(input_size, hidden1_size)

        # Layer 2: Combine both perspectives
        self.fc1 = nn.Linear(hidden1_size * 2, hidden2_size)

        # Output layer
        self.fc2 = nn.Linear(hidden2_size, 1)

        # ClippedReLU activation (standard in NNUE)
        self.clipped_relu = lambda x: torch.clamp(torch.relu(x), 0, 1)

    def forward(self, white_features, black_features):
        """
        Forward pass

        Args:
            white_features: HalfKP features from white's perspective (batch_size, 640)
            black_features: HalfKP features from black's perspective (batch_size, 640)

        Returns:
            Evaluation score (batch_size, 1)
        """
        # Feature transformation for both sides
        white_hidden = self.clipped_relu(self.ft_white(white_features))
        black_hidden = self.clipped_relu(self.ft_black(black_features))

        # Concatenate both perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=1)

        # Hidden layers
        x = self.clipped_relu(self.fc1(combined))
        x = self.fc2(x)

        return x


# ============================================================================
# CHESS ENGINE WITH NNUE
# ============================================================================

# Piece values for move ordering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

MATE_SCORE = 30000
MAX_PLY = 100


class TranspositionTable:
    """Transposition table for storing previously evaluated positions"""

    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

    def __init__(self, size_mb=2):
        self.max_entries = int((size_mb * 1024 * 1024) / 16)
        self.table = {}

    def probe(self, zobrist_hash, depth, alpha, beta):
        if zobrist_hash not in self.table:
            return None, None

        entry = self.table[zobrist_hash]
        stored_depth, stored_score, stored_flag, stored_move = entry

        if stored_depth >= depth:
            if stored_flag == self.EXACT:
                return stored_score, stored_move
            elif stored_flag == self.LOWER_BOUND and stored_score >= beta:
                return stored_score, stored_move
            elif stored_flag == self.UPPER_BOUND and stored_score <= alpha:
                return stored_score, stored_move

        return None, stored_move

    def store(self, zobrist_hash, depth, score, flag, best_move):
        if zobrist_hash in self.table:
            old_depth = self.table[zobrist_hash][0]
            if depth < old_depth:
                return

        self.table[zobrist_hash] = (depth, score, flag, best_move)

        if len(self.table) > self.max_entries:
            keys = list(self.table.keys())
            for key in keys[:len(keys) // 4]:
                del self.table[key]


class NNUEEngine:
    """Chess engine with NNUE evaluation - Level 4"""

    def __init__(self, model_path: str, device: str = "cpu", tt_size_mb=2):
        self.device = torch.device(device)
        self.model = NNUE(hidden1_size=1408, hidden2_size=32).to(self.device)

        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Search data structures
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.killer_moves = [[None, None] for _ in range(MAX_PLY)]
        self.history = defaultdict(int)

        # Statistics
        self.nodes_searched = 0
        self.tt_hits = 0

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using NNUE"""
        white_feat, black_feat = board_to_halfkp_features(board)

        white_tensor = torch.tensor(white_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        black_tensor = torch.tensor(black_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model(white_tensor, black_tensor).item()

        # Convert to centipawns
        score = score * 200

        return score

    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Most Valuable Victim - Least Valuable Attacker score"""
        if not board.is_capture(move):
            return 0

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        return PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]

    def order_moves(self, board: chess.Board, ply: int, tt_move: Optional[chess.Move] = None):
        """Order moves for better alpha-beta pruning"""
        moves = list(board.legal_moves)
        move_scores = []

        for move in moves:
            score = 0

            if tt_move and move == tt_move:
                score = 1_000_000
            elif board.is_capture(move):
                mvv_lva = self.mvv_lva_score(board, move)
                score = 100_000 + mvv_lva
            elif move in self.killer_moves[ply]:
                score = 10_000
            else:
                score = self.history[(move.from_square, move.to_square)]

            board.push(move)
            if board.is_check():
                score += 5_000
            board.pop()

            if move.promotion:
                score += 50_000

            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Quiescence search - search all captures to avoid horizon effect"""
        stand_pat = self.evaluate_position(board)

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in board.legal_moves:
            if not board.is_capture(move):
                continue

            board.push(move)

            if board.is_valid():
                score = -self.quiescence_search(board, -beta, -alpha)

                if score >= beta:
                    board.pop()
                    return beta

                if score > alpha:
                    alpha = score

            board.pop()

        return alpha

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        """Alpha-beta search with pruning"""
        self.nodes_searched += 1

        # Check for draw
        if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0

        # Probe transposition table
        zobrist = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self.tt.probe(zobrist, depth, alpha, beta)
        if tt_score is not None:
            self.tt_hits += 1
            return tt_score

        # Quiescence search at leaf nodes
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)

        # Check if game is over
        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE + ply
            return 0

        # Get ordered moves
        moves = self.order_moves(board, ply, tt_move)

        if not moves:
            if board.is_check():
                return -MATE_SCORE + ply
            return 0

        best_score = -float('inf')
        best_move = None
        flag = TranspositionTable.UPPER_BOUND

        for i, move in enumerate(moves):
            board.push(move)

            if i == 0:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self.alpha_beta(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                flag = TranspositionTable.EXACT

            if score >= beta:
                flag = TranspositionTable.LOWER_BOUND

                if not board.is_capture(move):
                    if self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move

                    self.history[(move.from_square, move.to_square)] += depth * depth

                break

        self.tt.store(zobrist, depth, best_score, flag, best_move)

        return best_score

    def get_best_move(self, board: chess.Board, depth: int = 4) -> chess.Move:
        """
        Get best move for current position

        Args:
            board: Current position
            depth: Search depth (default 4 for Level 4)

        Returns:
            Best move
        """
        self.nodes_searched = 0
        self.tt_hits = 0

        best_move = None
        best_score = -float('inf')
        alpha = -MATE_SCORE
        beta = MATE_SCORE

        moves = self.order_moves(board, 0, None)

        for move in moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        return best_move


# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

_engine = None


def _initialize_engine():
    """Initialize the NNUE engine (called once)"""
    global _engine

    if _engine is not None:
        return _engine

    # Find model file
    model_files = ["nnue_best.pth", "nnue_model.pth", "best_model.pth", "model.pth"]
    model_path = None

    # Search in multiple locations
    search_paths = [
        os.path.dirname(__file__),  # src directory
        os.path.join(os.path.dirname(__file__), "..", "compact"),  # compact directory
        ".",
        "../compact",
        "compact"
    ]

    for search_path in search_paths:
        for model_file in model_files:
            full_path = os.path.join(search_path, model_file)
            if os.path.exists(full_path):
                model_path = full_path
                break
        if model_path:
            break

    if model_path is None:
        raise FileNotFoundError(
            "No NNUE model found! Please ensure a trained model (.pth file) is available.\n"
            f"Looking for: {', '.join(model_files)}\n"
            f"In directories: {', '.join(search_paths)}"
        )

    _engine = NNUEEngine(model_path, device="cpu", tt_size_mb=2)
    return _engine


# ============================================================================
# MAIN API FUNCTION FOR CHESS-HACKS
# ============================================================================

def get_move(pgn: str) -> str:
    """
    Get the best move for the given position

    Args:
        pgn: Board state as PGN string

    Returns:
        Best move in UCI format (e.g., "e2e4")
    """
    # Parse PGN to get the board position
    game = chess.pgn.read_game(StringIO(pgn))

    if game is None:
        # If parsing fails, assume it's the starting position
        board = chess.Board()
    else:
        board = game.end().board()

    # Initialize engine if needed
    engine = _initialize_engine()

    # Get best move
    best_move = engine.get_best_move(board, depth=4)

    if best_move is None:
        # Fallback to any legal move if search fails
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move = legal_moves[0]
        else:
            raise ValueError("No legal moves available")

    # Return move in UCI format
    return best_move.uci()


# ============================================================================
# TESTING / STANDALONE MODE
# ============================================================================

if __name__ == "__main__":
    # Test the bot
    print("=" * 60)
    print("NNUE CHESS BOT - LEVEL 4")
    print("=" * 60)
    print()

    # Test with starting position
    test_pgn = ""

    try:
        move = get_move(test_pgn)
        print(f"Best move from starting position: {move}")
        print()
        print("✓ Bot is working correctly!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: You need a trained NNUE model file to use this bot.")
        print("Expected model files: nnue_best.pth, nnue_model.pth, best_model.pth, or model.pth")
