from .utils import chess_manager, GameContext
from chess import Move
import os
import torch
import chess
import chess.polyglot
import numpy as np
from typing import Optional, Tuple, Dict
from collections import defaultdict
import torch.nn as nn

# NNUE Model Architecture
class NNUE(nn.Module):
    """NNUE Architecture optimized for chess evaluation"""

    def __init__(self, input_size=640, hidden1_size=1408, hidden2_size=32):
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

        # Use ClippedReLU activation (standard in NNUE)
        self.clipped_relu = lambda x: torch.clamp(torch.relu(x), 0, 1)

    def forward(self, white_features, black_features):
        """Forward pass"""
        # Feature transformation for both sides
        white_hidden = self.clipped_relu(self.ft_white(white_features))
        black_hidden = self.clipped_relu(self.ft_black(black_features))

        # Concatenate both perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=1)

        # Hidden layers
        x = self.clipped_relu(self.fc1(combined))
        x = self.fc2(x)

        return x


def halfkp_index(king_square, piece_square, piece_type, piece_color):
    """Compute HalfKP feature index"""
    if piece_type == chess.KING:
        return None

    piece_idx = piece_type - 1  # 0-5 for P,N,B,R,Q,K
    color_offset = 0 if piece_color == chess.WHITE else 5
    feature_idx = (piece_idx + color_offset) * 64 + piece_square

    return feature_idx


def board_to_halfkp_features(board: chess.Board):
    """Convert board to HalfKP feature representation"""
    HALFKP_INPUT_SIZE = 640
    white_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)
    black_features = np.zeros(HALFKP_INPUT_SIZE, dtype=np.float32)

    # Find kings
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is None or black_king_square is None:
        return white_features, black_features

    # Populate features
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        # White's perspective
        idx = halfkp_index(white_king_square, square, piece.piece_type, piece.color)
        if idx is not None:
            white_features[idx] = 1.0

        # Black's perspective (board flipped)
        flipped_square = square ^ 56
        flipped_king_square = black_king_square ^ 56
        idx = halfkp_index(flipped_king_square, flipped_square, piece.piece_type, not piece.color)
        if idx is not None:
            black_features[idx] = 1.0

    return white_features, black_features


# Piece values for move ordering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables for positional bonuses (values in centipawns)
# Encourage central control, development, and king safety
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

KING_TABLE_MIDDLEGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

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
    """Advanced chess engine with NNUE evaluation"""

    def __init__(self, model_path: str, device: str = "cpu", tt_size_mb=2):
        self.device = torch.device(device)
        self.model = NNUE(hidden1_size=1408, hidden2_size=32).to(self.device)

        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle different naming conventions
        if "ft_w.weight" in state_dict and "ft_white.weight" not in state_dict:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key == "ft_w.weight":
                    new_state_dict["ft_white.weight"] = value
                elif key == "ft_w.bias":
                    new_state_dict["ft_white.bias"] = value
                elif key == "ft_b.weight":
                    new_state_dict["ft_black.weight"] = value
                elif key == "ft_b.bias":
                    new_state_dict["ft_black.bias"] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Search data structures
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.killer_moves = [[None, None] for _ in range(MAX_PLY)]
        self.history = defaultdict(int)

        self.nodes_searched = 0
        self.tt_hits = 0
        self.qnodes = 0

    def evaluate_position_fast(self, board: chess.Board) -> float:
        """
        Fast evaluation - material + piece-square tables only (no NNUE)
        Used for interior nodes during search
        """
        material_score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]

                # Piece-square table bonus
                sq = square if piece.color == chess.WHITE else chess.square_mirror(square)
                if piece.piece_type == chess.PAWN:
                    value += PAWN_TABLE[sq]
                elif piece.piece_type == chess.KNIGHT:
                    value += KNIGHT_TABLE[sq]
                elif piece.piece_type == chess.BISHOP:
                    value += BISHOP_TABLE[sq]
                elif piece.piece_type == chess.KING:
                    value += KING_TABLE_MIDDLEGAME[sq]

                if piece.color == chess.WHITE:
                    material_score += value
                else:
                    material_score -= value

        # Bonus for checks
        if board.is_check():
            material_score += 50 if board.turn else -50

        return material_score

    def evaluate_position(self, board: chess.Board, use_nnue: bool = True) -> float:
        """
        Hybrid evaluation combining material, piece-square tables, and NNUE

        use_nnue: If False, only use fast material eval (for interior nodes)
                 If True, add NNUE evaluation (for leaf nodes)
        """
        # Fast material evaluation
        material_score = self.evaluate_position_fast(board)

        # Only call NNUE at leaf nodes to save computation
        if not use_nnue:
            return material_score

        # NNUE positional evaluation (only at leaves)
        white_feat, black_feat = board_to_halfkp_features(board)
        white_tensor = torch.tensor(white_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        black_tensor = torch.tensor(black_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            nnue_score = self.model(white_tensor, black_tensor).item()

        # Scale NNUE output as additional positional adjustment (±30 centipawns max)
        nnue_adjustment = nnue_score * 30

        # Combine: material + NNUE refinement
        total_score = material_score + nnue_adjustment

        return total_score

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

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """Quiescence search - search all captures to avoid horizon effect"""
        self.qnodes += 1

        # Use NNUE only at the very end of quiescence (when no more captures)
        # For interior q-nodes, use fast eval
        use_nnue = (depth == 0)  # Only use NNUE at first q-node
        stand_pat = self.evaluate_position(board, use_nnue=use_nnue)

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in board.legal_moves:
            if not board.is_capture(move):
                continue

            board.push(move)

            if board.is_valid():
                score = -self.quiescence_search(board, -beta, -alpha, depth + 1)

                if score >= beta:
                    board.pop()
                    return beta

                if score > alpha:
                    alpha = score

            board.pop()

        return alpha

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, allow_null: bool = True) -> float:
        """Alpha-beta search with pruning"""
        self.nodes_searched += 1

        if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0

        alpha_mate = max(alpha, -MATE_SCORE + ply)
        beta_mate = min(beta, MATE_SCORE - ply - 1)
        if alpha_mate >= beta_mate:
            return alpha_mate

        zobrist = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self.tt.probe(zobrist, depth, alpha, beta)
        if tt_score is not None:
            self.tt_hits += 1
            return tt_score

        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)

        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE + ply
            return 0

        if (allow_null and depth >= 3 and not board.is_check() and self.has_non_pawn_material(board)):
            board.push(chess.Move.null())
            score = -self.alpha_beta(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            board.pop()

            if score >= beta:
                return beta

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

    def has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if side to move has non-pawn material"""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, board.turn):
                return True
        return False

    def get_best_move(self, board: chess.Board, depth: int = 4) -> Tuple[chess.Move, float]:
        """Get best move for current position using iterative deepening"""
        best_move = None
        best_score = -float('inf')

        for current_depth in range(1, depth + 1):
            self.nodes_searched = 0
            self.qnodes = 0
            self.tt_hits = 0

            score = -float('inf')
            alpha = -MATE_SCORE
            beta = MATE_SCORE

            moves = self.order_moves(board, 0, best_move)

            for i, move in enumerate(moves):
                board.push(move)

                if i == 0:
                    score = -self.alpha_beta(board, current_depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.alpha_beta(board, current_depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self.alpha_beta(board, current_depth - 1, -beta, -alpha, 1)

                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                if score > alpha:
                    alpha = score

            print(f"Depth {current_depth}: move={best_move}, score={best_score:.0f}, nodes={self.nodes_searched:,}")

        return best_move, best_score


# Global engine variable - will be initialized on first use
engine = None


def get_engine():
    """Lazy initialization of the NNUE engine"""
    global engine
    if engine is None:
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "compact", "nnue_best.pth")
        print(f"Loading NNUE model from: {MODEL_PATH}")
        engine = NNUEEngine(MODEL_PATH, device="cpu", tt_size_mb=2)
        print("NNUE engine loaded successfully!")
    return engine


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """Main bot logic using NNUE engine with depth 5 search"""
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Get engine instance (lazy initialization)
    eng = get_engine()

    # Use NNUE engine with depth 5 search
    best_move, score = eng.get_best_move(ctx.board, depth=5)

    # Log probabilities
    move_probs = {move: 0.01 for move in legal_moves}
    if best_move in move_probs:
        move_probs[best_move] = 0.9

    total = sum(move_probs.values())
    move_probs = {move: prob / total for move, prob in move_probs.items()}
    ctx.logProbabilities(move_probs)

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset engine state for new game"""
    eng = get_engine()
    eng.tt.table.clear()
    eng.killer_moves = [[None, None] for _ in range(MAX_PLY)]
    eng.history.clear()
    print("Engine state reset for new game")
