"""
Advanced Chess Engine with NNUE evaluation

Features:
- NNUE position evaluation
- Alpha-beta pruning with quiescence search
- Transposition table
- Iterative deepening
- Advanced move ordering (MVV-LVA, killer moves, history heuristic)
- Null move pruning

This is a STRONG engine designed to maximize playing strength within 10MB
"""

import os
import torch
import chess
import chess.polyglot
import numpy as np
from typing import Optional, Tuple, Dict
from collections import defaultdict

from nnue_model import NNUE, board_to_halfkp_features


# Piece values for move ordering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Evaluation constants
MATE_SCORE = 30000
MAX_PLY = 100


class TranspositionTable:
    """
    Transposition table for storing previously evaluated positions
    This dramatically speeds up search by avoiding re-evaluation
    """

    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

    def __init__(self, size_mb=2):
        """
        Args:
            size_mb: Size of TT in megabytes
        """
        # Each entry: hash(8) + score(4) + depth(1) + flag(1) + move(2) = ~16 bytes
        self.max_entries = int((size_mb * 1024 * 1024) / 16)
        self.table = {}

    def probe(self, zobrist_hash, depth, alpha, beta):
        """
        Probe the transposition table

        Returns:
            (score, move) if found and usable, else (None, None)
        """
        if zobrist_hash not in self.table:
            return None, None

        entry = self.table[zobrist_hash]
        stored_depth, stored_score, stored_flag, stored_move = entry

        # Only use if stored depth is >= current depth
        if stored_depth >= depth:
            if stored_flag == self.EXACT:
                return stored_score, stored_move
            elif stored_flag == self.LOWER_BOUND and stored_score >= beta:
                return stored_score, stored_move
            elif stored_flag == self.UPPER_BOUND and stored_score <= alpha:
                return stored_score, stored_move

        # Return best move even if score not usable
        return None, stored_move

    def store(self, zobrist_hash, depth, score, flag, best_move):
        """Store position in transposition table"""
        # Simple replacement scheme: always replace
        # (More sophisticated: replace if depth is higher)
        if zobrist_hash in self.table:
            old_depth = self.table[zobrist_hash][0]
            if depth < old_depth:
                return  # Don't replace with shallower search

        self.table[zobrist_hash] = (depth, score, flag, best_move)

        # Limit table size
        if len(self.table) > self.max_entries:
            # Remove oldest entries (simple approach)
            keys = list(self.table.keys())
            for key in keys[:len(keys) // 4]:  # Remove 25%
                del self.table[key]


class NNUEEngine:
    """
    Advanced chess engine with NNUE evaluation
    """

    def __init__(self, model_path: str, device: str = "cpu", tt_size_mb=2):
        """
        Initialize the engine

        Args:
            model_path: Path to NNUE model weights
            device: Device to run inference on
            tt_size_mb: Transposition table size in MB
        """
        self.device = torch.device(device)
        self.model = NNUE(hidden1_size=1408, hidden2_size=32).to(self.device)

        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different naming conventions (ft_w/ft_b vs ft_white/ft_black)
        # Map old names to new names if needed
        if "ft_w.weight" in state_dict and "ft_white.weight" not in state_dict:
            # Model uses old naming (ft_w, ft_b), map to new names
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
        self.killer_moves = [[None, None] for _ in range(MAX_PLY)]  # 2 killers per ply
        self.history = defaultdict(int)  # History heuristic

        # Statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.qnodes = 0

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate position using NNUE

        Returns:
            Evaluation in centipawns from white's perspective
        """
        # Get features
        white_feat, black_feat = board_to_halfkp_features(board)

        # Convert to tensors
        white_tensor = torch.tensor(white_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        black_tensor = torch.tensor(black_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Evaluate
        with torch.no_grad():
            score = self.model(white_tensor, black_tensor).item()

        # Convert to centipawns (scale by ~200 to match traditional evaluation)
        score = score * 200

        return score

    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """
        Most Valuable Victim - Least Valuable Attacker score
        Prioritize captures of valuable pieces with cheap pieces
        """
        if not board.is_capture(move):
            return 0

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        # MVV-LVA: victim_value * 10 - attacker_value
        return PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]

    def order_moves(self, board: chess.Board, ply: int, tt_move: Optional[chess.Move] = None):
        """
        Order moves for better alpha-beta pruning

        Move ordering:
        1. TT move (best move from transposition table)
        2. Winning captures (MVV-LVA)
        3. Killer moves
        4. Other captures
        5. Quiet moves (sorted by history heuristic)
        """
        moves = list(board.legal_moves)
        move_scores = []

        for move in moves:
            score = 0

            # TT move gets highest priority
            if tt_move and move == tt_move:
                score = 1_000_000

            # Captures
            elif board.is_capture(move):
                mvv_lva = self.mvv_lva_score(board, move)
                score = 100_000 + mvv_lva

            # Killer moves
            elif move in self.killer_moves[ply]:
                score = 10_000

            # Quiet moves - use history heuristic
            else:
                score = self.history[(move.from_square, move.to_square)]

            # Bonus for checks
            board.push(move)
            if board.is_check():
                score += 5_000
            board.pop()

            # Promotions
            if move.promotion:
                score += 50_000

            move_scores.append((move, score))

        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """
        Quiescence search - search all captures to avoid horizon effect

        This prevents the engine from missing tactics just beyond the search depth
        """
        self.qnodes += 1

        # Stand pat score
        stand_pat = self.evaluate_position(board)

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Only search captures and checks
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue

            board.push(move)

            # Skip if position is illegal
            if board.is_valid():
                score = -self.quiescence_search(board, -beta, -alpha)

                if score >= beta:
                    board.pop()
                    return beta

                if score > alpha:
                    alpha = score

            board.pop()

        return alpha

    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        allow_null: bool = True
    ) -> float:
        """
        Alpha-beta search with pruning

        Args:
            board: Current position
            depth: Remaining depth
            alpha: Alpha value
            beta: Beta value
            ply: Current ply from root
            allow_null: Whether null move is allowed

        Returns:
            Best evaluation
        """
        self.nodes_searched += 1

        # Check for draw
        if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0

        # Mate distance pruning
        alpha_mate = max(alpha, -MATE_SCORE + ply)
        beta_mate = min(beta, MATE_SCORE - ply - 1)
        if alpha_mate >= beta_mate:
            return alpha_mate

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
            return 0  # Stalemate

        # Null move pruning
        if (allow_null and depth >= 3 and not board.is_check()
            and self.has_non_pawn_material(board)):
            board.push(chess.Move.null())
            score = -self.alpha_beta(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            board.pop()

            if score >= beta:
                return beta

        # Get ordered moves
        moves = self.order_moves(board, ply, tt_move)

        if not moves:
            if board.is_check():
                return -MATE_SCORE + ply
            return 0  # Stalemate

        best_score = -float('inf')
        best_move = None
        flag = TranspositionTable.UPPER_BOUND

        for i, move in enumerate(moves):
            board.push(move)

            # PVS (Principal Variation Search)
            if i == 0:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Null window search
                score = -self.alpha_beta(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    # Re-search with full window
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                flag = TranspositionTable.EXACT

            if score >= beta:
                # Beta cutoff
                flag = TranspositionTable.LOWER_BOUND

                # Update killer moves (only for quiet moves)
                if not board.is_capture(move):
                    if self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move

                    # Update history
                    self.history[(move.from_square, move.to_square)] += depth * depth

                break

        # Store in transposition table
        self.tt.store(zobrist, depth, best_score, flag, best_move)

        return best_score

    def has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if side to move has non-pawn material"""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, board.turn):
                return True
        return False

    def iterative_deepening(self, board: chess.Board, max_depth: int) -> Tuple[chess.Move, float]:
        """
        Iterative deepening search - gradually increase depth

        This allows us to use TT effectively and provides better move ordering
        """
        best_move = None
        best_score = -float('inf')

        print(f"Starting iterative deepening to depth {max_depth}...")

        for depth in range(1, max_depth + 1):
            self.nodes_searched = 0
            self.qnodes = 0
            self.tt_hits = 0

            # Search
            score = -float('inf')
            alpha = -MATE_SCORE
            beta = MATE_SCORE

            moves = self.order_moves(board, 0, best_move)

            for i, move in enumerate(moves):
                board.push(move)

                if i == 0:
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.alpha_beta(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self.alpha_beta(board, depth - 1, -beta, -alpha, 1)

                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

                if score > alpha:
                    alpha = score

            print(f"Depth {depth}: move={best_move}, score={best_score:.0f}, "
                  f"nodes={self.nodes_searched:,}, qnodes={self.qnodes:,}, "
                  f"tt_hits={self.tt_hits:,}")

        return best_move, best_score

    def get_move(self, board: chess.Board, depth: int = 5) -> chess.Move:
        """
        Get best move for current position

        Args:
            board: Current position
            depth: Search depth

        Returns:
            Best move
        """
        move, score = self.iterative_deepening(board, depth)
        print(f"\nBest move: {move} (eval: {score:.0f})")
        return move


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("NNUE CHESS ENGINE")
    print("=" * 60)
    print()

    # Check for model
    if not os.path.exists("nnue_best.pth") and not os.path.exists("nnue_model.pth"):
        print("No NNUE model found!")
        print("Creating a random model for testing...")

        model = NNUE(hidden1_size=512, hidden2_size=32)
        torch.save(model.state_dict(), "nnue_random.pth")
        model_path = "nnue_random.pth"
    else:
        model_path = "nnue_best.pth" if os.path.exists("nnue_best.pth") else "nnue_model.pth"

    print(f"Loading model: {model_path}")
    engine = NNUEEngine(model_path, tt_size_mb=2)
    print("✓ Engine loaded\n")

    # Test position
    board = chess.Board()
    print("Starting position:")
    print(board)
    print()

    print("Finding best move (depth 3)...")
    move = engine.get_move(board, depth=3)
