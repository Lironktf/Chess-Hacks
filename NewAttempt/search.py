"""
Chess search engine with alpha-beta pruning, move ordering, and quiescence.
Implements negamax with iterative deepening and PVS.
"""
import chess
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tt import TranspositionTable, TTFlag
from nnue import NNUEEvaluator
from zobrist import compute_hash


# Constants
MATE_SCORE = 100000
MAX_PLY = 128
INFINITY = 2 * MATE_SCORE

# Piece values for MVV-LVA
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


@dataclass
class SearchStats:
    """Statistics for search analysis."""
    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    beta_cutoffs: int = 0
    time_ms: float = 0
    depth: int = 0
    seldepth: int = 0

    def nps(self) -> int:
        """Nodes per second."""
        if self.time_ms > 0:
            return int((self.nodes + self.qnodes) * 1000 / self.time_ms)
        return 0


class SearchEngine:
    """
    Chess search engine with NNUE evaluation.
    """

    def __init__(self, evaluator: NNUEEvaluator, tt_size_mb: int = 128):
        """Initialize search engine."""
        self.evaluator = evaluator
        self.tt = TranspositionTable(tt_size_mb)

        # Move ordering tables
        self.killer_moves = [[0, 0] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(64)]  # [from_sq][to_sq]

        # Search state
        self.stats = SearchStats()
        self.stop_search = False
        self.start_time = 0
        self.time_limit_ms = 0
        self.node_limit = 0

        # PV tracking
        self.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]
        self.pv_length = [0] * MAX_PLY

    def clear_tables(self):
        """Clear search tables for new game."""
        self.tt.clear()
        self.killer_moves = [[0, 0] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(64)]

    def _encode_move(self, move: chess.Move) -> int:
        """Encode move as integer for storage."""
        return (move.from_square << 6) | move.to_square

    def _decode_move(self, board: chess.Board, encoded: int) -> Optional[chess.Move]:
        """Decode move from integer."""
        if encoded == 0:
            return None

        from_sq = encoded >> 6
        to_sq = encoded & 63

        # Find matching legal move (handles promotions)
        for move in board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                return move

        return None

    def _mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        """Most Valuable Victim - Least Valuable Attacker score."""
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None:
            return 0

        victim_value = PIECE_VALUES.get(victim.piece_type, 0)
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0)

        return victim_value * 100 - attacker_value

    def _order_moves(self, board: chess.Board, moves: List[chess.Move],
                     tt_move: Optional[chess.Move], ply: int) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning.
        Priority: TT move > captures (MVV-LVA) > killers > history > others
        """
        scored_moves = []

        tt_move_encoded = self._encode_move(tt_move) if tt_move else 0
        killer1 = self.killer_moves[ply][0]
        killer2 = self.killer_moves[ply][1]

        for move in moves:
            encoded = self._encode_move(move)
            score = 0

            # TT move
            if encoded == tt_move_encoded:
                score = 1_000_000
            # Captures
            elif board.is_capture(move):
                score = 900_000 + self._mvv_lva(board, move)
            # Promotions
            elif move.promotion:
                promo_value = PIECE_VALUES.get(move.promotion, 0)
                score = 800_000 + promo_value
            # Killer moves
            elif encoded == killer1:
                score = 700_000
            elif encoded == killer2:
                score = 600_000
            # History heuristic
            else:
                score = self.history[move.from_square][move.to_square]

            scored_moves.append((score, move))

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def _update_killers(self, move: chess.Move, ply: int):
        """Update killer move table."""
        encoded = self._encode_move(move)

        if self.killer_moves[ply][0] != encoded:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = encoded

    def _update_history(self, move: chess.Move, depth: int):
        """Update history heuristic."""
        bonus = depth * depth
        self.history[move.from_square][move.to_square] += bonus

        # Cap to prevent overflow
        if self.history[move.from_square][move.to_square] > 10000:
            # Age all history scores
            for i in range(64):
                for j in range(64):
                    self.history[i][j] //= 2

    def _is_draw(self, board: chess.Board) -> bool:
        """Check for draw conditions."""
        # Fifty-move rule
        if board.halfmove_clock >= 100:
            return True

        # Insufficient material
        if board.is_insufficient_material():
            return True

        # Threefold repetition (simplified check)
        if board.is_repetition(2):
            return True

        return False

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """
        Quiescence search to resolve tactical sequences.
        Only searches captures and checks.
        """
        self.stats.qnodes += 1

        # Update selective depth
        if ply > self.stats.seldepth:
            self.stats.seldepth = ply

        # Check time limit
        if self.time_limit_ms > 0:
            elapsed = (time.time() - self.start_time) * 1000
            if elapsed >= self.time_limit_ms:
                self.stop_search = True
                return 0

        # Stand pat
        stand_pat = self.evaluator.evaluate(board)

        if stand_pat >= beta:
            return beta

        # Delta pruning: if stand_pat + queen_value < alpha, prune
        if stand_pat + 900 < alpha:
            return alpha

        if stand_pat > alpha:
            alpha = stand_pat

        # Generate and order captures
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures = self._order_moves(board, captures, None, ply)

        for move in captures:
            # Delta pruning: skip if captured piece value + stand_pat < alpha
            victim = board.piece_at(move.to_square)
            if victim:
                victim_value = PIECE_VALUES.get(victim.piece_type, 0)
                if stand_pat + victim_value + 200 < alpha:
                    continue

            # Make move
            self.evaluator.push(board, move)
            board.push(move)

            score = -self.quiesce(board, -beta, -alpha, ply + 1)

            # Unmake
            board.pop()
            self.evaluator.pop()

            if self.stop_search:
                return 0

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int, do_null: bool = True) -> int:
        """
        Negamax search with alpha-beta pruning.

        Args:
            board: Current position
            depth: Remaining depth
            alpha, beta: Search window
            ply: Distance from root
            do_null: Whether null-move pruning is allowed

        Returns:
            Evaluation score
        """
        # Update PV length
        self.pv_length[ply] = ply

        # Check for draw
        if ply > 0 and self._is_draw(board):
            return 0

        # Mate distance pruning
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        # Quiescence at leaf nodes
        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        self.stats.nodes += 1

        # Check limits
        if self.node_limit > 0 and self.stats.nodes >= self.node_limit:
            self.stop_search = True
            return 0

        if self.time_limit_ms > 0:
            elapsed = (time.time() - self.start_time) * 1000
            if elapsed >= self.time_limit_ms:
                self.stop_search = True
                return 0

        # Probe transposition table
        key = compute_hash(board)
        tt_result = self.tt.probe(key, depth, alpha, beta)

        tt_move = None
        if tt_result:
            score, best_move_encoded = tt_result
            if best_move_encoded:
                tt_move = self._decode_move(board, best_move_encoded)

            if score is not None:
                self.stats.tt_cutoffs += 1
                return score

            self.stats.tt_hits += 1

        # Null move pruning
        if (do_null and depth >= 3 and not board.is_check() and
            ply > 0 and self._has_non_pawn_material(board)):

            R = 2  # Reduction depth

            board.push(chess.Move.null())
            score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, False)
            board.pop()

            if self.stop_search:
                return 0

            if score >= beta:
                return beta

        # Generate and order moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # Checkmate or stalemate
            if board.is_check():
                return -MATE_SCORE + ply
            else:
                return 0

        moves = self._order_moves(board, legal_moves, tt_move, ply)

        best_score = -INFINITY
        best_move = moves[0]
        moves_searched = 0
        flag = TTFlag.UPPER

        for i, move in enumerate(moves):
            # Make move
            self.evaluator.push(board, move)
            board.push(move)

            # Principal Variation Search (PVS)
            if i == 0:
                # Full window search for first move
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late Move Reduction (LMR)
                reduction = 0
                if (depth >= 3 and moves_searched >= 4 and
                    not board.is_capture(move) and not move.promotion and
                    not board.is_check()):
                    reduction = 1

                # Null window search
                score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)

                # Re-search if necessary
                if score > alpha and (score < beta or reduction > 0):
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)

            # Unmake
            board.pop()
            self.evaluator.pop()

            moves_searched += 1

            if self.stop_search:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

                # Update PV
                self.pv_table[ply][ply] = self._encode_move(move)
                for next_ply in range(ply + 1, self.pv_length[ply + 1]):
                    self.pv_table[ply][next_ply] = self.pv_table[ply + 1][next_ply]
                self.pv_length[ply] = self.pv_length[ply + 1]

            if score >= beta:
                # Beta cutoff
                self.stats.beta_cutoffs += 1
                flag = TTFlag.LOWER

                # Update move ordering tables
                if not board.is_capture(move):
                    self._update_killers(move, ply)
                    self._update_history(move, depth)

                break

            if score > alpha:
                alpha = score
                flag = TTFlag.EXACT

        # Store in transposition table
        self.tt.store(key, depth, best_score, flag, self._encode_move(best_move))

        return best_score

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if side to move has non-pawn material."""
        color = board.turn
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, color):
                return True
        return False

    def search(self, board: chess.Board, depth: Optional[int] = None,
               time_limit_ms: Optional[int] = None,
               node_limit: Optional[int] = None) -> Tuple[chess.Move, int, SearchStats]:
        """
        Iterative deepening search.

        Args:
            board: Position to search
            depth: Maximum depth (default: 6)
            time_limit_ms: Time limit in milliseconds
            node_limit: Node limit

        Returns:
            (best_move, score, stats)
        """
        if depth is None:
            depth = 6

        # Initialize search state
        self.stats = SearchStats()
        self.stop_search = False
        self.start_time = time.time()
        self.time_limit_ms = time_limit_ms or 0
        self.node_limit = node_limit or 0

        # Reset accumulators
        self.evaluator.reset_accumulators(board)

        # New search
        self.tt.new_search()

        best_move = None
        best_score = 0

        # Iterative deepening
        for d in range(1, depth + 1):
            if self.stop_search:
                break

            # Aspiration window (skip for shallow depths)
            if d >= 4 and best_score != 0:
                window = 50
                alpha = best_score - window
                beta = best_score + window
            else:
                alpha = -INFINITY
                beta = INFINITY

            # Search with aspiration window
            score = self.negamax(board, d, alpha, beta, 0)

            # Re-search if outside window
            if (score <= alpha or score >= beta) and not self.stop_search:
                score = self.negamax(board, d, -INFINITY, INFINITY, 0)

            if self.stop_search:
                break

            best_score = score
            self.stats.depth = d

            # Get best move from PV
            if self.pv_length[0] > 0:
                encoded = self.pv_table[0][0]
                move = self._decode_move(board, encoded)
                if move:
                    best_move = move

        # Update stats
        self.stats.time_ms = (time.time() - self.start_time) * 1000

        # Fallback if no move found
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]

        return best_move, best_score, self.stats

    def get_pv(self, board: chess.Board) -> List[str]:
        """Get principal variation in SAN notation."""
        pv = []
        test_board = board.copy()

        for i in range(self.pv_length[0]):
            encoded = self.pv_table[0][i]
            move = self._decode_move(test_board, encoded)

            if move is None or move not in test_board.legal_moves:
                break

            pv.append(test_board.san(move))
            test_board.push(move)

        return pv
