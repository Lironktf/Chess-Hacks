"""
Main chess engine API.
Provides high-level interface for position analysis and best move selection.
"""
import chess
import chess.pgn
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from nnue import NNUEEvaluator
from search import SearchEngine, SearchStats


class ChessEngine:
    """
    High-performance chess engine with NNUE evaluation.

    Example usage:
        engine = ChessEngine()
        engine.load_weights('model.npz')
        engine.set_position('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        move, score, info = engine.get_best_move(depth=6)
        analysis = engine.analyze_position(depth=8)
    """

    def __init__(self, model_path: Optional[str] = None, tt_size_mb: int = 128):
        """
        Initialize engine.

        Args:
            model_path: Path to NNUE weights (optional)
            tt_size_mb: Transposition table size in MB
        """
        self.evaluator = NNUEEvaluator(model_path)
        self.searcher = SearchEngine(self.evaluator, tt_size_mb)
        self.board = chess.Board()

    def load_weights(self, path: str):
        """Load NNUE weights from file."""
        self.evaluator.load_weights(path)

    def save_weights(self, path: str):
        """Save NNUE weights to file."""
        self.evaluator.save_weights(path)

    def new_game(self):
        """Start a new game."""
        self.board = chess.Board()
        self.searcher.clear_tables()

    def set_position(self, fen: str):
        """
        Set position from FEN string.

        Args:
            fen: FEN position string
        """
        self.board = chess.Board(fen)

    def set_position_from_moves(self, moves: List[str], starting_fen: Optional[str] = None):
        """
        Set position from move sequence.

        Args:
            moves: List of moves in SAN or UCI notation
            starting_fen: Starting position (default: standard starting position)
        """
        if starting_fen:
            self.board = chess.Board(starting_fen)
        else:
            self.board = chess.Board()

        for move_str in moves:
            try:
                # Try SAN notation
                move = self.board.parse_san(move_str)
            except ValueError:
                # Try UCI notation
                try:
                    move = chess.Move.from_uci(move_str)
                except ValueError:
                    raise ValueError(f"Invalid move: {move_str}")

            if move not in self.board.legal_moves:
                raise ValueError(f"Illegal move: {move_str}")

            self.board.push(move)

    def make_move(self, move_str: str):
        """
        Make a move on the board.

        Args:
            move_str: Move in SAN or UCI notation

        Returns:
            chess.Move object
        """
        try:
            move = self.board.parse_san(move_str)
        except ValueError:
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                raise ValueError(f"Invalid move: {move_str}")

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_str}")

        self.board.push(move)
        return move

    def get_best_move(self, depth: Optional[int] = None,
                     time_limit_ms: Optional[int] = None,
                     node_limit: Optional[int] = None) -> Tuple[chess.Move, int, Dict]:
        """
        Find best move for current position.

        Args:
            depth: Search depth (default: 6)
            time_limit_ms: Time limit in milliseconds
            node_limit: Node limit

        Returns:
            (best_move, score, info_dict)
        """
        move, score, stats = self.searcher.search(
            self.board, depth, time_limit_ms, node_limit
        )

        info = {
            'depth': stats.depth,
            'seldepth': stats.seldepth,
            'nodes': stats.nodes + stats.qnodes,
            'qnodes': stats.qnodes,
            'time_ms': stats.time_ms,
            'nps': stats.nps(),
            'score_cp': score,
            'pv': self.searcher.get_pv(self.board),
            'tt_hits': stats.tt_hits,
            'tt_cutoffs': stats.tt_cutoffs,
            'beta_cutoffs': stats.beta_cutoffs,
        }

        return move, score, info

    def analyze_position(self, depth: int = 8, lines: int = 1) -> List[Dict]:
        """
        Analyze current position with multiple variations.

        Args:
            depth: Search depth
            lines: Number of variations to analyze (currently only supports 1)

        Returns:
            List of analysis dictionaries with move, score, and PV
        """
        # For now, just return single best line
        # Multi-PV search can be added later
        move, score, info = self.get_best_move(depth=depth)

        analysis = [{
            'move': self.board.san(move),
            'move_uci': move.uci(),
            'score_cp': score,
            'depth': info['depth'],
            'pv': info['pv'],
            'pv_san': ' '.join(info['pv']),
            'nodes': info['nodes'],
            'nps': info['nps'],
            'time_ms': info['time_ms'],
        }]

        return analysis

    def evaluate_position(self) -> int:
        """
        Get static evaluation of current position.

        Returns:
            Evaluation in centipawns from side-to-move perspective
        """
        self.evaluator.reset_accumulators(self.board)
        return self.evaluator.evaluate(self.board)

    def get_fen(self) -> str:
        """Get FEN of current position."""
        return self.board.fen()

    def get_board(self) -> chess.Board:
        """Get current board object."""
        return self.board.copy()

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_result(self) -> Optional[str]:
        """
        Get game result if game is over.

        Returns:
            '1-0', '0-1', '1/2-1/2', or None
        """
        if not self.board.is_game_over():
            return None

        result = self.board.result()
        return result

    def perft(self, depth: int) -> int:
        """
        Performance test: count leaf nodes at given depth.
        Useful for debugging move generation.

        Args:
            depth: Depth to search

        Returns:
            Number of leaf nodes
        """
        if depth == 0:
            return 1

        count = 0
        for move in self.board.legal_moves:
            self.board.push(move)
            count += self.perft(depth - 1)
            self.board.pop()

        return count

    def perft_divide(self, depth: int) -> Dict[str, int]:
        """
        Perft with per-move breakdown.

        Args:
            depth: Depth to search

        Returns:
            Dictionary mapping moves to node counts
        """
        results = {}
        total = 0

        for move in self.board.legal_moves:
            self.board.push(move)
            count = self.perft(depth - 1)
            self.board.pop()

            move_str = self.board.san(move)
            results[move_str] = count
            total += count

        results['total'] = total
        return results

    def get_legal_moves(self) -> List[str]:
        """Get list of legal moves in SAN notation."""
        return [self.board.san(move) for move in self.board.legal_moves]

    def play_game(self, depth: int = 6, max_moves: int = 200) -> str:
        """
        Play a game against itself.

        Args:
            depth: Search depth for both sides
            max_moves: Maximum number of moves

        Returns:
            PGN string of the game
        """
        self.new_game()
        moves = []

        for _ in range(max_moves):
            if self.board.is_game_over():
                break

            move, score, info = self.get_best_move(depth=depth)
            moves.append(self.board.san(move))
            self.board.push(move)

        # Create PGN
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-play"
        game.headers["Result"] = self.board.result()

        node = game
        self.new_game()
        for move_san in moves:
            move = self.board.parse_san(move_san)
            node = node.add_variation(move)
            self.board.push(move)

        return str(game)

    def print_board(self):
        """Print current board position."""
        print(self.board)
        print(f"FEN: {self.board.fen()}")
        print(f"Turn: {'White' if self.board.turn else 'Black'}")
        eval_cp = self.evaluate_position()
        print(f"Static eval: {eval_cp:+.2f} cp")


def main():
    """Example usage."""
    print("ChessBot Engine - High Performance NNUE Chess Engine")
    print("=" * 60)

    # Create engine
    engine = ChessEngine()

    # Test perft for move generation validation
    print("\nPerft test (depth 4)...")
    print(f"Nodes: {engine.perft(4):,}")

    # Test evaluation
    print("\nStatic evaluation from starting position:")
    print(f"Score: {engine.evaluate_position():+d} cp")

    # Test search
    print("\nSearching starting position (depth 6)...")
    move, score, info = engine.get_best_move(depth=6)

    print(f"\nBest move: {engine.board.san(move)}")
    print(f"Score: {score:+d} cp")
    print(f"Depth: {info['depth']}")
    print(f"Nodes: {info['nodes']:,}")
    print(f"NPS: {info['nps']:,}")
    print(f"Time: {info['time_ms']:.1f} ms")
    print(f"PV: {' '.join(info['pv'][:8])}")

    # Test a tactical position
    print("\n" + "=" * 60)
    print("Testing tactical position (Scholar's Mate defense):")
    engine.set_position_from_moves(['e4', 'e5', 'Bc4', 'Nc6', 'Qh5', 'Nf6'])
    engine.print_board()

    print("\nSearching...")
    move, score, info = engine.get_best_move(depth=6)

    print(f"\nBest move: {engine.board.san(move)} (defends mate)")
    print(f"Score: {score:+d} cp")
    print(f"PV: {' '.join(info['pv'][:8])}")


if __name__ == '__main__':
    main()
