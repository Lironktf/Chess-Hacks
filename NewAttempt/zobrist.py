"""
Zobrist hashing for fast position keys.
Precomputes random keys for incremental hash updates.
"""
import numpy as np
from typing import Dict

# Piece encoding: (color, piece_type) -> 0-11
# 0-5: white pieces (P,N,B,R,Q,K), 6-11: black pieces
PIECE_TYPES = 6
COLORS = 2
SQUARES = 64

class ZobristKeys:
    """Precomputed Zobrist hash keys for incremental hashing."""

    def __init__(self, seed: int = 42):
        """Initialize Zobrist keys with deterministic seed for reproducibility."""
        rng = np.random.RandomState(seed)

        # Piece-square keys: [piece_type][square]
        # piece_type: 0-11 (white P,N,B,R,Q,K, black P,N,B,R,Q,K)
        self.pieces = rng.randint(0, 2**64, size=(12, 64), dtype=np.uint64)

        # Castling rights: 4 bits (WK, WQ, BK, BQ)
        self.castling = rng.randint(0, 2**64, size=4, dtype=np.uint64)

        # En passant file: 8 files (a-h)
        self.en_passant = rng.randint(0, 2**64, size=8, dtype=np.uint64)

        # Side to move
        self.side = rng.randint(0, 2**64, dtype=np.uint64)

    def piece_key(self, piece_idx: int, square: int) -> int:
        """Get key for piece at square. piece_idx: 0-11, square: 0-63."""
        return int(self.pieces[piece_idx, square])

    def castling_key(self, castling_rights: int) -> int:
        """Get key for castling rights (4-bit mask)."""
        key = np.uint64(0)
        for i in range(4):
            if castling_rights & (1 << i):
                key ^= self.castling[i]
        return int(key)

    def ep_key(self, ep_square: int) -> int:
        """Get key for en passant square (file 0-7, or -1 for none)."""
        if ep_square < 0:
            return 0
        file = ep_square % 8
        return int(self.en_passant[file])

    def side_key(self) -> int:
        """Get key for side to move."""
        return int(self.side)


# Global instance
ZOBRIST = ZobristKeys()


def compute_hash(board) -> int:
    """
    Compute full Zobrist hash from python-chess Board.
    Used for initial position or verification.
    """
    from chess import PIECE_TYPES as PT

    hash_val = np.uint64(0)

    # Piece placement
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            # Map chess.Piece to 0-11 index
            piece_idx = (piece.piece_type - 1) + (0 if piece.color else 6)
            hash_val ^= ZOBRIST.pieces[piece_idx, square]

    # Castling rights
    castling_rights = 0
    if board.has_kingside_castling_rights(True):
        castling_rights |= 1
    if board.has_queenside_castling_rights(True):
        castling_rights |= 2
    if board.has_kingside_castling_rights(False):
        castling_rights |= 4
    if board.has_queenside_castling_rights(False):
        castling_rights |= 8
    hash_val ^= ZOBRIST.castling_key(castling_rights)

    # En passant
    if board.ep_square is not None:
        hash_val ^= ZOBRIST.ep_key(board.ep_square)

    # Side to move
    if not board.turn:  # Black to move
        hash_val ^= ZOBRIST.side

    return int(hash_val)
