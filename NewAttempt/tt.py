"""
Transposition table for search optimization.
Stores position evaluations with replacement scheme.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np


class TTFlag(IntEnum):
    """Bound type for stored score."""
    EXACT = 0   # PV-node, exact score
    LOWER = 1   # Beta cutoff, score >= stored
    UPPER = 2   # Alpha cutoff, score <= stored


@dataclass
class TTEntry:
    """Transposition table entry."""
    key: int            # Zobrist hash key (for verification)
    depth: int          # Search depth
    score: int          # Evaluation score (centipawns)
    flag: TTFlag        # Bound type
    best_move: int      # Best move (encoded as from_sq << 6 | to_sq)
    age: int            # Search iteration/age for replacement

    def __init__(self):
        self.key = 0
        self.depth = -1
        self.score = 0
        self.flag = TTFlag.EXACT
        self.best_move = 0
        self.age = 0


class TranspositionTable:
    """
    Hash table for position caching.
    Uses age-based replacement scheme.
    """

    def __init__(self, size_mb: int = 128):
        """
        Initialize TT with given size.

        Args:
            size_mb: Table size in megabytes
        """
        # Each entry: 8+4+4+1+2+1 = 20 bytes (with padding ~32 bytes)
        entry_size = 32
        self.size = (size_mb * 1024 * 1024) // entry_size

        # Use numpy for compact storage
        self.keys = np.zeros(self.size, dtype=np.uint64)
        self.depths = np.zeros(self.size, dtype=np.int16)
        self.scores = np.zeros(self.size, dtype=np.int32)
        self.flags = np.zeros(self.size, dtype=np.uint8)
        self.best_moves = np.zeros(self.size, dtype=np.uint16)
        self.ages = np.zeros(self.size, dtype=np.uint8)

        self.current_age = 0
        self.hits = 0
        self.misses = 0

    def clear(self):
        """Clear all entries."""
        self.keys.fill(0)
        self.depths.fill(-1)
        self.scores.fill(0)
        self.flags.fill(0)
        self.best_moves.fill(0)
        self.ages.fill(0)
        self.current_age = 0
        self.hits = 0
        self.misses = 0

    def new_search(self):
        """Increment age for new search."""
        self.current_age = (self.current_age + 1) % 256

    def probe(self, key: int, depth: int, alpha: int, beta: int) -> Optional[tuple]:
        """
        Probe table for position.

        Args:
            key: Zobrist hash
            depth: Current search depth
            alpha, beta: Search window

        Returns:
            (score, best_move) if usable, None otherwise
        """
        idx = key % self.size

        if self.keys[idx] != key:
            self.misses += 1
            return None

        self.hits += 1
        stored_depth = self.depths[idx]

        # Return best move even if depth is insufficient
        best_move = int(self.best_moves[idx])

        # Only use score if depth is sufficient
        if stored_depth < depth:
            return (None, best_move)

        score = int(self.scores[idx])
        flag = TTFlag(self.flags[idx])

        # Check if score is usable
        if flag == TTFlag.EXACT:
            return (score, best_move)
        elif flag == TTFlag.LOWER and score >= beta:
            return (score, best_move)
        elif flag == TTFlag.UPPER and score <= alpha:
            return (score, best_move)

        return (None, best_move)

    def store(self, key: int, depth: int, score: int, flag: TTFlag, best_move: int):
        """
        Store position in table.

        Args:
            key: Zobrist hash
            depth: Search depth
            score: Evaluation
            flag: Bound type
            best_move: Best move (encoded)
        """
        idx = key % self.size

        # Replacement scheme: replace if:
        # 1. Empty slot
        # 2. Same position (always replace)
        # 3. Older age
        # 4. Lower depth and same age
        if self.keys[idx] == 0:
            replace = True
        elif self.keys[idx] == key:
            replace = True
        elif self.ages[idx] != self.current_age:
            replace = True
        elif self.depths[idx] <= depth:
            replace = True
        else:
            replace = False

        if replace:
            self.keys[idx] = key
            self.depths[idx] = depth
            self.scores[idx] = score
            self.flags[idx] = flag
            self.best_moves[idx] = best_move
            self.ages[idx] = self.current_age

    def get_stats(self) -> dict:
        """Get table statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        filled = np.sum(self.keys != 0)

        return {
            'size': self.size,
            'filled': filled,
            'fill_rate': filled / self.size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }
