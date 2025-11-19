"""
NNUE (Efficiently Updatable Neural Network) evaluator.
Architecture: HalfKP features -> 512 -> 256 -> 1
Supports incremental accumulator updates for fast evaluation.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import chess


# Feature indices
# HalfKP: (piece_type, square, king_square, perspective)
# 6 piece types * 64 squares * 64 king squares * 2 perspectives
# But we use a simplified version: piece-square from each perspective

NUM_PIECES = 10  # P,N,B,R,Q (no king in features) * 2 colors
NUM_SQUARES = 64
FEATURE_SIZE = NUM_PIECES * NUM_SQUARES * 64  # piece-square per king position
HIDDEN1 = 512
HIDDEN2 = 256


def piece_to_index(piece: chess.Piece) -> int:
    """Convert piece to feature index (0-9, or -1 for king)."""
    # 0-4: white P,N,B,R,Q, 5-9: black P,N,B,R,Q
    if piece.piece_type == chess.KING:
        return -1
    idx = piece.piece_type - 1  # P=0, N=1, B=2, R=3, Q=4
    if not piece.color:  # Black
        idx += 5
    return idx


def get_feature_index(piece_idx: int, square: int, king_sq: int, flip: bool) -> int:
    """
    Get feature index for piece at square relative to king.

    Args:
        piece_idx: 0-9 (piece type and color)
        square: 0-63 (piece location)
        king_sq: 0-63 (friendly king location)
        flip: Whether to flip perspective (for black)
    """
    if flip:
        square = square ^ 56  # Flip rank
        king_sq = king_sq ^ 56
        # Flip piece color
        piece_idx = (piece_idx + 5) % 10

    return piece_idx * NUM_SQUARES * 64 + square * 64 + king_sq


class NNUENetwork(nn.Module):
    """PyTorch NNUE network for training."""

    def __init__(self):
        super().__init__()

        # Feature transformer: sparse input to dense hidden
        self.ft = nn.Linear(FEATURE_SIZE, HIDDEN1)

        # Output network
        self.fc1 = nn.Linear(HIDDEN1 * 2, HIDDEN2)  # Concat both perspectives
        self.fc2 = nn.Linear(HIDDEN2, 1)

    def forward(self, white_features, black_features):
        """
        Forward pass.

        Args:
            white_features: [batch, FEATURE_SIZE] sparse white perspective
            black_features: [batch, FEATURE_SIZE] sparse black perspective
        """
        # Feature transform with ReLU
        white_ft = torch.clamp(self.ft(white_features), 0, 1)  # ClippedReLU
        black_ft = torch.clamp(self.ft(black_features), 0, 1)

        # Concatenate perspectives (side to move first)
        combined = torch.cat([white_ft, black_ft], dim=1)

        # Output network
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        return x.squeeze(-1)


class NNUEEvaluator:
    """
    Fast NNUE evaluator with incremental updates.
    Maintains accumulator for both white and black perspectives.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize evaluator, optionally loading weights."""
        # Store weights as numpy for fast inference
        self.ft_weights = np.zeros((FEATURE_SIZE, HIDDEN1), dtype=np.float32)
        self.ft_bias = np.zeros(HIDDEN1, dtype=np.float32)
        self.fc1_weights = np.zeros((HIDDEN1 * 2, HIDDEN2), dtype=np.float32)
        self.fc1_bias = np.zeros(HIDDEN2, dtype=np.float32)
        self.fc2_weights = np.zeros((HIDDEN2, 1), dtype=np.float32)
        self.fc2_bias = np.float32(0)

        # Accumulators for incremental updates
        self.white_acc = np.zeros(HIDDEN1, dtype=np.float32)
        self.black_acc = np.zeros(HIDDEN1, dtype=np.float32)

        # Accumulator stack for unmake
        self.acc_stack = []

        if model_path:
            self.load_weights(model_path)
        else:
            # Initialize with small random weights
            self._init_random_weights()

    def _init_random_weights(self):
        """Initialize with small random weights for testing."""
        rng = np.random.RandomState(42)

        # Xavier/Glorot initialization for better numerical stability
        # scale by sqrt(2 / fan_in) for ReLU activations
        scale_ft = np.sqrt(2.0 / FEATURE_SIZE)
        scale_fc1 = np.sqrt(2.0 / (HIDDEN1 * 2))
        scale_fc2 = np.sqrt(2.0 / HIDDEN2)

        self.ft_weights = (rng.randn(FEATURE_SIZE, HIDDEN1).astype(np.float32) * scale_ft).clip(-0.1, 0.1)
        self.ft_bias = np.zeros(HIDDEN1, dtype=np.float32)
        self.fc1_weights = (rng.randn(HIDDEN1 * 2, HIDDEN2).astype(np.float32) * scale_fc1).clip(-0.1, 0.1)
        self.fc1_bias = np.zeros(HIDDEN2, dtype=np.float32)
        self.fc2_weights = (rng.randn(HIDDEN2, 1).astype(np.float32) * scale_fc2).clip(-0.1, 0.1)
        self.fc2_bias = np.float32(0.0)

    def load_weights(self, path: str):
        """Load weights from PyTorch checkpoint or numpy file."""
        if path.endswith('.pt') or path.endswith('.pth'):
            self._load_torch_weights(path)
        elif path.endswith('.npz'):
            self._load_numpy_weights(path)
        else:
            raise ValueError(f"Unknown weight format: {path}")

    def _load_torch_weights(self, path: str):
        """Load from PyTorch model."""
        checkpoint = torch.load(path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.ft_weights = state_dict['ft.weight'].t().numpy()
        self.ft_bias = state_dict['ft.bias'].numpy()
        self.fc1_weights = state_dict['fc1.weight'].t().numpy()
        self.fc1_bias = state_dict['fc1.bias'].numpy()
        self.fc2_weights = state_dict['fc2.weight'].t().numpy()
        self.fc2_bias = state_dict['fc2.bias'].numpy()[0]

    def _load_numpy_weights(self, path: str):
        """Load from numpy file."""
        data = np.load(path)
        self.ft_weights = data['ft_weights']
        self.ft_bias = data['ft_bias']
        self.fc1_weights = data['fc1_weights']
        self.fc1_bias = data['fc1_bias']
        self.fc2_weights = data['fc2_weights']
        self.fc2_bias = data['fc2_bias']

    def save_weights(self, path: str):
        """Save weights to numpy file."""
        np.savez(
            path,
            ft_weights=self.ft_weights,
            ft_bias=self.ft_bias,
            fc1_weights=self.fc1_weights,
            fc1_bias=self.fc1_bias,
            fc2_weights=self.fc2_weights,
            fc2_bias=self.fc2_bias,
        )

    def reset_accumulators(self, board: chess.Board):
        """
        Reset accumulators from scratch for given position.
        Call this when position is loaded initially.
        """
        self.white_acc = self.ft_bias.copy()
        self.black_acc = self.ft_bias.copy()

        # Find kings
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        if white_king_sq is None or black_king_sq is None:
            return  # Invalid position

        # Add all pieces
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_idx = piece_to_index(piece)

                # White perspective
                feat_idx_w = get_feature_index(piece_idx, square, white_king_sq, False)
                self.white_acc += self.ft_weights[feat_idx_w]

                # Black perspective
                feat_idx_b = get_feature_index(piece_idx, square, black_king_sq, True)
                self.black_acc += self.ft_weights[feat_idx_b]

    def push(self, board: chess.Board, move: chess.Move):
        """
        Update accumulators incrementally for move.
        Call before board.push(move).
        """
        # Save current state for unmake
        self.acc_stack.append((self.white_acc.copy(), self.black_acc.copy()))

        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        # Remove piece from source square
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type != chess.KING:
            piece_idx = piece_to_index(piece)

            feat_idx_w = get_feature_index(piece_idx, move.from_square, white_king_sq, False)
            self.white_acc -= self.ft_weights[feat_idx_w]

            feat_idx_b = get_feature_index(piece_idx, move.from_square, black_king_sq, True)
            self.black_acc -= self.ft_weights[feat_idx_b]

        # Remove captured piece
        captured = board.piece_at(move.to_square)
        if captured and captured.piece_type != chess.KING:
            piece_idx = piece_to_index(captured)

            feat_idx_w = get_feature_index(piece_idx, move.to_square, white_king_sq, False)
            self.white_acc -= self.ft_weights[feat_idx_w]

            feat_idx_b = get_feature_index(piece_idx, move.to_square, black_king_sq, True)
            self.black_acc -= self.ft_weights[feat_idx_b]

        # Add piece to destination (handle promotion)
        moved_piece = piece
        if move.promotion:
            moved_piece = chess.Piece(move.promotion, piece.color)

        if moved_piece and moved_piece.piece_type != chess.KING:
            piece_idx = piece_to_index(moved_piece)

            feat_idx_w = get_feature_index(piece_idx, move.to_square, white_king_sq, False)
            self.white_acc += self.ft_weights[feat_idx_w]

            feat_idx_b = get_feature_index(piece_idx, move.to_square, black_king_sq, True)
            self.black_acc += self.ft_weights[feat_idx_b]

        # Handle en passant
        if board.is_en_passant(move):
            ep_square = move.to_square - 8 if piece.color else move.to_square + 8
            ep_piece = board.piece_at(ep_square)
            if ep_piece:
                piece_idx = piece_to_index(ep_piece)

                feat_idx_w = get_feature_index(piece_idx, ep_square, white_king_sq, False)
                self.white_acc -= self.ft_weights[feat_idx_w]

                feat_idx_b = get_feature_index(piece_idx, ep_square, black_king_sq, True)
                self.black_acc -= self.ft_weights[feat_idx_b]

        # Handle castling (rook movement)
        if board.is_castling(move):
            # Determine rook movement
            if move.to_square > move.from_square:  # Kingside
                rook_from = move.to_square + 1
                rook_to = move.to_square - 1
            else:  # Queenside
                rook_from = move.to_square - 2
                rook_to = move.to_square + 1

            rook = board.piece_at(rook_from)
            if rook:
                piece_idx = piece_to_index(rook)

                # Remove rook from source
                feat_idx_w = get_feature_index(piece_idx, rook_from, white_king_sq, False)
                self.white_acc -= self.ft_weights[feat_idx_w]

                feat_idx_b = get_feature_index(piece_idx, rook_from, black_king_sq, True)
                self.black_acc -= self.ft_weights[feat_idx_b]

                # Add rook to destination
                feat_idx_w = get_feature_index(piece_idx, rook_to, white_king_sq, False)
                self.white_acc += self.ft_weights[feat_idx_w]

                feat_idx_b = get_feature_index(piece_idx, rook_to, black_king_sq, True)
                self.black_acc += self.ft_weights[feat_idx_b]

    def pop(self):
        """Restore accumulators after board.pop()."""
        if self.acc_stack:
            self.white_acc, self.black_acc = self.acc_stack.pop()

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate position from side-to-move perspective.
        Returns centipawns.
        """
        # Apply ClippedReLU (clamp to [0, 1])
        white_ft = np.clip(self.white_acc, 0, 1)
        black_ft = np.clip(self.black_acc, 0, 1)

        # Concatenate with side to move first
        if board.turn:  # White to move
            combined = np.concatenate([white_ft, black_ft])
        else:  # Black to move
            combined = np.concatenate([black_ft, white_ft])

        # FC1 with ReLU
        with np.errstate(all='ignore'):  # Suppress numerical warnings for untrained weights
            hidden = np.maximum(0, combined @ self.fc1_weights + self.fc1_bias)
            hidden = np.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=0.0)

            # FC2 (output)
            output = (hidden @ self.fc2_weights).sum() + self.fc2_bias
            output = np.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)

        # Scale to centipawns (adjust multiplier based on training)
        # Clamp to reasonable range
        cp_score = np.clip(output * 100, -10000, 10000)
        return int(cp_score)
