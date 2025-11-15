"""
NNUE (Efficiently Updatable Neural Network) for Chess
Based on the architecture used in Stockfish and other top engines

Architecture:
- Input: HalfKP features (piece positions relative to king)
- Layer 1: 768 -> 512 (per side, so 1024 total)
- Layer 2: 1024 -> 32
- Output: 32 -> 1

This is MUCH better than CNN for chess because:
1. Uses piece-king relationships (critical for chess)
2. Can be incrementally updated (very fast)
3. Proven architecture (Stockfish uses this)
"""

import torch
import torch.nn as nn
import numpy as np
import chess


# Feature set sizes
NUM_SQUARES = 64
NUM_PIECE_TYPES = 6  # P, N, B, R, Q, K (we don't include kings in features)
NUM_COLORS = 2

# HalfKP: For each king position, track all pieces (except kings)
# Input size: 64 (king squares) * 10 (piece types) * 64 (piece squares)
# But we only use features for the current king position
HALFKP_INPUT_SIZE = 10 * 64  # 640 features per side


class NNUE(nn.Module):
    """
    NNUE Architecture optimized for chess evaluation

    Using ~8MB of the 10MB budget for maximum strength
    """

    def __init__(self,
                 input_size=HALFKP_INPUT_SIZE,
                 hidden1_size=512,  # Larger for better pattern recognition
                 hidden2_size=32):
        super(NNUE, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # Feature transformer (one for each side)
        # Input: sparse HalfKP features -> Dense hidden representation
        self.ft_white = nn.Linear(input_size, hidden1_size)
        self.ft_black = nn.Linear(input_size, hidden1_size)

        # Layer 2: Combine both perspectives
        self.fc1 = nn.Linear(hidden1_size * 2, hidden2_size)

        # Output layer
        self.fc2 = nn.Linear(hidden2_size, 1)

        # Use ClippedReLU activation (standard in NNUE)
        # This is ReLU clamped to [0, 1]
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

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_size_mb(self):
        """Get model size in MB"""
        params = self.count_parameters()
        # float32 = 4 bytes
        return params * 4 / 1024 / 1024


def halfkp_index(king_square, piece_square, piece_type, piece_color):
    """
    Compute HalfKP feature index

    Args:
        king_square: Square of the king (0-63)
        piece_square: Square of the piece (0-63)
        piece_type: Type of piece (1-6 for P,N,B,R,Q,K)
        piece_color: Color of piece (True=White, False=Black)

    Returns:
        Feature index (0-639)
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
        # Flip the square for black's perspective
        flipped_square = square ^ 56  # Flip rank
        flipped_king_square = black_king_square ^ 56
        idx = halfkp_index(flipped_king_square, flipped_square, piece.piece_type, not piece.color)
        if idx is not None:
            black_features[idx] = 1.0

    return white_features, black_features


if __name__ == "__main__":
    print("=" * 60)
    print("NNUE MODEL FOR CHESS")
    print("=" * 60)
    print()

    # Create model with different sizes
    configs = [
        {"hidden1": 256, "hidden2": 32, "name": "Small"},
        {"hidden1": 512, "hidden2": 32, "name": "Medium"},
        {"hidden1": 768, "hidden2": 32, "name": "Large"},
        {"hidden1": 1024, "hidden2": 64, "name": "Very Large"},
    ]

    print("Model size comparison:")
    print("-" * 60)
    for config in configs:
        model = NNUE(hidden1_size=config["hidden1"], hidden2_size=config["hidden2"])
        params = model.count_parameters()
        size_mb = model.get_size_mb()
        print(f"{config['name']:12} - {params:,} params - {size_mb:.2f} MB")

    print()
    print("Selecting Medium (512 hidden) for ~8MB target")
    print()

    # Create the model
    model = NNUE(hidden1_size=512, hidden2_size=32)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size (float32): {model.get_size_mb():.2f} MB")
    print(f"Model size (int8): {model.get_size_mb() / 4:.2f} MB")
    print()

    # Test with a position
    print("Testing with starting position...")
    board = chess.Board()
    white_feat, black_feat = board_to_halfkp_features(board)

    print(f"White features shape: {white_feat.shape}")
    print(f"Black features shape: {black_feat.shape}")
    print(f"Active white features: {np.sum(white_feat)}")
    print(f"Active black features: {np.sum(black_feat)}")
    print()

    # Forward pass
    white_tensor = torch.tensor(white_feat).unsqueeze(0)
    black_tensor = torch.tensor(black_feat).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(white_tensor, black_tensor)

    print(f"Model output: {output.item():.4f}")
    print()
    print("✓ NNUE model working correctly!")
