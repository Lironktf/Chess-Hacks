"""
PyTorch dataset for chess position evaluation
"""

import torch
from torch.utils.data import Dataset


class ChessValueDataset(Dataset):
    """
    Dataset for training chess value network

    Each sample is a (board_state, result) pair where:
    - board_state: 13x8x8 tensor representing the position
    - result: scalar in [-1, 1] representing the game outcome
    """

    def __init__(self, board_states, results):
        """
        Args:
            board_states: numpy array of shape (N, 13, 8, 8)
            results: numpy array of shape (N,) with values in [-1, 1]
        """
        self.board_states = torch.tensor(board_states, dtype=torch.float32)
        self.results = torch.tensor(results, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.board_states)

    def __getitem__(self, idx):
        return self.board_states[idx], self.results[idx]


if __name__ == "__main__":
    # Test the dataset
    import numpy as np

    # Create dummy data
    X = np.random.randn(100, 13, 8, 8).astype(np.float32)
    y = np.random.choice([-1.0, 0.0, 1.0], size=100).astype(np.float32)

    dataset = ChessValueDataset(X, y)
    print(f"Dataset size: {len(dataset)}")

    # Test getting an item
    board, result = dataset[0]
    print(f"Board shape: {board.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result value: {result.item()}")
