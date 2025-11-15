"""
Compact Chess Value Network - Evaluates board positions
This is a VALUE network (not policy) - outputs a single score indicating who's winning
Combined with alpha-beta search to find the best moves
"""

import torch
import torch.nn as nn


class CompactChessNet(nn.Module):
    """
    Ultra-compact neural network for chess position evaluation.

    Architecture:
    - Input: 13x8x8 (12 piece planes + 1 legal moves plane)
    - Conv1: 13 → 32 channels (3x3 kernel)
    - Conv2: 32 → 64 channels (3x3 kernel)
    - FC1: 64*8*8 → 128
    - FC2: 128 → 1 (value head)

    Total params: ~680K → ~2.7MB (float32) → ~680KB (int8 quantized)
    """

    def __init__(self):
        super(CompactChessNet, self).__init__()

        # Convolutional layers - extract spatial patterns
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)  # Keep 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Keep 8x8

        # Batch normalization for better training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers - combine features into evaluation
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)  # Single output: position evaluation

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Output in [-1, 1] range

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Board state tensor (batch_size, 13, 8, 8)
        Returns:
            Position evaluation in [-1, 1] range
            -1 = Black winning, 0 = Draw, +1 = White winning
        """
        # Convolutional layers with batch norm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # Squash to [-1, 1]

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = CompactChessNet()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randn(1, 13, 8, 8)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")

    # Calculate model size
    param_size = model.count_parameters() * 4  # float32 = 4 bytes
    print(f"\nModel size (float32): {param_size / 1024 / 1024:.2f} MB")
    print(f"Model size (int8 quantized): {param_size / 1024 / 1024 / 4:.2f} MB")
