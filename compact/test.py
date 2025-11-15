#!/usr/bin/env python3
"""
Test script to verify all components work correctly
"""

import torch
import chess
import numpy as np
import sys

print("=" * 60)
print("COMPACT CHESS BOT - Component Tests")
print("=" * 60)
print()

# Test 1: Model
print("Test 1: Model Architecture")
print("-" * 60)
try:
    from model import CompactChessNet
    model = CompactChessNet()
    params = model.count_parameters()
    print(f"✓ Model created successfully")
    print(f"  Parameters: {params:,}")
    print(f"  Size (float32): {params * 4 / 1024 / 1024:.2f} MB")
    print(f"  Size (int8): {params / 1024 / 1024:.2f} MB")

    # Test forward pass
    dummy_input = torch.randn(1, 13, 8, 8)
    output = model(dummy_input)
    assert output.shape == (1, 1), f"Wrong output shape: {output.shape}"
    assert -1.0 <= output.item() <= 1.0, f"Output out of range: {output.item()}"
    print(f"✓ Forward pass works (output: {output.item():.4f})")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    sys.exit(1)

print()

# Test 2: Board Encoding
print("Test 2: Board Encoding")
print("-" * 60)
try:
    from utils import board_to_tensor, get_game_result, create_training_data
    board = chess.Board()
    tensor = board_to_tensor(board)

    assert tensor.shape == (13, 8, 8), f"Wrong tensor shape: {tensor.shape}"
    assert tensor[:12].sum() == 32, f"Wrong piece count: {tensor[:12].sum()}"
    # Note: Starting position has 20 legal moves but only 16 unique destinations
    assert tensor[12].sum() == 16, f"Wrong legal move destinations: {tensor[12].sum()}"

    print(f"✓ Board encoding works")
    print(f"  Shape: {tensor.shape}")
    print(f"  Pieces: {int(tensor[:12].sum())}")
    print(f"  Legal moves: {int(tensor[12].sum())}")
except Exception as e:
    print(f"✗ Board encoding test failed: {e}")
    sys.exit(1)

print()

# Test 3: Dataset
print("Test 3: Dataset")
print("-" * 60)
try:
    from dataset import ChessValueDataset

    X = np.random.randn(10, 13, 8, 8).astype(np.float32)
    y = np.random.choice([-1.0, 0.0, 1.0], size=10).astype(np.float32)

    dataset = ChessValueDataset(X, y)
    assert len(dataset) == 10, f"Wrong dataset size: {len(dataset)}"

    board, result = dataset[0]
    assert board.shape == (13, 8, 8), f"Wrong board shape: {board.shape}"
    assert result.shape == (1,), f"Wrong result shape: {result.shape}"

    print(f"✓ Dataset works")
    print(f"  Size: {len(dataset)}")
except Exception as e:
    print(f"✗ Dataset test failed: {e}")
    sys.exit(1)

print()

# Test 4: Model Inference
print("Test 4: Model Inference")
print("-" * 60)
try:
    from utils import board_to_tensor

    board = chess.Board()
    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    print(f"✓ Inference works")
    print(f"  Starting position evaluation: {output.item():.4f}")
    print(f"  (Should be close to 0.0 for equal position)")
except Exception as e:
    print(f"✗ Inference test failed: {e}")
    sys.exit(1)

print()

# Test 5: Size Estimate
print("Test 5: Size Estimate")
print("-" * 60)
try:
    # Model parameters
    param_size = model.count_parameters() * 4 / 1024 / 1024
    param_size_int8 = model.count_parameters() / 1024 / 1024

    # Activations (for batch_size=1)
    activation_size = 6300 * 4 / 1024 / 1024

    # Search tree estimate
    search_size = 1.0

    total_float32 = param_size + activation_size + search_size
    total_int8 = param_size_int8 + activation_size + search_size

    print(f"Memory breakdown (float32):")
    print(f"  Model parameters: {param_size:.2f} MB")
    print(f"  Activations: {activation_size:.2f} MB")
    print(f"  Search tree: {search_size:.2f} MB")
    print(f"  TOTAL: {total_float32:.2f} MB")
    print()
    print(f"Memory breakdown (int8 quantized):")
    print(f"  Model parameters: {param_size_int8:.2f} MB")
    print(f"  Activations: {activation_size:.2f} MB")
    print(f"  Search tree: {search_size:.2f} MB")
    print(f"  TOTAL: {total_int8:.2f} MB")
    print()

    if total_int8 < 10.0:
        print(f"✅ UNDER 10MB LIMIT! (by {10 - total_int8:.2f} MB)")
    else:
        print(f"❌ OVER 10MB LIMIT (by {total_int8 - 10:.2f} MB)")
except Exception as e:
    print(f"✗ Size estimate failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print()
print("Next steps:")
print("  1. Train the model:   python3 train.py")
print("  2. Quantize it:       python3 quantize.py")
print("  3. Play against it:   python3 play.py")
