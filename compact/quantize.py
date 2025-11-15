"""
Model quantization script - Compresses the model to INT8 for size optimization
"""

import os
import torch
import torch.nn as nn
from model import CompactChessNet


def quantize_model(input_path: str, output_path: str):
    """
    Quantize a model to INT8 to reduce size by ~4x

    Args:
        input_path: Path to the float32 model
        output_path: Path to save the quantized model
    """
    print(f"Loading model from: {input_path}")

    # Load the model
    model = CompactChessNet()
    model.load_state_dict(torch.load(input_path, map_location='cpu'))
    model.eval()

    print(f"Original model parameters: {model.count_parameters():,}")

    # Get original size
    original_size = os.path.getsize(input_path) / 1024 / 1024
    print(f"Original model size: {original_size:.2f} MB")

    # Quantize the model
    print("\nQuantizing model to INT8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Layers to quantize
        dtype=torch.qint8
    )

    # Save quantized model
    torch.save(quantized_model.state_dict(), output_path)
    print(f"✓ Saved quantized model to: {output_path}")

    # Get quantized size
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Quantized model size: {quantized_size:.2f} MB")

    compression_ratio = original_size / quantized_size
    print(f"Compression ratio: {compression_ratio:.2f}x")

    return quantized_size


def check_total_size():
    """
    Check total RAM+VRAM usage of the chess bot
    """
    print("\n" + "=" * 60)
    print("SIZE ANALYSIS")
    print("=" * 60)

    total_size = 0

    # Check model file
    model_files = ["best_model.pth", "final_model.pth", "model_int8.pth"]
    model_file = None

    for f in model_files:
        if os.path.exists(f):
            model_file = f
            break

    if model_file:
        model_size = os.path.getsize(model_file) / 1024 / 1024
        print(f"\nModel file: {model_file}")
        print(f"  Size: {model_size:.2f} MB")
        total_size += model_size
    else:
        print("\nWarning: No model file found!")

    # Estimate runtime memory usage
    print("\nEstimated runtime memory:")

    # Model parameters in memory
    model = CompactChessNet()
    param_count = model.count_parameters()
    param_memory = param_count * 4 / 1024 / 1024  # float32
    print(f"  Model parameters: {param_memory:.2f} MB")

    # Intermediate activations (batch_size=1)
    # Conv1 output: 32 * 8 * 8 = 2048 floats
    # Conv2 output: 64 * 8 * 8 = 4096 floats
    # FC1 output: 128 floats
    # Total: ~6300 floats
    activation_memory = 6300 * 4 / 1024 / 1024
    print(f"  Activations (inference): {activation_memory:.2f} MB")

    # Board state and search tree (estimate)
    search_memory = 1.0  # Conservative estimate
    print(f"  Search tree: {search_memory:.2f} MB")

    runtime_memory = param_memory + activation_memory + search_memory
    print(f"\nTotal runtime memory: {runtime_memory:.2f} MB")

    # Total (disk + runtime)
    total = model_size + runtime_memory if model_file else runtime_memory
    print(f"\n{'=' * 60}")
    print(f"TOTAL SIZE (RAM + VRAM): {total:.2f} MB")
    print(f"{'=' * 60}")

    if total < 10.0:
        print(f"✅ SUCCESS! Under 10MB limit (with {10 - total:.2f} MB to spare)")
    else:
        print(f"❌ OVER LIMIT by {total - 10:.2f} MB")

    return total


if __name__ == "__main__":
    import sys

    # Check for model files
    if os.path.exists("best_model.pth"):
        quantize_model("best_model.pth", "model_int8.pth")
    elif os.path.exists("final_model.pth"):
        quantize_model("final_model.pth", "model_int8.pth")
    else:
        print("No model found! Train a model first with: python train.py")
        sys.exit(1)

    # Check total size
    total_size = check_total_size()

    print("\n" + "=" * 60)
    if total_size < 10.0:
        print("✅ Your chess bot is ready to compete!")
        print("   Run 'python engine.py' to play against it.")
    else:
        print("⚠️  Model needs further compression.")
        print("   Consider reducing network size or using more aggressive quantization.")
    print("=" * 60)
