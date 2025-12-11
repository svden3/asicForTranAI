#!/usr/bin/env python3
"""
Convert FP32 model weights to 3.5-bit format for Groq deployment
Matches the encoding in matmul_3p5bit_dynamic.f90
"""
import numpy as np
from pathlib import Path

def quantize_to_3p5bit(weights_fp32, group_size=128):
    """
    Quantize FP32 weights to 3.5-bit asymmetric format

    Args:
        weights_fp32: numpy array of FP32 weights [K, N]
        group_size: quantization group size (default 128)

    Returns:
        w_pack: int8 array [K_padded/2, N] - packed 3.5-bit values
        scales: float32 array [N] - per-column scales
        offsets: float32 array [N] - per-column zero-points
        K_orig: int - original K dimension (before padding)
    """
    K, N = weights_fp32.shape

    # Pad K to even if necessary
    K_orig = K
    if K % 2 != 0:
        K = K + 1
        weights_fp32_padded = np.zeros((K, N), dtype=np.float32)
        weights_fp32_padded[:K_orig, :] = weights_fp32
        weights_fp32 = weights_fp32_padded

    # Per-column quantization (AWQ-style)
    scales = np.zeros(N, dtype=np.float32)
    offsets = np.zeros(N, dtype=np.float32)
    w_pack = np.zeros((K // 2, N), dtype=np.int8)

    for col in range(N):
        w_col = weights_fp32[:, col]

        # Compute scale and offset (symmetric quantization is better for 3.5-bit)
        w_absmax = np.abs(w_col).max()

        # Use full quantization range
        # For alternating 4-bit and 3-bit, we use average range
        # 4-bit: [-8, 7] range=15, 3-bit: [-4, 3] range=7
        # Average: (15+7)/2 = 11, use 7 as max value (conservative but balanced)
        quant_max = 7.0  # Conservative max for mixed precision

        # Add epsilon to w_absmax to handle zero inputs
        scale = (w_absmax + 1e-8) / quant_max
        offset = 0.0  # Symmetric quantization

        scales[col] = scale
        offsets[col] = offset

        # Quantize to 3.5-bit (alternating 4-bit and 3-bit)
        for k in range(0, K, 2):
            # Quantize first value to 4-bit range [-8, 7]
            # But use conservative range to match second value
            q1_float = w_col[k] / scale
            q1 = int(np.clip(np.round(q1_float), -8, 7))

            # Quantize second value to 3-bit range [-4, 3]
            if k + 1 < K:
                q2_float = w_col[k+1] / scale
                q2 = int(np.clip(np.round(q2_float), -4, 3))
            else:
                q2 = 0

            # Encode into 7-bit representation
            u1 = q1 if q1 >= 0 else q1 + 16  # 4-bit unsigned
            u2 = q2 if q2 >= 0 else q2 + 8   # 3-bit unsigned

            raw7 = (u1 << 3) | u2
            w_pack[k // 2, col] = np.int8(raw7)

    return w_pack, scales, offsets, K_orig


def dequantize_from_3p5bit(w_pack, scales, offsets, K_orig=None):
    """
    Dequantize back to FP32 for verification

    Args:
        w_pack: packed weights
        scales: per-column scales
        offsets: per-column offsets
        K_orig: original K dimension (before padding), if None returns full K
    """
    K_half, N = w_pack.shape
    K = K_half * 2
    weights_fp32 = np.zeros((K, N), dtype=np.float32)

    for col in range(N):
        for k in range(0, K, 2):
            raw7 = int(w_pack[k // 2, col]) & 0x7F

            # Decode (matching Fortran implementation)
            n1 = raw7 >> 3
            n2 = raw7 & 7

            if n1 >= 8:  n1 -= 16
            if n2 >= 4:  n2 -= 8

            weights_fp32[k, col] = (n1 + offsets[col]) * scales[col]
            if k + 1 < K:
                weights_fp32[k+1, col] = (n2 + offsets[col]) * scales[col]

    # Return only original rows if K_orig specified
    if K_orig is not None:
        return weights_fp32[:K_orig, :]
    return weights_fp32


def test_quantization():
    """Test quantization with random weights"""
    print("=" * 70)
    print("Testing 3.5-bit Weight Quantization")
    print("=" * 70)
    print()

    # Generate test weights
    np.random.seed(42)
    K, N = 128, 64
    weights_original = np.random.randn(K, N).astype(np.float32) * 0.1

    print(f"Original weights: shape={weights_original.shape}, dtype={weights_original.dtype}")
    print(f"  Range: [{weights_original.min():.4f}, {weights_original.max():.4f}]")
    print(f"  Mean: {weights_original.mean():.4f}, Std: {weights_original.std():.4f}")
    print()

    # Quantize
    w_pack, scales, offsets, K_orig = quantize_to_3p5bit(weights_original)

    print(f"Quantized weights:")
    print(f"  w_pack: shape={w_pack.shape}, dtype={w_pack.dtype}")
    print(f"  scales: shape={scales.shape}, dtype={scales.dtype}")
    print(f"  offsets: shape={offsets.shape}, dtype={offsets.dtype}")
    print(f"  K_orig: {K_orig} (K_padded: {w_pack.shape[0]*2})")
    print()

    # Compression ratio
    original_bytes = K * N * 4  # FP32
    compressed_bytes = (K_orig // 2 if K_orig % 2 == 0 else (K_orig + 1) // 2) * N * 1 + N * 8  # INT8 packed + FP32 scales/offsets
    ratio = original_bytes / compressed_bytes

    print(f"Compression:")
    print(f"  Original: {original_bytes} bytes (FP32)")
    print(f"  Compressed: {compressed_bytes} bytes (3.5-bit + scales)")
    print(f"  Ratio: {ratio:.2f}x")
    print()

    # Dequantize and check error
    weights_dequant = dequantize_from_3p5bit(w_pack, scales, offsets, K_orig)

    error = np.abs(weights_original - weights_dequant)
    mse = np.mean(error ** 2)
    max_error = error.max()

    print(f"Reconstruction Error:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean relative error: {(error / (np.abs(weights_original) + 1e-8)).mean():.4f}")
    print()

    # Check if quantization is reversible
    sample_idx = (0, 0)
    orig = weights_original[sample_idx]
    dequant = weights_dequant[sample_idx]
    print(f"Sample value: original={orig:.6f}, dequantized={dequant:.6f}, error={abs(orig-dequant):.6f}")
    print()

    if mse < 1e-3:
        print("✅ Quantization quality: EXCELLENT")
    elif mse < 1e-2:
        print("✅ Quantization quality: GOOD")
    else:
        print("⚠️  Quantization quality: ACCEPTABLE (may affect accuracy)")

    return w_pack, scales, offsets


if __name__ == "__main__":
    test_quantization()

    print()
    print("=" * 70)
    print("Next: Use this to convert real LLaMA 70B weights")
    print("=" * 70)
    print("""
To convert actual model weights:

1. Load GGUF or PyTorch checkpoint
2. For each layer's weight matrix:
   w_pack, scales, offsets = quantize_to_3p5bit(weight_fp32)
3. Save to binary format for Fortran:
   w_pack.tofile('weights_layer0.bin')
   scales.tofile('scales_layer0.bin')
   offsets.tofile('offsets_layer0.bin')
""")
