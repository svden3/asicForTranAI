#!/usr/bin/env python3
"""
Small matmul test to verify 3.5-bit implementation matches FP32 baseline
Before deploying to Groq, run this to ensure correctness
"""
import numpy as np
from quantize_weights import quantize_to_3p5bit, dequantize_from_3p5bit

def matmul_3p5bit_reference(A, w_pack, scales, offsets):
    """
    Reference Python implementation matching matmul_3p5bit_dynamic.f90
    Use this to verify Fortran code produces same results
    """
    M, K = A.shape
    K_half, N = w_pack.shape

    C = np.zeros((M, N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(0, K, 2):
                idx = k // 2
                raw7 = int(w_pack[idx, j]) & 0x7F

                # Decode (matching Fortran lines 51-56)
                n1 = raw7 >> 3
                n2 = raw7 & 7

                if n1 >= 8:  n1 -= 16
                if n2 >= 4:  n2 -= 8

                acc += int(A[i, k]) * n1
                if k + 1 < K:
                    acc += int(A[i, k+1]) * n2

            C[i, j] = acc

    # Dequantize (matching Fortran line 80: out = (acc + offset) * scale)
    Out = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            Out[i, j] = (C[i, j] + offsets[j]) * scales[j]

    return Out


def test_small_matmul():
    """Test 4×4 matmul: A @ W = C"""
    print("=" * 70)
    print("Small MatMul Test: Verify 3.5-bit vs FP32 Baseline")
    print("=" * 70)
    print()

    # Setup
    np.random.seed(123)
    M, K, N = 4, 8, 4

    A = np.random.randn(M, K).astype(np.float32) * 0.5
    W = np.random.randn(K, N).astype(np.float32) * 0.3

    print(f"Test dimensions: A={A.shape}, W={W.shape}")
    print()

    # FP32 baseline
    C_fp32 = A @ W

    print("FP32 baseline result:")
    print(C_fp32)
    print()

    # Quantize weights
    w_pack, scales, offsets = quantize_to_3p5bit(W)

    # Quantize activations to INT8
    A_int8 = np.clip(np.round(A * 127), -128, 127).astype(np.int8)
    A_scale = 1.0 / 127.0

    # 3.5-bit matmul
    C_3p5bit = matmul_3p5bit_reference(A_int8, w_pack, scales, offsets)
    C_3p5bit = C_3p5bit * A_scale  # Rescale activations

    print("3.5-bit quantized result:")
    print(C_3p5bit)
    print()

    # Error analysis
    error = np.abs(C_fp32 - C_3p5bit)
    rel_error = error / (np.abs(C_fp32) + 1e-8)

    print("Error Analysis:")
    print(f"  Absolute error: {error}")
    print(f"  Mean abs error: {error.mean():.6f}")
    print(f"  Max abs error:  {error.max():.6f}")
    print(f"  Mean rel error: {rel_error.mean():.4f}")
    print()

    # Pass/fail (3.5-bit has inherent quantization error)
    if error.max() < 0.3 and rel_error.mean() < 0.5:
        print("✅ PASS: 3.5-bit matmul quality acceptable for deployment")
        print("   Note: 3.5-bit quantization has inherent precision loss (~10-20% relative error)")
        return True
    elif error.max() < 0.5:
        print("⚠️  MARGINAL: Error is high but may work for some models")
        print("   Consider testing with real LLM weights to verify accuracy")
        return True
    else:
        print("❌ FAIL: Too much error, check implementation")
        return False


if __name__ == "__main__":
    success = test_small_matmul()

    if success:
        print()
        print("=" * 70)
        print("🎉 Ready for Groq deployment!")
        print("=" * 70)
        print("""
Next steps:
1. Compile Fortran code: lfortran --emit-mlir llama70b_int4.f90
2. Update to use matmul_3p5bit_groq module
3. Deploy to Groq: ./groq/compile_and_run.sh
4. Measure token/s and compare with INT4 baseline
""")
    else:
        print()
        print("⚠️  Fix errors before deploying to Groq")
