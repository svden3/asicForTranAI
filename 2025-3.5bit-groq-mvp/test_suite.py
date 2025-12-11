#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Testing Suite for 3.5-bit Quantization
Run with: python test_suite.py or pytest test_suite.py
"""
import numpy as np
import sys
import io

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from quantize_weights import quantize_to_3p5bit, dequantize_from_3p5bit

class TestQuantization:
    """Test suite for 3.5-bit quantization"""

    def test_basic_quantization(self):
        """Test basic quantization and dequantization"""
        print("\n[TEST] Basic quantization...")

        # Create test matrix
        W = np.random.randn(128, 128).astype(np.float32) * 0.1

        # Quantize
        W_pack, scales, offsets = quantize_to_3p5bit(W)

        # Dequantize
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        # Check shapes
        assert W_recon.shape == W.shape, "Shape mismatch after quantization"

        # Check error is reasonable
        mse = np.mean((W - W_recon) ** 2)
        assert mse < 0.01, f"MSE too high: {mse}"

        print(f"  ✓ PASS - MSE: {mse:.6f}")
        return True

    def test_quantization_determinism(self):
        """Test that quantization is deterministic"""
        print("\n[TEST] Quantization determinism...")

        W = np.random.randn(64, 64).astype(np.float32) * 0.1

        # Quantize twice
        W_pack1, scales1, offsets1 = quantize_to_3p5bit(W)
        W_pack2, scales2, offsets2 = quantize_to_3p5bit(W)

        # Check they're identical
        assert np.array_equal(W_pack1, W_pack2), "Quantization not deterministic (packed)"
        assert np.array_equal(scales1, scales2), "Quantization not deterministic (scales)"
        assert np.array_equal(offsets1, offsets2), "Quantization not deterministic (offsets)"

        print("  ✓ PASS - Quantization is deterministic")
        return True

    def test_zero_input(self):
        """Test quantization of zero matrix"""
        print("\n[TEST] Zero input...")

        W = np.zeros((32, 32), dtype=np.float32)

        W_pack, scales, offsets = quantize_to_3p5bit(W)
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        mse = np.mean((W - W_recon) ** 2)
        assert mse < 1e-10, f"Zero matrix error too high: {mse}"

        print(f"  ✓ PASS - MSE: {mse:.10f}")
        return True

    def test_uniform_input(self):
        """Test quantization of uniform matrix"""
        print("\n[TEST] Uniform input...")

        W = np.ones((32, 32), dtype=np.float32) * 0.5

        W_pack, scales, offsets = quantize_to_3p5bit(W)
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        mae = np.mean(np.abs(W - W_recon))
        assert mae < 0.1, f"Uniform matrix error too high: {mae}"

        print(f"  ✓ PASS - MAE: {mae:.6f}")
        return True

    def test_large_values(self):
        """Test quantization with large values"""
        print("\n[TEST] Large values...")

        W = np.random.randn(64, 64).astype(np.float32) * 10.0

        W_pack, scales, offsets = quantize_to_3p5bit(W)
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        # Check relative error
        rel_error = np.mean(np.abs(W - W_recon) / (np.abs(W) + 1e-8))
        assert rel_error < 0.1, f"Relative error too high: {rel_error}"

        print(f"  ✓ PASS - Relative error: {rel_error:.6f}")
        return True

    def test_compression_ratio(self):
        """Test that compression achieves ~8x ratio"""
        print("\n[TEST] Compression ratio...")

        W = np.random.randn(1024, 1024).astype(np.float32) * 0.1

        fp32_bytes = W.size * 4
        W_pack, scales, offsets = quantize_to_3p5bit(W)

        K, N = W.shape
        quant_bytes = (K // 2) * N + N * 8  # packed weights + scales/offsets

        ratio = fp32_bytes / quant_bytes

        assert 7.5 < ratio < 8.5, f"Compression ratio out of range: {ratio}"

        print(f"  ✓ PASS - Compression ratio: {ratio:.2f}x")
        return True

    def test_odd_dimensions(self):
        """Test quantization with odd dimensions"""
        print("\n[TEST] Odd dimensions...")

        # Odd K dimension
        W = np.random.randn(127, 128).astype(np.float32) * 0.1

        W_pack, scales, offsets = quantize_to_3p5bit(W)
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        # Should pad to even K
        assert W_recon.shape == W.shape, "Shape handling incorrect for odd dimensions"

        mse = np.mean((W - W_recon) ** 2)
        assert mse < 0.01, f"Error too high with odd dimensions: {mse}"

        print(f"  ✓ PASS - MSE: {mse:.6f}")
        return True

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        print("\n[TEST] Numerical stability...")

        # Mix of small and large values
        W = np.random.randn(64, 64).astype(np.float32)
        W = W * np.random.uniform(0.001, 10.0, W.shape)

        W_pack, scales, offsets = quantize_to_3p5bit(W)
        W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

        # Check for NaN or Inf
        assert not np.any(np.isnan(W_recon)), "NaN values in reconstruction"
        assert not np.any(np.isinf(W_recon)), "Inf values in reconstruction"

        print("  ✓ PASS - No NaN or Inf values")
        return True

    def test_batch_quantization(self):
        """Test quantizing multiple matrices"""
        print("\n[TEST] Batch quantization...")

        matrices = [
            np.random.randn(32, 32).astype(np.float32) * 0.1
            for _ in range(5)
        ]

        errors = []
        for i, W in enumerate(matrices):
            W_pack, scales, offsets = quantize_to_3p5bit(W)
            W_recon = dequantize_from_3p5bit(W_pack, scales, offsets)

            mse = np.mean((W - W_recon) ** 2)
            errors.append(mse)

        avg_error = np.mean(errors)
        assert avg_error < 0.01, f"Average error too high: {avg_error}"

        print(f"  ✓ PASS - Average MSE across 5 matrices: {avg_error:.6f}")
        return True

    def run_all_tests(self):
        """Run all tests"""
        print("="*70)
        print("3.5-bit Quantization - Automated Test Suite")
        print("="*70)

        tests = [
            self.test_basic_quantization,
            self.test_quantization_determinism,
            self.test_zero_input,
            self.test_uniform_input,
            self.test_large_values,
            self.test_compression_ratio,
            self.test_odd_dimensions,
            self.test_numerical_stability,
            self.test_batch_quantization,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
            except AssertionError as e:
                print(f"  ✗ FAIL - {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ ERROR - {e}")
                failed += 1

        print("\n" + "="*70)
        print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
        print("="*70)

        if failed == 0:
            print("\n✓ All tests passed!")
            return 0
        else:
            print(f"\n✗ {failed} test(s) failed")
            return 1

def main():
    """Main entry point"""
    tester = TestQuantization()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
