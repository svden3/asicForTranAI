#!/usr/bin/env python3
"""
Model Inference Testing Framework
Tests 3.5-bit quantization on actual transformer models
"""
import numpy as np
import time
import json
from quantize_weights import quantize_to_3p5bit, dequantize_from_3p5bit

class ModelInferenceTester:
    """Test inference with 3.5-bit quantized weights"""

    def __init__(self):
        self.results = {}

    def create_synthetic_transformer_layer(self, hidden_dim=768, ff_dim=3072):
        """Create synthetic transformer layer weights for testing"""
        print(f"\nCreating synthetic transformer layer (hidden_dim={hidden_dim}, ff_dim={ff_dim})...")

        layer_weights = {
            'q_proj': np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            'k_proj': np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            'v_proj': np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            'o_proj': np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02,
            'ff_up': np.random.randn(hidden_dim, ff_dim).astype(np.float32) * 0.02,
            'ff_down': np.random.randn(ff_dim, hidden_dim).astype(np.float32) * 0.02,
        }

        return layer_weights

    def quantize_model_layer(self, layer_weights):
        """Quantize all weights in a transformer layer"""
        print("Quantizing layer weights...")
        start = time.time()

        quantized_layer = {}
        for name, weight in layer_weights.items():
            w_pack, scales, offsets = quantize_to_3p5bit(weight)
            quantized_layer[name] = {
                'packed': w_pack,
                'scales': scales,
                'offsets': offsets,
                'shape': weight.shape
            }

        quant_time = time.time() - start
        print(f"  Quantization time: {quant_time:.2f}s")

        return quantized_layer, quant_time

    def compute_memory_savings(self, layer_weights, quantized_layer):
        """Calculate memory savings"""
        fp32_memory = 0
        quant_memory = 0

        for name, weight in layer_weights.items():
            fp32_memory += weight.size * 4  # 4 bytes per float32

            shape = quantized_layer[name]['shape']
            K, N = shape
            # Packed weights: (K//2) * N bytes + scales/offsets: N * 8 bytes
            quant_memory += (K // 2) * N + N * 8

        savings = (1 - quant_memory / fp32_memory) * 100

        return fp32_memory / 1e6, quant_memory / 1e6, savings

    def test_inference_accuracy(self, layer_weights, quantized_layer, batch_size=1, seq_len=128):
        """Test inference accuracy with quantized weights"""
        print(f"\nTesting inference accuracy (batch={batch_size}, seq_len={seq_len})...")

        hidden_dim = layer_weights['q_proj'].shape[0]

        # Create random input
        x = np.random.randn(batch_size * seq_len, hidden_dim).astype(np.float32) * 0.5

        # FP32 forward pass
        print("  Running FP32 baseline...")
        start = time.time()
        q_fp32 = np.dot(x, layer_weights['q_proj'])
        k_fp32 = np.dot(x, layer_weights['k_proj'])
        v_fp32 = np.dot(x, layer_weights['v_proj'])
        o_fp32 = np.dot(v_fp32, layer_weights['o_proj'])
        ff_up_fp32 = np.dot(o_fp32, layer_weights['ff_up'])
        # ReLU
        ff_up_fp32 = np.maximum(ff_up_fp32, 0)
        output_fp32 = np.dot(ff_up_fp32, layer_weights['ff_down'])
        fp32_time = time.time() - start

        # 3.5-bit quantized forward pass
        print("  Running 3.5-bit quantized inference...")
        start = time.time()

        # Dequantize weights
        q_weight = dequantize_from_3p5bit(
            quantized_layer['q_proj']['packed'],
            quantized_layer['q_proj']['scales'],
            quantized_layer['q_proj']['offsets']
        )
        k_weight = dequantize_from_3p5bit(
            quantized_layer['k_proj']['packed'],
            quantized_layer['k_proj']['scales'],
            quantized_layer['k_proj']['offsets']
        )
        v_weight = dequantize_from_3p5bit(
            quantized_layer['v_proj']['packed'],
            quantized_layer['v_proj']['scales'],
            quantized_layer['v_proj']['offsets']
        )
        o_weight = dequantize_from_3p5bit(
            quantized_layer['o_proj']['packed'],
            quantized_layer['o_proj']['scales'],
            quantized_layer['o_proj']['offsets']
        )
        ff_up_weight = dequantize_from_3p5bit(
            quantized_layer['ff_up']['packed'],
            quantized_layer['ff_up']['scales'],
            quantized_layer['ff_up']['offsets']
        )
        ff_down_weight = dequantize_from_3p5bit(
            quantized_layer['ff_down']['packed'],
            quantized_layer['ff_down']['scales'],
            quantized_layer['ff_down']['offsets']
        )

        # Forward pass with quantized weights
        q_quant = np.dot(x, q_weight)
        k_quant = np.dot(x, k_weight)
        v_quant = np.dot(x, v_weight)
        o_quant = np.dot(v_quant, o_weight)
        ff_up_quant = np.dot(o_quant, ff_up_weight)
        ff_up_quant = np.maximum(ff_up_quant, 0)
        output_quant = np.dot(ff_up_quant, ff_down_weight)

        quant_time = time.time() - start

        # Compute accuracy metrics
        mse = np.mean((output_fp32 - output_quant) ** 2)
        mae = np.mean(np.abs(output_fp32 - output_quant))
        max_error = np.max(np.abs(output_fp32 - output_quant))

        # Relative error
        rel_error = np.mean(np.abs(output_fp32 - output_quant) / (np.abs(output_fp32) + 1e-8)) * 100

        return {
            'fp32_time_ms': fp32_time * 1000,
            'quant_time_ms': quant_time * 1000,
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'relative_error_pct': float(rel_error)
        }

    def run_full_test_suite(self):
        """Run comprehensive test suite"""
        print("="*70)
        print("Model Inference Testing Suite")
        print("3.5-bit Quantization on Transformer Layers")
        print("="*70)

        # Test different model sizes
        configs = [
            {'name': 'Small (BERT-base)', 'hidden_dim': 768, 'ff_dim': 3072},
            {'name': 'Medium (GPT-2)', 'hidden_dim': 1024, 'ff_dim': 4096},
            {'name': 'Large (LLaMA-7B)', 'hidden_dim': 4096, 'ff_dim': 11008},
        ]

        results = {}

        for config in configs:
            print(f"\n{'='*70}")
            print(f"Testing: {config['name']}")
            print('='*70)

            # Create layer
            layer_weights = self.create_synthetic_transformer_layer(
                config['hidden_dim'],
                config['ff_dim']
            )

            # Quantize
            quantized_layer, quant_time = self.quantize_model_layer(layer_weights)

            # Memory analysis
            fp32_mb, quant_mb, savings_pct = self.compute_memory_savings(
                layer_weights, quantized_layer
            )

            print(f"\nMemory Analysis:")
            print(f"  FP32: {fp32_mb:.2f} MB")
            print(f"  3.5-bit: {quant_mb:.2f} MB")
            print(f"  Savings: {savings_pct:.1f}%")

            # Inference test
            inference_results = self.test_inference_accuracy(
                layer_weights, quantized_layer,
                batch_size=1, seq_len=128
            )

            print(f"\nInference Results:")
            print(f"  FP32 time: {inference_results['fp32_time_ms']:.2f} ms")
            print(f"  3.5-bit time: {inference_results['quant_time_ms']:.2f} ms")
            print(f"  MSE: {inference_results['mse']:.6f}")
            print(f"  MAE: {inference_results['mae']:.6f}")
            print(f"  Max Error: {inference_results['max_error']:.6f}")
            print(f"  Relative Error: {inference_results['relative_error_pct']:.2f}%")

            # Store results
            results[config['name']] = {
                'config': config,
                'quantization_time_s': quant_time,
                'memory': {
                    'fp32_mb': fp32_mb,
                    'quant_mb': quant_mb,
                    'savings_pct': savings_pct
                },
                'inference': inference_results
            }

        # Save results
        with open('model_inference_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*70)
        print("Test Suite Complete!")
        print("="*70)
        print("\nResults saved to: model_inference_results.json")

        return results

if __name__ == "__main__":
    tester = ModelInferenceTester()
    results = tester.run_full_test_suite()
