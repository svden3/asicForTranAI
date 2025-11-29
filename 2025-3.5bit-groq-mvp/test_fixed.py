#!/usr/bin/env python3
"""
Verify that the FIXED version produces correct quantization
"""
import numpy as np

def decode_3p5bit_fixed(raw7):
    """Fixed version matching matmul_3p5bit_dynamic.f90 (lines 51-56)"""
    n1 = raw7 >> 3              # ishft(raw7, -3)
    n2 = raw7 & 7               # iand(raw7, 7)

    if n1 >= 8:  n1 -= 16       # 4-bit sign extension
    if n2 >= 4:  n2 -= 8        # 3-bit sign extension

    return n1, n2

def encode_3p5bit(n1, n2):
    """Encode two values into 7-bit representation"""
    # Ensure valid ranges
    assert -8 <= n1 <= 7, f"n1 out of range: {n1}"
    assert -4 <= n2 <= 3, f"n2 out of range: {n2}"

    # Convert to unsigned
    u1 = n1 if n1 >= 0 else n1 + 16
    u2 = n2 if n2 >= 0 else n2 + 8

    # Pack into 7 bits
    raw7 = (u1 << 3) | u2
    return raw7

print("=" * 70)
print("VERIFICATION: Fixed 3.5-bit Quantization")
print("=" * 70)
print()

# Test round-trip encoding/decoding
print("Testing round-trip encode → decode:")
print("(n1, n2) | raw7 | decoded (n1, n2) | Match?")
print("-" * 60)

test_cases = [
    (0, 0),
    (7, 3),      # Max positive
    (-8, -4),    # Min negative
    (3, -2),     # Mixed
    (-5, 1),     # Mixed
]

all_pass = True
for n1_orig, n2_orig in test_cases:
    raw7 = encode_3p5bit(n1_orig, n2_orig)
    n1_dec, n2_dec = decode_3p5bit_fixed(raw7)
    match = (n1_orig == n1_dec) and (n2_orig == n2_dec)
    all_pass = all_pass and match

    status = "✓" if match else "✗"
    print(f"({n1_orig:3d},{n2_orig:3d}) | {raw7:4d} | ({n1_dec:3d},{n2_dec:3d})       | {status}")

print()
if all_pass:
    print("✅ ALL TESTS PASSED! Quantization is correct.")
else:
    print("❌ SOME TESTS FAILED! Check implementation.")

print()
print("=" * 70)
print("Value Range Analysis:")
print("=" * 70)

# Decode all possible 7-bit values
n1_values = set()
n2_values = set()

for raw7 in range(128):
    n1, n2 = decode_3p5bit_fixed(raw7)
    n1_values.add(n1)
    n2_values.add(n2)

print(f"n1 range: [{min(n1_values)}, {max(n1_values)}] - {len(n1_values)} unique values")
print(f"n2 range: [{min(n2_values)}, {max(n2_values)}] - {len(n2_values)} unique values")
print(f"Total unique combinations: {len(n1_values)} × {len(n2_values)} = {len(n1_values) * len(n2_values)}")
print()
print("✅ Expected: n1 ∈ [-8, 7] (16 values), n2 ∈ [-4, 3] (8 values)")

print()
print("=" * 70)
print("Next Steps:")
print("=" * 70)
print("""
1. ✅ Quantization logic is CORRECT
2. 🔧 Create quantizer to convert FP32 weights to 3.5-bit format
3. 🧪 Test on small matmul (4×4 matrix) to verify numerical accuracy
4. 🚀 Deploy to Groq and measure token/s
""")
