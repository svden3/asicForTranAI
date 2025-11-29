#!/usr/bin/env python3
"""
Verify the bug in the original 3.5-bit quantization implementation
"""

def original_buggy_version(raw7):
    """Original implementation from matmul_3p5bit_dynamic.f90"""
    n1 = raw7 >> 4              # ishft(raw7, -4)
    n2 = raw7 & 15              # iand(raw7, 15)

    # Original sign extension
    if n1 >= 8:
        n1 = n1 - 16           # ❌ This NEVER happens!
    if n2 >= 8:
        n2 = n2 - 16

    return n1, n2

def fixed_4plus3_version(raw7):
    """Fixed: 4 high bits + 3 low bits"""
    n1 = raw7 >> 3              # High 4 bits
    n2 = raw7 & 7               # Low 3 bits

    # Correct sign extension
    if n1 >= 8:
        n1 = n1 - 16
    if n2 >= 4:
        n2 = n2 - 8

    return n1, n2

print("=" * 70)
print("BUG ANALYSIS: Original 3.5-bit Quantization")
print("=" * 70)
print()

# Test all 7-bit values
n1_values_original = set()
n2_values_original = set()
n1_values_fixed = set()
n2_values_fixed = set()

print("Sample values (original vs. fixed):")
print("Raw7 | Original (n1, n2) | Fixed 4+3 (n1, n2)")
print("-" * 50)

for raw7 in range(0, 128, 8):
    n1_orig, n2_orig = original_buggy_version(raw7)
    n1_fix, n2_fix = fixed_4plus3_version(raw7)

    n1_values_original.add(n1_orig)
    n2_values_original.add(n2_orig)
    n1_values_fixed.add(n1_fix)
    n2_values_fixed.add(n2_fix)

    print(f"{raw7:4d} | ({n1_orig:3d}, {n2_orig:3d})        | ({n1_fix:3d}, {n2_fix:3d})")

print()
print("=" * 70)
print("RESULTS:")
print("=" * 70)

print("\n🐛 ORIGINAL (BUGGY) VERSION:")
print(f"   n1 range: {min(n1_values_original)} to {max(n1_values_original)}")
print(f"   n2 range: {min(n2_values_original)} to {max(n2_values_original)}")
print(f"   → n1 uses only {len(n1_values_original)} values (should be 16 for 4-bit signed)")
print(f"   → BUG: 'if (n1 >= 8)' condition NEVER executes!")

print("\n✅ FIXED VERSION (4+3 asymmetric):")
print(f"   n1 range: {min(n1_values_fixed)} to {max(n1_values_fixed)}")
print(f"   n2 range: {min(n2_values_fixed)} to {max(n2_values_fixed)}")
print(f"   → n1 uses {len(n1_values_fixed)} values (4-bit signed: -8 to +7)")
print(f"   → n2 uses {len(n2_values_fixed)} values (3-bit signed: -4 to +3)")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("""
1. FIX THE BUG: Change line 14-17 in matmul_3p5bit_dynamic.f90:

   OLD (BUGGY):
   n1 = ishft(raw7, -4)          ! Gets 0-7 only
   n2 = iand(raw7, 15)           ! Gets 0-15 ✓
   if (n1 >= 8)  n1 = n1 - 16    ! ❌ NEVER executes!
   if (n2 >= 8)  n2 = n2 - 16    ! ✓ Correct

   NEW (FIXED):
   n1 = ishft(raw7, -3)          ! Gets 0-15 (4 bits)
   n2 = iand(raw7, 7)            ! Gets 0-7  (3 bits)
   if (n1 >= 8)  n1 = n1 - 16    ! ✓ Now works correctly
   if (n2 >= 4)  n2 = n2 - 8     ! ✓ 3-bit sign extension

2. OR: Use true symmetric 3.5-bit encoding (see matmul_3p5bit_FIXED.f90)

3. UPDATE DOCUMENTATION: Current scheme is 4+3 asymmetric, not true "3.5-bit"
""")

# Performance impact analysis
print("\n" + "=" * 70)
print("PERFORMANCE IMPACT:")
print("=" * 70)

print("""
The bug causes INCORRECT quantization/dequantization:
  - All n1 values are POSITIVE ONLY (0-7), missing negative range
  - This will cause accuracy loss in model weights
  - Groq LPU will compute WRONG results!

Before deploying to Groq, MUST fix this bug.
Test with small matmul first, compare against known good values.
""")
