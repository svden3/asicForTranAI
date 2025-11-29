#!/bin/bash
# Set up Lean 4 formal verification project
# Creates complete project structure with starter code

set -e

echo "==========================================="
echo "Lean 4 Verification Project Setup"
echo "==========================================="
echo ""

# Check if elan (Lean version manager) is installed
if ! command -v elan &> /dev/null; then
    echo "❌ Elan (Lean version manager) not found!"
    echo ""
    echo "Install with:"
    echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    echo ""
    read -p "Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
        source ~/.profile || source ~/.bashrc || true
    else
        exit 1
    fi
fi

echo "✓ Elan found: $(elan --version)"
echo ""

# Check if lean is installed
if ! command -v lean &> /dev/null; then
    echo "Installing Lean 4 stable..."
    elan install stable
    elan default stable
fi

echo "✓ Lean found: $(lean --version | head -1)"
echo ""

# Project directory
PROJECT_DIR="../lean-verification"
PROJECT_NAME="Quantization3p5bit"

# Create project
if [ -d "$PROJECT_DIR" ]; then
    echo "⚠️  Project directory already exists: $PROJECT_DIR"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $PROJECT_DIR
    else
        echo "Aborted."
        exit 0
    fi
fi

echo "Creating Lean project: $PROJECT_NAME..."
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Initialize project
lake init $PROJECT_NAME
cd $PROJECT_NAME

echo "✓ Project initialized"
echo ""

# Add Mathlib dependency
echo "Adding Mathlib4 dependency..."
cat >> lakefile.lean <<EOF

-- Add Mathlib dependency
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
EOF

echo "✓ Added Mathlib4"
echo ""

# Create source files
echo "Creating source files..."

# Basic.lean
cat > ${PROJECT_NAME}/Basic.lean <<'EOF'
import Mathlib.Data.Real.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Tactic

/-! # Basic Quantization Definitions

This file contains the fundamental definitions for INT4 and 3.5-bit quantization.
-/

-- Quantization parameters
structure QuantParams where
  scale : ℝ
  zero_point : ℝ
  n_bits : ℕ
  scale_pos : 0 < scale

namespace Quantization

-- Quantization function (ℝ → ℤ)
def quantize (x : ℝ) (p : QuantParams) : ℤ :=
  let q_unclamped := ⌊(x - p.zero_point) / p.scale⌉
  let q_min := -(2^(p.n_bits - 1))
  let q_max := 2^(p.n_bits - 1) - 1
  max q_min (min q_unclamped q_max)

-- Dequantization function (ℤ → ℝ)
def dequantize (q : ℤ) (p : QuantParams) : ℝ :=
  (q : ℝ) * p.scale + p.zero_point

-- INT4 parameters (baseline)
def int4_params (scale zero_point : ℝ) (h : 0 < scale) : QuantParams :=
  { scale := scale
    zero_point := zero_point
    n_bits := 4
    scale_pos := h }

-- Helper lemmas
lemma round_error (x : ℝ) : |x - ⌊x⌉| ≤ 1/2 := by
  sorry  -- TODO: Prove

end Quantization
EOF

# ErrorBounds.lean
cat > ${PROJECT_NAME}/ErrorBounds.lean <<'EOF'
import Quantization3p5bit.Basic

/-! # Error Bound Theorems

This file contains the main error bound theorems for quantization.
-/

namespace Quantization

-- Main theorem: Quantization error is bounded by scale/2
theorem quantization_error_bound (x : ℝ) (p : QuantParams) :
  let q := quantize x p
  let x' := dequantize q p
  |x - x'| ≤ p.scale / 2 := by
  sorry  -- TODO: Prove

-- INT4 specific bound
theorem int4_error_bound (x : ℝ) (scale zero_point : ℝ) (h : 0 < scale) :
  let p := int4_params scale zero_point h
  |x - dequantize (quantize x p) p| ≤ scale / 2 := by
  apply quantization_error_bound

end Quantization
EOF

# MatMul.lean
cat > ${PROJECT_NAME}/MatMul.lean <<'EOF'
import Mathlib.Data.Matrix.Basic
import Quantization3p5bit.Basic
import Quantization3p5bit.ErrorBounds

/-! # Matrix Multiplication Verification

This file verifies the correctness of quantized matrix multiplication.
-/

namespace Quantization

-- Matrix types
def Matrix (α : Type) (m n : ℕ) := Fin m → Fin n → α

-- INT4 matmul (simplified model)
def matmul_int4
  (A : Matrix ℝ m k)
  (W : Matrix ℝ k n)
  (p : QuantParams) :
  Matrix ℤ m n :=
  sorry  -- TODO: Implement

-- Correctness theorem
theorem matmul_int4_correct
  (A : Matrix ℝ m k)
  (W : Matrix ℝ k n)
  (p : QuantParams)
  (hA : ∀ i j, |A i j| ≤ 1)
  (hW : ∀ i j, |W i j| ≤ 1) :
  sorry := by  -- TODO: State and prove
  sorry

-- No integer overflow
theorem no_int32_overflow
  (M N K : ℕ)
  (hK : K ≤ 8192) :
  sorry := by  -- TODO: State and prove
  sorry

end Quantization
EOF

echo "✓ Created source files"
echo ""

# Update dependencies
echo "Fetching Mathlib4 (this may take 5-10 minutes)..."
lake update
lake exe cache get || echo "⚠️  Cache download failed (expected for new projects)"

echo ""
echo "==========================================="
echo "Lean 4 Project Setup Complete!"
echo "==========================================="
echo ""
echo "Project location: $PROJECT_DIR/$PROJECT_NAME"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_DIR/$PROJECT_NAME"
echo "  2. code .  # Open in VS Code"
echo "  3. Edit ${PROJECT_NAME}/Basic.lean"
echo "  4. lake build  # Build project"
echo ""
echo "VS Code Extensions (install if not present):"
echo "  code --install-extension leanprover.lean4"
echo ""
echo "Resources:"
echo "  - Lean 4 Manual: https://lean-lang.org/lean4/doc/"
echo "  - Mathlib docs: https://leanprover-community.github.io/mathlib4_docs/"
echo "  - Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/"
echo ""
echo "==========================================="
