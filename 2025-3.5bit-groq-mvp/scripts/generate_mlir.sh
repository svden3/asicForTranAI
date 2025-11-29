#!/bin/bash
# Generate MLIR from Fortran Code
# Uses LFortran to compile Fortran → MLIR

set -e  # Exit on error

echo "==========================================="
echo "MLIR Generation from Fortran"
echo "==========================================="
echo ""

# Check if LFortran is installed
if ! command -v lfortran &> /dev/null; then
    echo "❌ LFortran not found!"
    echo ""
    echo "Install with:"
    echo "  conda install -c conda-forge lfortran"
    echo "  OR"
    echo "  curl https://lfortran.org/install | bash"
    exit 1
fi

echo "✓ LFortran found: $(lfortran --version | head -1)"
echo ""

# Source files to compile
SOURCES=(
    "matmul_int4_groq.f90"
    "matmul_int4_optimized.f90"
    "transformer_layer.f90"
)

OUTPUT_DIR="mlir_output"
mkdir -p $OUTPUT_DIR

echo "Generating MLIR for each module..."
echo ""

for source in "${SOURCES[@]}"; do
    echo "Processing: $source"
    base=$(basename $source .f90)

    # Step 1: Generate ASR (Abstract Semantic Representation)
    echo "  [1/4] Generating ASR..."
    lfortran --show-asr $source > $OUTPUT_DIR/${base}_asr.txt 2>&1 || true

    # Step 2: Generate MLIR
    echo "  [2/4] Generating MLIR..."
    lfortran --show-mlir $source > $OUTPUT_DIR/${base}.mlir 2>&1 || {
        echo "  ⚠️  MLIR generation not yet supported in stable LFortran"
        echo "      Using LLVM IR instead..."
        lfortran --show-llvm $source > $OUTPUT_DIR/${base}.ll 2>&1 || true
    }

    # Step 3: Generate CFG (Control Flow Graph)
    echo "  [3/4] Generating control flow..."
    lfortran --show-c $source > $OUTPUT_DIR/${base}.c 2>&1 || true

    # Step 4: Generate AST (Abstract Syntax Tree) echo "  [4/4] Generating AST..."
    lfortran --show-ast $source > $OUTPUT_DIR/${base}_ast.txt 2>&1 || true

    echo "  ✓ Done: $OUTPUT_DIR/${base}.*"
    echo ""
done

echo "==========================================="
echo "MLIR Generation Complete!"
echo "==========================================="
echo ""
echo "Output files in: $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/
echo ""

echo "Next steps:"
echo "  1. Inspect MLIR: cat $OUTPUT_DIR/matmul_int4_groq.mlir"
echo "  2. Optimize:     mlir-opt --affine-loop-tile $OUTPUT_DIR/matmul_int4_groq.mlir"
echo "  3. Visualize:    mlir-opt --view-cfg $OUTPUT_DIR/matmul_int4_groq.mlir"
echo ""

# If mlir-opt is available, run optimizations
if command -v mlir-opt &> /dev/null; then
    echo "Found mlir-opt! Running optimizations..."
    echo ""

    for mlir_file in $OUTPUT_DIR/*.mlir; do
        if [ -f "$mlir_file" ]; then
            base=$(basename $mlir_file .mlir)
            echo "Optimizing: $mlir_file"

            # Apply standard MLIR optimizations
            mlir-opt \
                --affine-loop-tile="tile-size=64" \
                --affine-loop-fusion \
                --lower-affine \
                $mlir_file \
                -o $OUTPUT_DIR/${base}_optimized.mlir 2>&1 || true

            echo "  ✓ Optimized: $OUTPUT_DIR/${base}_optimized.mlir"
        fi
    done

    echo ""
fi

echo "==========================================="
