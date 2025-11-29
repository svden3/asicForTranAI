# MLIR Generation Output

**Note**: LFortran is not currently installed. This directory contains:
1. Manual MLIR examples (what LFortran would generate)
2. LLVM IR from gfortran (alternative representation)
3. Installation instructions for LFortran

## Install LFortran (Required for MLIR)

### Option 1: Conda (Recommended)
```bash
conda install -c conda-forge lfortran
```

### Option 2: Direct Install
```bash
curl https://lfortran.org/install | bash
```

### Option 3: Build from Source
```bash
git clone https://github.com/lfortran/lfortran
cd lfortran
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

## Generated Files

- `matmul_int4_groq_example.mlir` - Example MLIR (manually created)
- `matmul_int4_groq.ll` - LLVM IR from gfortran
- `matmul_int4_optimized.ll` - LLVM IR (optimized version)

## After Installing LFortran

Run the generation script:
```bash
cd ..
./scripts/generate_mlir.sh
```

This will generate actual MLIR from your Fortran code.
