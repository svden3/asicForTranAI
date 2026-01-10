#!/bin/bash
# Quick Start Script for LLaMA 70B 3.5-bit Parallel Implementations
# Helps users select and test the best parallel strategy for their hardware

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}LLaMA 70B 3.5-bit Parallel Implementation Quick Start${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Detect hardware
echo -e "${YELLOW}Detecting hardware...${NC}"

# Detect CPU cores
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
echo "  CPU cores: $CPU_CORES"

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "  GPUs: $GPU_COUNT × $GPU_NAME"
else
    GPU_COUNT=0
    echo "  GPUs: None detected"
fi

# Detect MPI
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version | head -n 1)
    echo "  MPI: $MPI_VERSION"
    HAS_MPI=true
else
    echo "  MPI: Not installed"
    HAS_MPI=false
fi

# Detect Fortran compiler
if command -v ifort &> /dev/null; then
    FORTRAN_COMPILER="ifort"
elif command -v gfortran &> /dev/null; then
    FORTRAN_COMPILER="gfortran"
else
    FORTRAN_COMPILER="none"
fi
echo "  Fortran compiler: $FORTRAN_COMPILER"

echo ""

# Recommend strategy based on hardware
echo -e "${YELLOW}Recommended parallel strategy:${NC}"
echo ""

if [ $GPU_COUNT -ge 128 ]; then
    echo -e "${GREEN}✓ Hybrid MPI+OpenMP (3D Parallelism)${NC}"
    echo "  Your system: $GPU_COUNT GPUs"
    echo "  Expected speedup: ~9000× vs baseline"
    echo "  Throughput: ~1.1M tokens/sec"
    RECOMMENDED="hybrid"
elif [ $GPU_COUNT -ge 8 ]; then
    echo -e "${GREEN}✓ Pipeline Parallel${NC}"
    echo "  Your system: $GPU_COUNT GPUs"
    echo "  Expected speedup: ~720× vs baseline"
    echo "  Throughput: ~90,000 tokens/sec"
    RECOMMENDED="pipeline"
elif [ $GPU_COUNT -ge 2 ]; then
    echo -e "${GREEN}✓ MPI Data/Model Parallel${NC}"
    echo "  Your system: $GPU_COUNT GPUs"
    echo "  Expected speedup: ~${GPU_COUNT}90× vs baseline"
    RECOMMENDED="mpi"
elif [ $GPU_COUNT -eq 1 ]; then
    echo -e "${GREEN}✓ cuBLAS GPU + Batch Parallel${NC}"
    echo "  Your system: 1 GPU"
    echo "  Expected speedup: ~100× (single), ~380× (batch 32)"
    RECOMMENDED="gpu"
elif [ "$CPU_CORES" != "unknown" ] && [ $CPU_CORES -ge 32 ]; then
    echo -e "${GREEN}✓ OpenMP Nested (Two-level)${NC}"
    echo "  Your system: $CPU_CORES cores"
    echo "  Expected speedup: ~18× vs baseline"
    RECOMMENDED="openmp-nested"
elif [ "$CPU_CORES" != "unknown" ] && [ $CPU_CORES -ge 8 ]; then
    echo -e "${GREEN}✓ OpenMP Enhanced${NC}"
    echo "  Your system: $CPU_CORES cores"
    echo "  Expected speedup: ~12× vs baseline"
    RECOMMENDED="openmp"
else
    echo -e "${YELLOW}✓ OpenMP SIMD (Baseline)${NC}"
    echo "  Your system: $CPU_CORES cores"
    echo "  Expected speedup: ~7× vs baseline"
    RECOMMENDED="simd"
fi

echo ""
echo -e "${YELLOW}Would you like to:${NC}"
echo "  1) Build and test recommended implementation"
echo "  2) Choose a different implementation"
echo "  3) Run comprehensive benchmarks"
echo "  4) View documentation"
echo "  5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Building $RECOMMENDED implementation...${NC}"

        case $RECOMMENDED in
            "openmp")
                make -f Makefile.parallel openmp
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                echo "Run with: export OMP_NUM_THREADS=$CPU_CORES && ./bin/llama_openmp_enhanced"
                ;;
            "openmp-nested")
                make -f Makefile.parallel openmp
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                echo "Run with: export OMP_NUM_THREADS=$CPU_CORES && ./bin/llama_openmp_enhanced"
                echo "Note: Nested parallelism is enabled by default"
                ;;
            "mpi")
                if [ "$HAS_MPI" = false ]; then
                    echo -e "${RED}Error: MPI not installed. Install with:${NC}"
                    echo "  Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev"
                    echo "  macOS: brew install open-mpi"
                    exit 1
                fi
                make -f Makefile.parallel mpi
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                echo "Run with: mpirun -np $GPU_COUNT ./bin/llama_mpi_parallel"
                ;;
            "pipeline")
                if [ "$HAS_MPI" = false ]; then
                    echo -e "${RED}Error: MPI not installed.${NC}"
                    exit 1
                fi
                make -f Makefile.parallel pipeline
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                echo "Run with: mpirun -np $GPU_COUNT ./bin/llama_pipeline_parallel"
                ;;
            "hybrid")
                if [ "$HAS_MPI" = false ]; then
                    echo -e "${RED}Error: MPI not installed.${NC}"
                    exit 1
                fi
                make -f Makefile.parallel hybrid
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                OMP_THREADS=8
                echo "Run with: mpirun -np $GPU_COUNT -x OMP_NUM_THREADS=$OMP_THREADS ./bin/llama_hybrid_parallel"
                ;;
            "gpu")
                make -f Makefile.parallel gpu
                echo ""
                echo -e "${GREEN}Build complete!${NC}"
                echo "Run cuBLAS: ./bin/llama_gpu_cublas"
                echo "Run OpenACC: ./bin/llama_gpu_openacc"
                ;;
            "simd")
                echo "SIMD implementation is in matmul_simd_optimized.f90"
                echo "Compile with: $FORTRAN_COMPILER -fopenmp -O3 matmul_simd_optimized.f90"
                ;;
        esac
        ;;
    2)
        echo ""
        echo -e "${YELLOW}Available implementations:${NC}"
        echo "  1) OpenMP Enhanced (8-16 cores) - 12× speedup"
        echo "  2) OpenMP Nested (32+ cores) - 18× speedup"
        echo "  3) MPI Data Parallel (multi-GPU) - Linear scaling"
        echo "  4) Pipeline Parallel (8 GPUs) - 720× speedup"
        echo "  5) Batch Parallel (multi-sequence) - 19× speedup"
        echo "  6) Hybrid MPI+OpenMP (128 GPUs) - 9000× speedup"
        echo "  7) cuBLAS GPU - 100× speedup"
        echo ""
        read -p "Enter choice [1-7]: " impl_choice

        case $impl_choice in
            1) make -f Makefile.parallel openmp ;;
            2) make -f Makefile.parallel openmp ;;
            3) make -f Makefile.parallel mpi ;;
            4) make -f Makefile.parallel pipeline ;;
            5) make -f Makefile.parallel batch ;;
            6) make -f Makefile.parallel hybrid ;;
            7) make -f Makefile.parallel gpu ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    3)
        echo ""
        echo -e "${BLUE}Running comprehensive benchmarks...${NC}"
        make -f Makefile.parallel run-benchmark
        echo ""
        echo -e "${GREEN}Benchmark complete!${NC}"
        echo "Results saved to: benchmark_parallel_results.json"
        ;;
    4)
        echo ""
        echo -e "${BLUE}Documentation:${NC}"
        echo ""
        echo "  📖 PARALLEL_OPTIMIZATION_GUIDE.md"
        echo "     Complete guide with compilation instructions, usage examples,"
        echo "     performance tuning tips, and hardware recommendations."
        echo ""
        echo "  📖 PARALLEL_IMPLEMENTATIONS_SUMMARY.md"
        echo "     Quick reference with performance tables and configurations."
        echo ""
        echo "  📖 README.md"
        echo "     Project overview and getting started guide."
        echo ""
        if command -v less &> /dev/null; then
            read -p "Open PARALLEL_OPTIMIZATION_GUIDE.md? [y/N]: " open_doc
            if [ "$open_doc" = "y" ] || [ "$open_doc" = "Y" ]; then
                less PARALLEL_OPTIMIZATION_GUIDE.md
            fi
        fi
        ;;
    5)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}Quick Start Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Test the built implementation"
echo "  2. Run benchmarks: make -f Makefile.parallel run-benchmark"
echo "  3. Read PARALLEL_OPTIMIZATION_GUIDE.md for advanced tuning"
echo ""
echo "For questions or issues, see the documentation or GitHub issues."
