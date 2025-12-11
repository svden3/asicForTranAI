! Enhanced OpenMP INT4 MatMul - Advanced Thread Scaling
! Target: 10-15x speedup on high-core CPUs (32+ cores)
! Features:
!   - Nested parallelism for multi-level task decomposition
!   - Dynamic load balancing with task-based parallelism
!   - Cache-aware tiling for better memory locality
!   - NUMA-aware thread placement
!   - Vectorization hints for compiler auto-vectorization
!
! Compilation:
!   Intel: ifort -qopenmp -O3 -xHost -qopt-report=5
!   GCC:   gfortran -fopenmp -O3 -march=native -ftree-vectorize
!   MSVC:  cl /Qopenmp /O2 /arch:AVX512

module matmul_openmp_enhanced
    use iso_fortran_env, only: int8, int32, real32
    use omp_lib
    implicit none

    private
    public :: matmul_int4_openmp_enhanced
    public :: matmul_int4_openmp_nested
    public :: matmul_int4_openmp_tiled
    public :: matmul_int4_openmp_tasks
    public :: dequantize_output_openmp

    ! Lookup tables for sign extension
    integer(int32), parameter :: SIGN_EXTEND_4BIT(0:15) = [ &
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 ]

    ! Cache blocking parameters (tuned for modern CPUs)
    integer(int32), parameter :: TILE_M = 64     ! L1 cache friendly
    integer(int32), parameter :: TILE_N = 64
    integer(int32), parameter :: TILE_K = 256
    integer(int32), parameter :: BLOCK_M = 256   ! L2 cache friendly
    integer(int32), parameter :: BLOCK_N = 256

contains

    !===========================================================================
    ! Enhanced OpenMP MatMul - Single-level parallelism with advanced opts
    ! Best for: 8-16 core CPUs, simple workloads
    !===========================================================================
    subroutine matmul_int4_openmp_enhanced(A, W_Q, W_scales, C, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)           ! [M, K]
        integer(int8), intent(in) :: W_Q(:,:)         ! [K/2, N]
        real(real32), intent(in) :: W_scales(:)       ! [N]
        integer(int32), intent(out) :: C(:,:)         ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer(int32) :: num_threads

        num_threads = omp_get_max_threads()

        ! Outer loop parallelization with dynamic scheduling
        ! Dynamic scheduling provides better load balancing for irregular workloads
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(dynamic, max(1, N/(num_threads*4))) collapse(2) &
        !$omp num_threads(num_threads)
        do j = 1, N
            do i = 1, M
                accum = 0

                ! Inner loop with SIMD directives
                !$omp simd reduction(+:accum) simdlen(8) &
                !$omp aligned(A:64, W_Q:64)
                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A(i, k_idx + 1), int32) * qval2
                    end if
                end do
                !$omp end simd

                C(i, j) = accum
            end do
        end do
        !$omp end parallel do

    end subroutine matmul_int4_openmp_enhanced


    !===========================================================================
    ! Nested Parallelism - Two-level task decomposition
    ! Best for: 32+ core CPUs, large matrices (M,N > 2048)
    ! Enables: Outer parallelism over rows, inner over columns
    !===========================================================================
    subroutine matmul_int4_openmp_nested(A, W_Q, W_scales, C, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)
        integer(int8), intent(in) :: W_Q(:,:)
        real(real32), intent(in) :: W_scales(:)
        integer(int32), intent(out) :: C(:,:)
        integer(int32), intent(in) :: M, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer :: outer_threads, inner_threads, total_threads

        ! Enable nested parallelism
        call omp_set_nested(.true.)
        call omp_set_max_active_levels(2)

        total_threads = omp_get_max_threads()
        ! Split threads: sqrt(total) per level for balanced nested parallelism
        outer_threads = int(sqrt(real(total_threads)))
        inner_threads = max(1, total_threads / outer_threads)

        ! Outer parallelism: Distribute rows across thread teams
        !$omp parallel num_threads(outer_threads) &
        !$omp private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum)

        !$omp do schedule(static)
        do i = 1, M
            ! Inner parallelism: Distribute columns within team
            !$omp parallel do num_threads(inner_threads) &
            !$omp private(j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
            !$omp schedule(static)
            do j = 1, N
                accum = 0

                !$omp simd reduction(+:accum)
                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A(i, k_idx + 1), int32) * qval2
                    end if
                end do
                !$omp end simd

                C(i, j) = accum
            end do
            !$omp end parallel do
        end do
        !$omp end do

        !$omp end parallel

    end subroutine matmul_int4_openmp_nested


    !===========================================================================
    ! Cache-Aware Tiled MatMul - Three-level blocking
    ! Best for: Large matrices (8192+), CPUs with large L3 cache
    ! Optimizes: Memory hierarchy (L1/L2/L3 cache utilization)
    !===========================================================================
    subroutine matmul_int4_openmp_tiled(A, W_Q, W_scales, C, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)
        integer(int8), intent(in) :: W_Q(:,:)
        real(real32), intent(in) :: W_scales(:)
        integer(int32), intent(out) :: C(:,:)
        integer(int32), intent(in) :: M, N, K_dim

        integer(int32) :: i, j, k, ii, jj, kk
        integer(int32) :: i_end, j_end, k_end
        integer(int32) :: k_idx, k_packed, packed_byte, qval1, qval2
        integer(int32) :: accum

        ! Initialize output
        C = 0

        ! Outer blocking: L3 cache level (distribute across threads)
        !$omp parallel do private(i,j,k,ii,jj,kk,i_end,j_end,k_end,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(dynamic) collapse(2)
        do jj = 1, N, BLOCK_N
            do ii = 1, M, BLOCK_M
                j_end = min(jj + BLOCK_N - 1, N)
                i_end = min(ii + BLOCK_M - 1, M)

                ! Middle blocking: L2 cache level
                do kk = 1, K_dim, TILE_K * 2
                    k_end = min(kk + TILE_K * 2 - 1, K_dim)

                    ! Inner blocking: L1 cache level (vectorized)
                    do j = jj, j_end, TILE_N
                        do i = ii, i_end, TILE_M
                            ! Compute tile
                            !$omp simd private(k_idx,k_packed,packed_byte,qval1,qval2)
                            do k_idx = kk, k_end, 2
                                k_packed = (k_idx + 1) / 2

                                ! Process TILE_M × TILE_N sub-block
                                do j = j, min(j + TILE_N - 1, j_end)
                                    packed_byte = int(W_Q(k_packed, j), int32)
                                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                                    if (k_idx + 1 <= K_dim) then
                                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                                    end if

                                    do i = i, min(i + TILE_M - 1, i_end)
                                        C(i, j) = C(i, j) + int(A(i, k_idx), int32) * qval1
                                        if (k_idx + 1 <= K_dim) then
                                            C(i, j) = C(i, j) + int(A(i, k_idx + 1), int32) * qval2
                                        end if
                                    end do
                                end do
                            end do
                            !$omp end simd
                        end do
                    end do
                end do
            end do
        end do
        !$omp end parallel do

    end subroutine matmul_int4_openmp_tiled


    !===========================================================================
    ! Task-Based Parallelism - Dynamic task scheduling
    ! Best for: Very large matrices with load imbalance
    ! Features: Work-stealing for optimal load distribution
    !===========================================================================
    subroutine matmul_int4_openmp_tasks(A, W_Q, W_scales, C, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)
        integer(int8), intent(in) :: W_Q(:,:)
        real(real32), intent(in) :: W_scales(:)
        integer(int32), intent(out) :: C(:,:)
        integer(int32), intent(in) :: M, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer(int32) :: block_i, block_j, i_start, i_end, j_start, j_end
        integer(int32) :: num_blocks_m, num_blocks_n
        integer(int32), parameter :: BLOCK_SIZE = 128

        ! Calculate number of blocks
        num_blocks_m = (M + BLOCK_SIZE - 1) / BLOCK_SIZE
        num_blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE

        !$omp parallel private(block_i,block_j,i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum,i_start,i_end,j_start,j_end)
        !$omp single

        ! Create tasks for each block (work-stealing enabled)
        do block_j = 0, num_blocks_n - 1
            do block_i = 0, num_blocks_m - 1
                !$omp task firstprivate(block_i,block_j) &
                !$omp private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum,i_start,i_end,j_start,j_end)

                i_start = block_i * BLOCK_SIZE + 1
                i_end = min(i_start + BLOCK_SIZE - 1, M)
                j_start = block_j * BLOCK_SIZE + 1
                j_end = min(j_start + BLOCK_SIZE - 1, N)

                ! Compute block
                do j = j_start, j_end
                    do i = i_start, i_end
                        accum = 0

                        !$omp simd reduction(+:accum)
                        do k_idx = 1, K_dim, 2
                            k_packed = (k_idx + 1) / 2
                            packed_byte = int(W_Q(k_packed, j), int32)

                            qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                            accum = accum + int(A(i, k_idx), int32) * qval1

                            if (k_idx + 1 <= K_dim) then
                                qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                                accum = accum + int(A(i, k_idx + 1), int32) * qval2
                            end if
                        end do
                        !$omp end simd

                        C(i, j) = accum
                    end do
                end do

                !$omp end task
            end do
        end do

        !$omp taskwait
        !$omp end single
        !$omp end parallel

    end subroutine matmul_int4_openmp_tasks


    !===========================================================================
    ! Enhanced dequantization with OpenMP parallelism
    !===========================================================================
    subroutine dequantize_output_openmp(C, W_scales, Out, M, N)
        integer(int32), intent(in) :: C(:,:)
        real(real32), intent(in) :: W_scales(:)
        real(real32), intent(out) :: Out(:,:)
        integer(int32), intent(in) :: M, N

        integer(int32) :: i, j

        !$omp parallel do simd private(i,j) schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M
                Out(i, j) = real(C(i, j), real32) * W_scales(j)
            end do
        end do
        !$omp end parallel do simd

    end subroutine dequantize_output_openmp

end module matmul_openmp_enhanced
