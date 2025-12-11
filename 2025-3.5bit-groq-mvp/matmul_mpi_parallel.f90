! MPI Parallelized INT4 MatMul - Distributed Memory Parallelism
! Target: Near-linear scaling across multiple GPUs/nodes
! Uses MPI for model/data/tensor parallelism
!
! Performance scaling (ideal):
!   2 GPUs: 2x throughput (data parallel) or 2x model capacity
!   4 GPUs: 4x throughput or 4x model capacity
!   8 GPUs: 8x throughput or 8x model capacity
!
! Compilation:
!   Windows (Intel MPI): mpiifort -qopenmp matmul_mpi_parallel.f90
!   Linux (OpenMPI): mpifort -fopenmp matmul_mpi_parallel.f90

module matmul_mpi_parallel
    use iso_fortran_env, only: int8, int32, real32
    use mpi_f08
    use omp_lib
    implicit none

    private
    public :: mpi_matmul_init, mpi_matmul_shutdown
    public :: matmul_int4_mpi_data_parallel
    public :: matmul_int4_mpi_model_parallel
    public :: matmul_int4_mpi_tensor_parallel
    public :: mpi_rank, mpi_size, mpi_is_initialized

    ! MPI state
    logical :: mpi_is_initialized = .false.
    integer :: mpi_rank = 0
    integer :: mpi_size = 1
    type(MPI_Comm) :: mpi_comm_world

    ! Precomputed lookup tables for sign extension
    integer(int32), parameter :: SIGN_EXTEND_4BIT(0:15) = [ &
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 ]

contains

    !===========================================================================
    ! Initialize MPI runtime
    ! Call once at program start (before any MPI operations)
    !===========================================================================
    subroutine mpi_matmul_init()
        integer :: ierr

        if (mpi_is_initialized) return

        call MPI_Init(ierr)
        if (ierr /= MPI_SUCCESS) then
            print *, "ERROR: MPI initialization failed"
            stop
        end if

        call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)

        mpi_comm_world = MPI_COMM_WORLD
        mpi_is_initialized = .true.

        if (mpi_rank == 0) then
            print '(A)', "=========================================="
            print '(A)', "MPI Parallel MatMul Initialized"
            print '(A)', "=========================================="
            print '(A,I0)', "  MPI ranks: ", mpi_size
            print '(A,I0)', "  OpenMP threads per rank: ", omp_get_max_threads()
            print '(A,I0)', "  Total parallelism: ", mpi_size * omp_get_max_threads()
            print '(A)', "=========================================="
        end if
    end subroutine mpi_matmul_init


    !===========================================================================
    ! Shutdown MPI runtime
    ! Call once at program end (after all MPI operations)
    !===========================================================================
    subroutine mpi_matmul_shutdown()
        integer :: ierr

        if (.not. mpi_is_initialized) return

        call MPI_Finalize(ierr)
        mpi_is_initialized = .false.

        if (mpi_rank == 0) then
            print '(A)', "✓ MPI shutdown complete"
        end if
    end subroutine mpi_matmul_shutdown


    !===========================================================================
    ! DATA PARALLELISM: Distribute batch dimension across MPI ranks
    ! Each rank processes M/mpi_size rows independently
    !
    ! Use case: Process multiple sequences in parallel (batch inference)
    ! Scaling: Perfect linear scaling (no communication during compute)
    !
    ! Input:  A[M, K] distributed as A_local[M/P, K] on each rank
    ! Output: C[M, N] distributed as C_local[M/P, N] on each rank
    !===========================================================================
    subroutine matmul_int4_mpi_data_parallel(A_local, W_Q, W_scales, C_local, M_local, N, K_dim)
        integer(int8), intent(in) :: A_local(:,:)       ! [M_local, K]
        integer(int8), intent(in) :: W_Q(:,:)           ! [K/2, N]
        real(real32), intent(in) :: W_scales(:)         ! [N]
        integer(int32), intent(out) :: C_local(:,:)     ! [M_local, N]
        integer(int32), intent(in) :: M_local, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum

        ! No MPI communication needed - perfect data parallelism
        ! Each rank processes its local batch independently

        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M_local
                accum = 0

                ! INT4 matrix multiplication (same as sequential)
                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A_local(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A_local(i, k_idx + 1), int32) * qval2
                    end if
                end do

                C_local(i, j) = accum
            end do
        end do
        !$omp end parallel do

        ! Note: No MPI_Allgather needed if each rank only needs its local results
        ! For full result on rank 0, add MPI_Gather(C_local -> C_global)

    end subroutine matmul_int4_mpi_data_parallel


    !===========================================================================
    ! MODEL PARALLELISM: Distribute output columns across MPI ranks
    ! Each rank computes N/mpi_size output columns
    !
    ! Use case: Split large weight matrices across GPUs (memory reduction)
    ! Scaling: Linear scaling, minimal communication (only for final concat)
    !
    ! Input:  A[M, K] replicated on all ranks
    !         W_Q distributed as W_Q_local[K/2, N/P] on each rank
    ! Output: C[M, N] distributed as C_local[M, N/P] on each rank
    !===========================================================================
    subroutine matmul_int4_mpi_model_parallel(A, W_Q_local, W_scales_local, C_local, M, N_local, K_dim)
        integer(int8), intent(in) :: A(:,:)             ! [M, K]
        integer(int8), intent(in) :: W_Q_local(:,:)     ! [K/2, N_local]
        real(real32), intent(in) :: W_scales_local(:)   ! [N_local]
        integer(int32), intent(out) :: C_local(:,:)     ! [M, N_local]
        integer(int32), intent(in) :: M, N_local, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum

        ! Each rank computes its subset of output columns
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N_local
            do i = 1, M
                accum = 0

                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q_local(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A(i, k_idx + 1), int32) * qval2
                    end if
                end do

                C_local(i, j) = accum
            end do
        end do
        !$omp end parallel do

        ! Note: To get full C[M, N], use MPI_Allgather(C_local -> C_global)
        ! Or keep distributed for next layer (pipeline parallelism)

    end subroutine matmul_int4_mpi_model_parallel


    !===========================================================================
    ! TENSOR PARALLELISM: Distribute K dimension with MPI_Allreduce
    ! Each rank computes partial sums, then reduces to get final result
    !
    ! Use case: Very large matrices that don't fit in single GPU memory
    ! Scaling: Good scaling, requires MPI_Allreduce per matmul
    !
    ! Input:  A distributed as A_local[M, K/P] on each rank
    !         W_Q distributed as W_Q_local[K/(2*P), N] on each rank
    ! Output: C[M, N] replicated on all ranks via MPI_Allreduce
    !===========================================================================
    subroutine matmul_int4_mpi_tensor_parallel(A_local, W_Q_local, W_scales, C, M, N, K_local)
        integer(int8), intent(in) :: A_local(:,:)       ! [M, K_local]
        integer(int8), intent(in) :: W_Q_local(:,:)     ! [K_local/2, N]
        real(real32), intent(in) :: W_scales(:)         ! [N]
        integer(int32), intent(out) :: C(:,:)           ! [M, N]
        integer(int32), intent(in) :: M, N, K_local

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer(int32), allocatable :: C_partial(:,:)   ! Local partial results
        integer :: ierr

        allocate(C_partial(M, N))

        ! Each rank computes partial matrix multiplication over its K_local slice
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M
                accum = 0

                do k_idx = 1, K_local, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q_local(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A_local(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_local) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A_local(i, k_idx + 1), int32) * qval2
                    end if
                end do

                C_partial(i, j) = accum
            end do
        end do
        !$omp end parallel do

        ! Reduce partial results across all ranks: C = sum(C_partial_rank_i)
        call MPI_Allreduce(C_partial, C, M * N, MPI_INTEGER, MPI_SUM, &
                           mpi_comm_world, ierr)

        deallocate(C_partial)

    end subroutine matmul_int4_mpi_tensor_parallel


    !===========================================================================
    ! Helper: Dequantize output with scale factors (FP32 result)
    ! Can be used after any of the above MPI matmul variants
    !===========================================================================
    subroutine dequantize_output_mpi(C, W_scales, Out, M, N)
        integer(int32), intent(in) :: C(:,:)      ! [M, N]
        real(real32), intent(in) :: W_scales(:)   ! [N]
        real(real32), intent(out) :: Out(:,:)     ! [M, N]
        integer(int32), intent(in) :: M, N

        integer(int32) :: i, j

        !$omp parallel do private(i,j) schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M
                Out(i, j) = real(C(i, j), real32) * W_scales(j)
            end do
        end do
        !$omp end parallel do

    end subroutine dequantize_output_mpi

end module matmul_mpi_parallel
