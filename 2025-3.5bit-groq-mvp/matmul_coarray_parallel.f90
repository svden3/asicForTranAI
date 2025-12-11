! Coarray Fortran Parallelized INT4 MatMul - PGAS Programming Model
! Target: Simplified parallel programming with built-in sync primitives
! Uses Fortran 2023 coarrays for distributed memory parallelism
!
! Advantages over MPI:
!   - Simpler syntax (no explicit message passing)
!   - One-sided communication model (PGAS)
!   - Built-in synchronization primitives
!   - Better compiler optimizations
!
! Compilation:
!   Intel ifort: ifort -coarray=shared (single node) or -coarray=distributed (multi-node)
!   GCC gfortran: gfortran -fcoarray=single (sequential) or -fcoarray=lib (parallel with OpenCoarrays)
!   NAG nagfor: nagfor -coarray=cosmp

module matmul_coarray_parallel
    use iso_fortran_env, only: int8, int32, real32
    use omp_lib
    implicit none

    private
    public :: matmul_int4_coarray_data_parallel
    public :: matmul_int4_coarray_model_parallel
    public :: matmul_int4_coarray_tensor_parallel
    public :: get_coarray_image_info

    ! Precomputed lookup tables
    integer(int32), parameter :: SIGN_EXTEND_4BIT(0:15) = [ &
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 ]

contains

    !===========================================================================
    ! Get coarray image information (similar to MPI rank/size)
    !===========================================================================
    subroutine get_coarray_image_info(my_image, num_images)
        integer, intent(out) :: my_image, num_images

        my_image = this_image()
        num_images = num_images()

        if (my_image == 1) then
            print '(A)', "=========================================="
            print '(A)', "Coarray Parallel MatMul"
            print '(A)', "=========================================="
            print '(A,I0)', "  Coarray images: ", num_images
            print '(A,I0)', "  OpenMP threads per image: ", omp_get_max_threads()
            print '(A,I0)', "  Total parallelism: ", num_images * omp_get_max_threads()
            print '(A)', "=========================================="
        end if
    end subroutine get_coarray_image_info


    !===========================================================================
    ! DATA PARALLELISM: Distribute batch dimension across coarray images
    ! Each image processes M/num_images rows independently
    !
    ! Coarray advantage: Implicit data distribution, no explicit scatter/gather
    !===========================================================================
    subroutine matmul_int4_coarray_data_parallel(A_local, W_Q, W_scales, C_local, M_local, N, K_dim)
        integer(int8), intent(in) :: A_local(:,:)[*]    ! [M_local, K] - coarray
        integer(int8), intent(in) :: W_Q(:,:)           ! [K/2, N]
        real(real32), intent(in) :: W_scales(:)         ! [N]
        integer(int32), intent(out) :: C_local(:,:)[*]  ! [M_local, N] - coarray
        integer(int32), intent(in) :: M_local, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer :: my_image, num_imgs

        my_image = this_image()
        num_imgs = num_images()

        ! Each image processes its local data (no communication needed)
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M_local
                accum = 0

                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A_local(i, k_idx)[my_image], int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A_local(i, k_idx + 1)[my_image], int32) * qval2
                    end if
                end do

                C_local(i, j)[my_image] = accum
            end do
        end do
        !$omp end parallel do

        ! Synchronize all images
        sync all

        ! Optional: Gather results to image 1
        ! if (my_image == 1) then
        !     do img = 2, num_imgs
        !         C_global(offset:offset+M_local-1, :) = C_local(:, :)[img]
        !     end do
        ! end if

    end subroutine matmul_int4_coarray_data_parallel


    !===========================================================================
    ! MODEL PARALLELISM: Distribute output columns across coarray images
    ! Each image computes N/num_images output columns
    !
    ! Coarray advantage: One-sided read access to replicated input data
    !===========================================================================
    subroutine matmul_int4_coarray_model_parallel(A, W_Q_local, W_scales_local, C_local, M, N_local, K_dim)
        integer(int8), intent(in) :: A(:,:)[*]          ! [M, K] - replicated coarray
        integer(int8), intent(in) :: W_Q_local(:,:)     ! [K/2, N_local]
        real(real32), intent(in) :: W_scales_local(:)   ! [N_local]
        integer(int32), intent(out) :: C_local(:,:)[*]  ! [M, N_local] - coarray
        integer(int32), intent(in) :: M, N_local, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer :: my_image

        my_image = this_image()

        ! Each image computes its subset of output columns
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N_local
            do i = 1, M
                accum = 0

                do k_idx = 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q_local(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    ! One-sided read from replicated A coarray
                    accum = accum + int(A(i, k_idx)[my_image], int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A(i, k_idx + 1)[my_image], int32) * qval2
                    end if
                end do

                C_local(i, j)[my_image] = accum
            end do
        end do
        !$omp end parallel do

        sync all

    end subroutine matmul_int4_coarray_model_parallel


    !===========================================================================
    ! TENSOR PARALLELISM: Distribute K dimension with coarray reduction
    ! Each image computes partial sums, then reduces to get final result
    !
    ! Coarray advantage: Built-in co_sum collective for reductions
    !===========================================================================
    subroutine matmul_int4_coarray_tensor_parallel(A_local, W_Q_local, W_scales, C, M, N, K_local)
        integer(int8), intent(in) :: A_local(:,:)[*]    ! [M, K_local] - coarray
        integer(int8), intent(in) :: W_Q_local(:,:)     ! [K_local/2, N]
        real(real32), intent(in) :: W_scales(:)         ! [N]
        integer(int32), intent(out) :: C(:,:)           ! [M, N]
        integer(int32), intent(in) :: M, N, K_local

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer(int32), allocatable :: C_partial(:,:)
        integer :: my_image

        my_image = this_image()
        allocate(C_partial(M, N))

        ! Each image computes partial matrix multiplication
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum) &
        !$omp schedule(static) collapse(2)
        do j = 1, N
            do i = 1, M
                accum = 0

                do k_idx = 1, K_local, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q_local(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A_local(i, k_idx)[my_image], int32) * qval1

                    if (k_idx + 1 <= K_local) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A_local(i, k_idx + 1)[my_image], int32) * qval2
                    end if
                end do

                C_partial(i, j) = accum
            end do
        end do
        !$omp end parallel do

        ! Coarray collective reduction - much simpler than MPI_Allreduce!
        ! Note: co_sum requires Fortran 2018, use manual reduction for older compilers
        call manual_coarray_sum_reduction(C_partial, C, M, N)

        deallocate(C_partial)

    end subroutine matmul_int4_coarray_tensor_parallel


    !===========================================================================
    ! Manual reduction across coarray images (for Fortran 2008 compatibility)
    ! Implements tree-based reduction for better scaling
    !===========================================================================
    subroutine manual_coarray_sum_reduction(partial, total, M, N)
        integer(int32), intent(in) :: partial(:,:)
        integer(int32), intent(out) :: total(:,:)
        integer(int32), intent(in) :: M, N

        integer :: my_image, num_imgs, stride, peer
        integer(int32), allocatable :: buffer(:,:)[:]

        my_image = this_image()
        num_imgs = num_images()

        ! Allocate coarray buffer for communication
        allocate(buffer(M, N)[*])
        buffer(:, :)[my_image] = partial(:, :)

        sync all

        ! Tree-based reduction (logarithmic depth)
        stride = 1
        do while (stride < num_imgs)
            if (mod(my_image - 1, 2 * stride) == 0) then
                peer = my_image + stride
                if (peer <= num_imgs) then
                    ! Add peer's contribution
                    buffer(:, :)[my_image] = buffer(:, :)[my_image] + buffer(:, :)[peer]
                end if
            end if
            stride = stride * 2
            sync all
        end do

        ! Image 1 has the final result
        if (my_image == 1) then
            total(:, :) = buffer(:, :)[1]
        else
            ! Broadcast result from image 1 to all others
            total(:, :) = buffer(:, :)[1]
        end if

        sync all
        deallocate(buffer)

    end subroutine manual_coarray_sum_reduction


    !===========================================================================
    ! Dequantize with scale factors (parallel across images)
    !===========================================================================
    subroutine dequantize_output_coarray(C, W_scales, Out, M, N)
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

    end subroutine dequantize_output_coarray

end module matmul_coarray_parallel
