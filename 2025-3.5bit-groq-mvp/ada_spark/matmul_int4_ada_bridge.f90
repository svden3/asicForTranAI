! Fortran-Ada FFI Bridge for 4-bit MatMul
! Exposes matmul_int4_groq subroutines to Ada/SPARK safety layer
! Uses iso_c_binding for language interoperability

module matmul_int4_ada_bridge
    use iso_c_binding, only: c_int32, c_int8, c_float
    use matmul_int4_groq, only: matmul_int4_awq, dequantize_output
    implicit none

contains

    !---------------------------------------------------------------------------
    ! matmul_int4_awq_wrapper
    ! C-compatible wrapper for Ada pragma Import (Fortran, ...)
    !
    ! Ada calls this via FFI, we call the Fortran matmul_int4_awq implementation
    ! Memory layout is compatible: Ada "Convention => Fortran" = Fortran arrays
    !---------------------------------------------------------------------------
    subroutine matmul_int4_awq_wrapper( &
        A, W_Q, W_scales, C, M, N, K_dim) &
        bind(C, name="matmul_int4_awq_wrapper")

        ! C-compatible types (match Ada Int8/Int32/Float32)
        integer(c_int32), intent(in), value :: M, N, K_dim
        integer(c_int8), intent(in) :: A(M, K_dim)
        integer(c_int8), intent(in) :: W_Q(K_dim/2, N)
        real(c_float), intent(in) :: W_scales(N)
        integer(c_int32), intent(out) :: C(M, N)

        ! Call the actual Fortran implementation
        ! Type conversion is safe: c_int8 = int8, c_int32 = int32, c_float = real32
        call matmul_int4_awq( &
            A        = A, &
            W_Q      = W_Q, &
            W_scales = W_scales, &
            C        = C, &
            M        = M, &
            N        = N, &
            K_dim    = K_dim)

    end subroutine matmul_int4_awq_wrapper


    !---------------------------------------------------------------------------
    ! dequantize_output_wrapper
    ! C-compatible wrapper for Ada pragma Import (Fortran, ...)
    !
    ! Converts INT32 accumulator to FP32 output with per-column scaling
    !---------------------------------------------------------------------------
    subroutine dequantize_output_wrapper( &
        C, W_scales, Out, M, N) &
        bind(C, name="dequantize_output_wrapper")

        integer(c_int32), intent(in), value :: M, N
        integer(c_int32), intent(in) :: C(M, N)
        real(c_float), intent(in) :: W_scales(N)
        real(c_float), intent(out) :: Out(M, N)

        ! Call the actual Fortran implementation
        call dequantize_output( &
            C        = C, &
            W_scales = W_scales, &
            Out      = Out, &
            M        = M, &
            N        = N)

    end subroutine dequantize_output_wrapper


    !---------------------------------------------------------------------------
    ! Optional: Fused wrapper for performance
    ! Combines matmul + dequantization in single FFI call
    ! Reduces Ada-Fortran transitions from 2 to 1
    !---------------------------------------------------------------------------
    subroutine matmul_int4_fused_wrapper( &
        A, W_Q, W_scales, Out, M, N, K_dim) &
        bind(C, name="matmul_int4_fused_wrapper")

        integer(c_int32), intent(in), value :: M, N, K_dim
        integer(c_int8), intent(in) :: A(M, K_dim)
        integer(c_int8), intent(in) :: W_Q(K_dim/2, N)
        real(c_float), intent(in) :: W_scales(N)
        real(c_float), intent(out) :: Out(M, N)

        ! Temporary INT32 accumulator
        integer(c_int32) :: C(M, N)

        ! Step 1: INT8 matmul → INT32 accumulator
        call matmul_int4_awq( &
            A        = A, &
            W_Q      = W_Q, &
            W_scales = W_scales, &
            C        = C, &
            M        = M, &
            N        = N, &
            K_dim    = K_dim)

        ! Step 2: INT32 → FP32 dequantization
        call dequantize_output( &
            C        = C, &
            W_scales = W_scales, &
            Out      = Out, &
            M        = M, &
            N        = N)

    end subroutine matmul_int4_fused_wrapper

end module matmul_int4_ada_bridge
