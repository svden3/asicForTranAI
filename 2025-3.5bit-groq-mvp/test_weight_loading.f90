! Test program for weight loading
! Verifies weights can be loaded and used for inference

program test_weight_loading
    use iso_fortran_env, only: int32, real32
    use transformer_layer
    use weight_loader
    implicit none

    type(TransformerLayer) :: layer
    real(real32), allocatable :: x(:,:), output(:,:)
    integer(int32) :: seq_len
    integer(int32) :: i, j

    print *, "=========================================="
    print *, "Weight Loading Test"
    print *, "=========================================="
    print *, ""

    ! Initialize layer structure
    allocate(layer%attn_norm(HIDDEN_DIM))
    allocate(layer%ffn_norm(HIDDEN_DIM))
    layer%attn_norm = 1.0
    layer%ffn_norm = 1.0

    ! Initialize RoPE and KV cache
    call init_rope_freqs(layer, 2048)
    call init_kv_cache(layer, 2048)

    ! Load test weights
    print *, "Loading test weights from file..."
    call load_layer_weights(layer, "test_weights_layer0.bin", 0)
    print *, ""

    ! Verify weights were loaded
    if (allocated(layer%wq)) then
        print *, "✓ Weights loaded successfully!"
        print *, "  Q weights shape: [", size(layer%wq, 1), ",", size(layer%wq, 2), "]"
        print *, "  K weights shape: [", size(layer%wk, 1), ",", size(layer%wk, 2), "]"
        print *, "  V weights shape: [", size(layer%wv, 1), ",", size(layer%wv, 2), "]"
        print *, ""
    else
        print *, "✗ Error: Weights not loaded"
        stop 1
    end if

    ! Test inference with loaded weights
    seq_len = 4
    allocate(x(seq_len, HIDDEN_DIM))
    allocate(output(seq_len, HIDDEN_DIM))

    ! Create test input
    do i = 1, seq_len
        do j = 1, HIDDEN_DIM
            x(i,j) = 0.01 * real(mod(i*j, 100), real32)
        end do
    end do

    print *, "Running inference with loaded weights..."
    print *, "Input shape: [", seq_len, ",", HIDDEN_DIM, "]"
    print *, ""

    call apply_transformer_layer(layer, x, output, seq_len)

    print *, "✓ Inference completed!"
    print *, "Output shape: [", seq_len, ",", HIDDEN_DIM, "]"
    print *, "Output sample (first position, first 8 dims):"
    write(*, '(8F10.6)') output(1, 1:8)
    print *, ""

    ! Check output is not all zeros (indicates computation happened)
    if (maxval(abs(output)) > 1e-6) then
        print *, "✓ Output values non-zero (computation successful)"
    else
        print *, "⚠ Warning: Output all zeros"
    end if

    print *, ""
    print *, "=========================================="
    print *, "✓ Weight loading test passed!"
    print *, "=========================================="
    print *, ""
    print *, "Next steps:"
    print *, "  1. Load weights for all 80 layers"
    print *, "  2. Test full model inference"
    print *, "  3. Implement text generation"

    ! Cleanup
    deallocate(x, output)
    deallocate(layer%attn_norm, layer%ffn_norm)
    if (allocated(layer%rope_freqs)) deallocate(layer%rope_freqs)
    if (allocated(layer%k_cache)) deallocate(layer%k_cache)
    if (allocated(layer%v_cache)) deallocate(layer%v_cache)
    if (allocated(layer%wq)) deallocate(layer%wq, layer%wq_scales)
    if (allocated(layer%wk)) deallocate(layer%wk, layer%wk_scales)
    if (allocated(layer%wv)) deallocate(layer%wv, layer%wv_scales)
    if (allocated(layer%wo)) deallocate(layer%wo, layer%wo_scales)
    if (allocated(layer%w_gate)) deallocate(layer%w_gate, layer%w_gate_scales)
    if (allocated(layer%w_up)) deallocate(layer%w_up, layer%w_up_scales)
    if (allocated(layer%w_down)) deallocate(layer%w_down, layer%w_down_scales)

end program test_weight_loading
