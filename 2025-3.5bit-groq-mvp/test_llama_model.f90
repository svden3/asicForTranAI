! Test program for full LLaMA 70B model (80 layers)
! Compile: make llama_test

program test_llama_model
    use iso_fortran_env, only: int32, real32
    use llama_model
    implicit none

    type(LLaMAModel) :: model
    integer(int32), parameter :: TEST_SEQ_LEN = 4
    integer(int32) :: token_ids(TEST_SEQ_LEN)
    real(real32), allocatable :: output_logits(:,:)

    print *, "=========================================="
    print *, "LLaMA 70B Full Model Test"
    print *, "80 Transformer Layers"
    print *, "Pure Fortran 2023 - ASIC Optimized"
    print *, "=========================================="
    print *, ""

    ! Initialize the full 80-layer model
    call init_llama_model(model)

    ! Create test input tokens
    token_ids = [1, 2, 3, 4]  ! Example token IDs
    allocate(output_logits(TEST_SEQ_LEN, 32000))

    print *, "Running forward pass through 80 layers..."
    print *, "Input tokens:", token_ids
    print *, ""

    ! Forward pass
    call forward_llama(model, token_ids, output_logits, TEST_SEQ_LEN)

    print *, "✓ Forward pass completed!"
    print *, ""
    print *, "Output logits shape: [", TEST_SEQ_LEN, ", 32000]"
    print *, "Output sample (first position, first 8 logits):"
    write(*, '(8F12.6)') output_logits(1, 1:8)
    print *, ""

    print *, "Model Statistics:"
    print *, "  Total layers processed:", model%num_layers
    print *, "  Parameters (approx):"
    print *, "    Embeddings: 32000 × 8192 = 262M"
    print *, "    Each layer: ~875M params"
    print *, "    80 layers: ~70B params"
    print *, "    Total: ~70B parameters"
    print *, ""

    print *, "Next steps:"
    print *, "  1. Load real LLaMA 70B weights from safetensors"
    print *, "  2. Implement tokenizer (SentencePiece)"
    print *, "  3. Add sampling/generation loop"
    print *, "  4. Benchmark on Groq ASIC (target: 3100+ tok/s)"
    print *, ""

    ! Cleanup
    deallocate(output_logits)
    call cleanup_llama_model(model)

    print *, "✓ Test completed successfully!"

end program test_llama_model
