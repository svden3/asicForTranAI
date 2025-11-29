! Simple test: Just run one forward pass
program test_forward
    use iso_fortran_env, only: int32, real32
    use llama_model
    implicit none

    type(LLaMAModel) :: model
    integer(int32) :: token_ids(10)
    real(real32), allocatable :: logits(:,:)
    integer :: i

    print *, "Simple forward pass test"
    print *, "========================"

    ! Initialize model
    print *, "Initializing model..."
    call init_llama_model(model)
    print *, "Model initialized!"

    ! Create test tokens
    token_ids = [(i, i=1,10)]
    print *, "Test tokens:", token_ids

    ! Allocate output
    allocate(logits(10, 32000))

    ! Forward pass
    print *, "Running forward pass..."
    call forward_llama(model, token_ids, logits, 10)
    print *, "Forward pass complete!"

    print *, "Output logits shape: [", size(logits,1), ",", size(logits,2), "]"
    print *, "Sample logits:", logits(10, 1:5)

    ! Cleanup
    call cleanup_llama_model(model)
    deallocate(logits)

    print *, "✓ Test passed!"

end program test_forward
