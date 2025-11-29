! Ultra-simple test: Just initialize model and access embeddings
program test_simple
    use iso_fortran_env, only: int32, real32
    use llama_model
    implicit none

    type(LLaMAModel) :: model
    integer(int32) :: token_id
    real(real32), allocatable :: embedding(:)

    print *, "Simple embedding test"
    print *, "====================="

    ! Initialize model
    print *, "Initializing model..."
    call init_llama_model(model)
    print *, "Model initialized!"

    ! Allocate embedding vector
    allocate(embedding(8192))

    ! Try to access embedding for token 1
    token_id = 1
    print *, "Accessing embedding for token", token_id
    embedding = model%token_embeddings(token_id, :)
    print *, "Embedding first 5 values:", embedding(1:5)

    ! Cleanup
    call cleanup_llama_model(model)
    deallocate(embedding)

    print *, "✓ Test passed!"

end program test_simple
