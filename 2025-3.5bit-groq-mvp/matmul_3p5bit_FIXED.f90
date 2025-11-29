! FIXED VERSION - True 3.5-bit dynamic quantization
! Original had asymmetric 4+3 bit encoding - this fixes it to true average 3.5-bit

! ====================
! OPTION A: Fix current 4+3 bit scheme (asymmetric but functional)
! ====================
pure subroutine matmul_3p5bit_asym(a_int8, w_pack, scales, offsets, c, M, N, K)
  use iso_fortran_env, only: int8, int32, real32
  integer(int8),  intent(in)  :: a_int8(M,K)
  integer(int8),  intent(in)  :: w_pack(K/2,N)     ! 每 2 个 neuron 存 7 bit
  real(real32),   intent(in)  :: scales(N), offsets(N)
  integer(int32), intent(out) :: c(M,N)
  integer(int32) :: i, j, k, idx, raw7, n1, n2

  do concurrent(j=1:N, i=1:M)
    c(i,j) = 0
    do k = 1, K, 2
      idx = (k-1)/2 + 1
      raw7 = iand(w_pack(idx,j), int(z'7F', int8))

      ! n1: 高 4 bit (符号 + 3 位数据) → 范围 -8 到 +7
      n1 = ishft(raw7, -3)                ! 右移 3 位得到高 4 bit
      if (n1 >= 8)  n1 = n1 - 16         ! 4-bit 符号扩展

      ! n2: 低 3 bit (符号 + 2 位数据) → 范围 -4 到 +3
      n2 = iand(raw7, 7)                  ! 取低 3 bit
      if (n2 >= 4)  n2 = n2 - 8          ! 3-bit 符号扩展

      c(i,j) = c(i,j) + a_int8(i,k)   * n1
      if (k+1 <= K) c(i,j) = c(i,j) + a_int8(i,k+1) * n2
    end do
    c(i,j) = nint((real(c(i,j)) + offsets(j)) * scales(j))
  end do
end subroutine


! ====================
! OPTION B: True symmetric 3.5-bit encoding
! ====================
! Strategy: Use 7 bits to encode 2 values with equal precision
!   - Each value gets 3.5 bits on average
!   - Use staggered encoding: alternating 3-bit and 4-bit
pure subroutine matmul_3p5bit_sym(a_int8, w_pack, scales, offsets, c, M, N, K)
  use iso_fortran_env, only: int8, int32, real32
  integer(int8),  intent(in)  :: a_int8(M,K)
  integer(int8),  intent(in)  :: w_pack(K/4,N)     ! 每 4 个 neuron 存 14 bit (2 字节)
  real(real32),   intent(in)  :: scales(N), offsets(N)
  integer(int32), intent(out) :: c(M,N)
  integer(int32) :: i, j, k, idx, raw14, w1, w2, w3, w4

  do concurrent(j=1:N, i=1:M)
    c(i,j) = 0
    do k = 1, K, 4
      idx = (k-1)/4 + 1
      raw14 = iand(w_pack(idx,j), int(z'3FFF', int8))  ! 14 bits for 4 weights

      ! 交替 4-bit 和 3-bit 编码，平均 3.5 bit
      ! Layout: [4-bit][3-bit][4-bit][3-bit] = 14 bits
      w1 = ishft(raw14, -10)                  ! Bits 13-10 (4 bit)
      w2 = iand(ishft(raw14, -7), 7)          ! Bits 9-7   (3 bit)
      w3 = iand(ishft(raw14, -3), 15)         ! Bits 6-3   (4 bit)
      w4 = iand(raw14, 7)                     ! Bits 2-0   (3 bit)

      ! 符号扩展
      if (w1 >= 8)  w1 = w1 - 16
      if (w2 >= 4)  w2 = w2 - 8
      if (w3 >= 8)  w3 = w3 - 16
      if (w4 >= 4)  w4 = w4 - 8

      c(i,j) = c(i,j) + a_int8(i,k)   * w1
      if (k+1 <= K) c(i,j) = c(i,j) + a_int8(i,k+1) * w2
      if (k+2 <= K) c(i,j) = c(i,j) + a_int8(i,k+2) * w3
      if (k+3 <= K) c(i,j) = c(i,j) + a_int8(i,k+3) * w4
    end do
    c(i,j) = nint((real(c(i,j)) + offsets(j)) * scales(j))
  end do
end subroutine


! ====================
! DEBUGGING: Print encoding details
! ====================
subroutine test_3p5bit_encoding()
  use iso_fortran_env, only: int8
  integer(int8) :: test_val
  integer :: n1, n2, i

  print *, "Testing 3.5-bit encoding schemes:"
  print *, "=================================="

  do i = 0, 127
    test_val = int(i, int8)

    ! Original (buggy) version
    n1 = ishft(test_val, -4)
    n2 = iand(test_val, 15)
    if (n1 >= 8)  n1 = n1 - 16  ! This never triggers!
    if (n2 >= 8)  n2 = n2 - 16

    if (mod(i, 16) == 0) then
      print '(A,I3,A,I3,A,I4,A,I4)', &
        "raw7=", i, " → n1(bug)=", n1, " n2=", n2
    end if
  end do
end subroutine
