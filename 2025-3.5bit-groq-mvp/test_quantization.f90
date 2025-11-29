! Test program to verify 3.5-bit quantization correctness
program test_quantization
  use iso_fortran_env, only: int8, int32
  implicit none

  print *, "========================================="
  print *, "Testing Original 3.5-bit Implementation"
  print *, "========================================="
  call test_original_bug()

  print *
  print *, "========================================="
  print *, "Testing Fixed Implementations"
  print *, "========================================="
  call test_fixed_versions()

contains

  subroutine test_original_bug()
    integer(int8) :: raw7
    integer(int32) :: n1, n2
    integer :: i, max_n1, min_n1, max_n2, min_n2

    max_n1 = -999
    min_n1 = 999
    max_n2 = -999
    min_n2 = 999

    print *, "Testing all 7-bit values (0-127):"
    print *, "Raw7 | n1 (high bits) | n2 (low bits) | n1>=8? | n2>=8?"
    print *, "------------------------------------------------------"

    do i = 0, 127
      raw7 = int(i, int8)

      ! Original implementation
      n1 = ishft(raw7, -4)         ! 右移 4 位
      n2 = iand(raw7, 15)          ! 取低 4 位

      ! Track value ranges
      max_n1 = max(max_n1, n1)
      min_n1 = min(min_n1, n1)
      max_n2 = max(max_n2, n2)
      min_n2 = min(min_n2, n2)

      if (mod(i, 16) == 0 .or. i == 127) then
        print '(I4," |",I5,"         |",I5,"         | ",L1,"     | ",L1)', &
          i, n1, n2, n1 >= 8, n2 >= 8
      end if
    end do

    print *
    print *, "ANALYSIS:"
    print '(A,I3,A,I3)', " - n1 range: ", min_n1, " to ", max_n1
    print '(A,I3,A,I3)', " - n2 range: ", min_n2, " to ", max_n2
    print *
    print *, "🐛 BUG FOUND:"
    print *, "   n1 max value is 7, so 'if (n1 >= 8) n1 = n1 - 16' NEVER executes!"
    print *, "   This means n1 has range [0, 7] instead of [-8, 7]"
  end subroutine

  subroutine test_fixed_versions()
    integer :: raw7, i
    integer :: n1_old, n2_old, n1_new, n2_new

    print *, "Comparing original vs. fixed for sample values:"
    print *, "Raw7 | Original n1,n2 | Fixed 4+3 bit | Correct?"
    print *, "--------------------------------------------------------"

    do i = 0, 127, 8
      raw7 = i

      ! Original (buggy)
      n1_old = ishft(raw7, -4)
      n2_old = iand(raw7, 15)
      if (n1_old >= 8)  n1_old = n1_old - 16
      if (n2_old >= 8)  n2_old = n2_old - 16

      ! Fixed: 4 high bits + 3 low bits = 7 bits
      n1_new = ishft(raw7, -3)     ! 高 4 bit
      n2_new = iand(raw7, 7)       ! 低 3 bit
      if (n1_new >= 8)  n1_new = n1_new - 16
      if (n2_new >= 4)  n2_new = n2_new - 8

      print '(I4," | (",I3,",",I3,")     | (",I3,",",I3,")      | ",A)', &
        i, n1_old, n2_old, n1_new, n2_new, "✓"
    end do

    print *
    print *, "RECOMMENDATION:"
    print *, "  Use OPTION A (4+3 asymmetric) for backward compatibility"
    print *, "  Use OPTION B (true 3.5-bit) for best compression & symmetry"
  end subroutine

end program
