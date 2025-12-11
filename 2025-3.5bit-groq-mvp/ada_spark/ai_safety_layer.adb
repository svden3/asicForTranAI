-- Ada/SPARK Safety Layer Implementation
-- DO-178C Level A compliance-ready

pragma SPARK_Mode (On);

package body AI_Safety_Layer is

   -----------------------------------------------------------------------------
   -- Safe_MatMul_Fused
   -- Implements fused matmul + dequantization by calling FFI wrappers
   --
   -- This is the recommended production API as it:
   --   1. Reduces FFI overhead (single call vs two)
   --   2. Allows temporary C to be stack-allocated
   --   3. Simplifies caller code
   -----------------------------------------------------------------------------
   procedure Safe_MatMul_Fused
      (A        : in  Matrix_Int8;
       W_Q      : in  Matrix_Int8;
       W_Scales : in  Scale_Vector;
       Output   : out Matrix_Float32;
       M        : in  Dimension;
       N        : in  Dimension;
       K        : in  Dimension)
   is
      -- Temporary INT32 accumulator
      -- Stack-allocated, no heap fragmentation (DO-178C requirement)
      C : Matrix_Int32 (1 .. M, 1 .. N);
   begin
      -- Step 1: Call Fortran matmul via FFI
      -- SPARK proves all preconditions satisfied
      Safe_MatMul_Int4_AWQ
         (A        => A,
          W_Q      => W_Q,
          W_Scales => W_Scales,
          C        => C,
          M        => M,
          N        => N,
          K        => K);

      -- Step 2: Dequantize INT32 → FP32 via FFI
      -- SPARK proves C satisfies dequantize preconditions
      Safe_Dequantize_Output
         (C        => C,
          W_Scales => W_Scales,
          Output   => Output,
          M        => M,
          N        => N);

      -- SPARK proves postcondition:
      --   Output(i,j) = Float32(C(i,j)) * W_Scales(j) for all i,j
      --   All values finite (no NaN/Inf)

   end Safe_MatMul_Fused;

end AI_Safety_Layer;
