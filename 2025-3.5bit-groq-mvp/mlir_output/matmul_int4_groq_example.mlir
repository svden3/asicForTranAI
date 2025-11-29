// Example MLIR representation of matmul_int4_groq.f90
// This is what LFortran would generate (manually created for demonstration)
// Based on: matmul_int4_awq subroutine

module {
  // Function signature
  func.func @matmul_int4_awq(
    %A: memref<?x?xi8>,           // Input activations [M, K] INT8
    %W_Q: memref<?x?xi8>,         // Quantized weights [K/8, N] INT4 packed
    %W_scales: memref<?xf32>,     // Per-column scales [N] FP32
    %C: memref<?x?xi32>,          // Output accumulator [M, N] INT32
    %M: i32, %N: i32, %K_dim: i32 // Dimensions
  ) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c15 = arith.constant 15 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c_neg4 = arith.constant -4 : i32

    // Outer parallel loop: do concurrent(j=1:N, i=1:M)
    affine.parallel (%i, %j) = (0, 0) to (%M, %N) {

      // Initialize C[i,j] = 0
      %zero = arith.constant 0 : i32
      memref.store %zero, %C[%i, %j] : memref<?x?xi32>

      // Inner loop: process packed INT4 values
      // for (k_idx = 1; k_idx < K_dim; k_idx += 2)
      scf.for %k_idx = %c1 to %K_dim step %c2 {

        // Compute packed index: k_packed = (k_idx + 1) / 2
        %k_plus_1 = arith.addi %k_idx, %c1 : i32
        %k_packed = arith.divui %k_plus_1, %c2 : i32

        // Load packed byte: packed_byte = W_Q[k_packed, j]
        %packed_byte_i8 = memref.load %W_Q[%k_packed, %j] : memref<?x?xi8>
        %packed_byte = arith.extui %packed_byte_i8 : i8 to i32

        // ===== Extract first 4-bit value (lower bits) =====
        // qval = packed_byte & 0xF
        %qval_lower = arith.andi %packed_byte, %c15 : i32

        // Sign extension: if (qval >= 8) qval = qval - 16
        %is_negative = arith.cmpi sge, %qval_lower, %c8 : i32
        %qval1 = arith.select %is_negative,
          arith.subi %qval_lower, %c16 : i32,
          %qval_lower : i32

        // Multiply-accumulate: C[i,j] += A[i, k_idx] * qval1
        %a_val_i8 = memref.load %A[%i, %k_idx] : memref<?x?xi8>
        %a_val = arith.extsi %a_val_i8 : i8 to i32
        %prod1 = arith.muli %a_val, %qval1 : i32

        %old_c = memref.load %C[%i, %j] : memref<?x?xi32>
        %new_c = arith.addi %old_c, %prod1 : i32
        memref.store %new_c, %C[%i, %j] : memref<?x?xi32>

        // ===== Extract second 4-bit value (upper bits) =====
        // Check if k_idx + 1 <= K_dim
        %k_idx_plus_1 = arith.addi %k_idx, %c1 : i32
        %has_second = arith.cmpi sle, %k_idx_plus_1, %K_dim : i32

        scf.if %has_second {
          // qval = (packed_byte >> 4) & 0xF
          %shifted = arith.shrui %packed_byte, %c_neg4 : i32
          %qval_upper = arith.andi %shifted, %c15 : i32

          // Sign extension
          %is_negative2 = arith.cmpi sge, %qval_upper, %c8 : i32
          %qval2 = arith.select %is_negative2,
            arith.subi %qval_upper, %c16 : i32,
            %qval_upper : i32

          // Multiply-accumulate
          %a_val2_i8 = memref.load %A[%i, %k_idx_plus_1] : memref<?x?xi8>
          %a_val2 = arith.extsi %a_val2_i8 : i8 to i32
          %prod2 = arith.muli %a_val2, %qval2 : i32

          %old_c2 = memref.load %C[%i, %j] : memref<?x?xi32>
          %new_c2 = arith.addi %old_c2, %prod2 : i32
          memref.store %new_c2, %C[%i, %j] : memref<?x?xi32>
        }
      }
    }
    return
  }

  // Dequantization function
  func.func @dequantize_output(
    %C: memref<?x?xi32>,
    %W_scales: memref<?xf32>,
    %Out: memref<?x?xf32>,
    %M: i32, %N: i32
  ) {
    // Parallel dequantization: Out[i,j] = C[i,j] * W_scales[j]
    affine.parallel (%i, %j) = (0, 0) to (%M, %N) {
      %c_int32 = memref.load %C[%i, %j] : memref<?x?xi32>
      %c_fp32 = arith.sitofp %c_int32 : i32 to f32
      %scale = memref.load %W_scales[%j] : memref<?xf32>
      %result = arith.mulf %c_fp32, %scale : f32
      memref.store %result, %Out[%i, %j] : memref<?x?xf32>
    }
    return
  }
}

// ============================================
// Optimization Opportunities (for mlir-opt)
// ============================================
//
// 1. Loop tiling:
//    mlir-opt --affine-loop-tile="tile-size=64"
//
// 2. Loop fusion:
//    mlir-opt --affine-loop-fusion
//
// 3. Lower to standard dialect:
//    mlir-opt --lower-affine --convert-scf-to-cf
//
// 4. Groq backend compilation:
//    groq-compiler input.mlir -o output.lpubin
//
// Key MLIR features demonstrated:
// - affine.parallel → Groq systolic array mapping
// - memref → Hardware memory layout
// - arith.* → Hardware arithmetic units
// - scf.for/if → Structured control flow
