import Lake
open Lake DSL

package quantization3p5bit where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib Quantization3p5bitProof where
  -- Add our proof file to the library
  roots := #[`Quantization3p5bitProof]
