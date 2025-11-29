# LLaMA 70B INT4 Weights (Groq Format)

## Overview
This directory should contain the 4-bit AWQ quantized weights for LLaMA 70B, formatted for Groq LPU deployment.

## Weight Format

### File Structure
```
llama-70b-awq-int4-groq/
├── model.safetensors         # Main weight file (~35GB)
├── config.json               # Model configuration
├── tokenizer.model           # SentencePiece tokenizer
├── tokenizer_config.json     # Tokenizer settings
└── quantization_config.json  # AWQ quantization parameters
```

### Quantization Specs

- **Method**: AWQ (Activation-aware Weight Quantization)
- **Bits**: 4-bit INT4
- **Group Size**: 128
- **Alignment**: 1024-byte boundaries (Groq requirement)
- **Format**: Packed 2 values per byte (4-bit × 2 = 8-bit)

## How to Obtain Weights

### Option 1: Download Pre-Quantized (Recommended)

```bash
# Using Hugging Face CLI
huggingface-cli download \
    TheBloke/LLaMA-70B-AWQ \
    --local-dir ./llama-70b-awq-int4-groq/

# Or use git-lfs
git lfs install
git clone https://huggingface.co/TheBloke/LLaMA-70B-AWQ
```

### Option 2: Quantize Yourself

```bash
# Install AutoAWQ
pip install autoawq

# Run quantization script
python3 << 'EOF'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-70b-hf"
quant_path = "./llama-70b-awq-int4-groq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(tokenizer, quant_config=quant_config)

# Save for Groq (with alignment)
model.save_quantized(quant_path, safetensors=True)
tokenizer.save_pretrained(quant_path)
EOF
```

### Option 3: Convert from GGUF Format

```bash
# If you have llama-70b-4bit.gguf
pip install gguf-tools

gguf-convert \
    --input llama-70b-4bit.gguf \
    --output-format safetensors \
    --output ./llama-70b-awq-int4-groq/model.safetensors
```

## Groq-Specific Optimizations

### 1. Memory Alignment
Weights must be aligned to 1024-byte boundaries for optimal DMA transfers:

```python
import torch
import safetensors

# Load weights
weights = safetensors.load_file("model.safetensors")

# Align each tensor
aligned_weights = {}
for name, tensor in weights.items():
    # Pad to 1024-byte boundary
    numel = tensor.numel()
    aligned_numel = ((numel + 255) // 256) * 256
    aligned_tensor = torch.zeros(aligned_numel, dtype=tensor.dtype)
    aligned_tensor[:numel] = tensor.flatten()
    aligned_weights[name] = aligned_tensor.reshape(tensor.shape)

# Save aligned version
safetensors.save_file(aligned_weights, "model_aligned.safetensors")
```

### 2. Weight Packing
Pack two 4-bit values per byte for efficient storage:

```fortran
! Fortran unpacking (in matmul_int4_groq.f90)
packed_byte = W_Q(k_packed, j)
lower_4bit = iand(packed_byte, 15)          ! Bits 0-3
upper_4bit = iand(ishft(packed_byte, -4), 15) ! Bits 4-7
```

## License & Usage

### LLaMA 2 Weights
- **License**: Meta's LLaMA 2 License
- **Commercial Use**: Allowed (with restrictions for >700M users)
- **Link**: https://ai.meta.com/llama/license/

### LLaMA 3 Weights (Alternative)
- **License**: More permissive than LLaMA 2
- **Commercial Use**: Generally allowed
- **Link**: https://llama.meta.com/llama-downloads/

**IMPORTANT**: Download weights only if you agree to Meta's license terms.

## Verification

After downloading, verify integrity:

```bash
# Check file size (~35GB for INT4)
du -sh model.safetensors

# Load in Python to verify
python3 << 'EOF'
import safetensors
weights = safetensors.load_file("model.safetensors")
print(f"Loaded {len(weights)} tensors")
print(f"Total parameters: {sum(t.numel() for t in weights.values()):,}")
EOF
```

Expected output:
```
Loaded 723 tensors
Total parameters: 70,014,038,016
```

## Size Estimates

| Format | Size | Notes |
|--------|------|-------|
| FP16 | 140 GB | Original precision |
| INT8 | 70 GB | 2x compression |
| INT4 (AWQ) | 35 GB | 4x compression, minimal quality loss |
| INT3 | 26 GB | Experimental, higher loss |
| INT2 | 18 GB | Extreme compression, quality degraded |

---

**Ready to download?** Use Option 1 above, then run the Groq deployment script!
