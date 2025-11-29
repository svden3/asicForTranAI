# Contributing to asicForTranAI

Thank you for your interest in this 35-year journey from Fortran numerical computing to ASIC AI inference!

## How to Contribute

### Adding Historical Code (1990-2000)

If you have access to historical computational code from the early parallel computing era:

1. **1990 Fortran Code**: Add to `1990-fortran-numerical/`
   - Original source files
   - Documentation of algorithms
   - Performance notes from original hardware

2. **SGI/ML Code**: Add to `2000-sgi-ml-viz/`
   - Machine learning library implementations
   - OpenGL visualization code
   - Notes about working with SGI hardware

### Adding Modern Implementations (2025)

1. **Groq/ASIC Code**: Add to `2025-3.5bit-groq-mvp/`
   - Fortran inference implementations
   - Quantization schemes
   - Performance benchmarks

2. **Verification**: Add to `spark-llama-safety/` or `lean-alphaproof-mcts/`
   - SPARK Ada formal proofs
   - Lean theorem proving
   - Safety arguments

### Code Quality Standards

- **Fortran**: Follow modern Fortran standards (F90+)
- **SPARK Ada**: All code must pass GNATprove verification
- **Lean**: Proofs must compile with Lean 4
- **Documentation**: Explain the "why" not just the "what"

### Commit Messages

Follow conventional commits:
```
feat: Add 1990 parallel solver implementation
docs: Document SGI hardware optimizations
proof: Add Lean theorem for quantization bounds
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Add your code with documentation
4. Ensure all verification passes (if applicable)
5. Submit PR with clear description

## Areas Especially Welcoming Contributions

1. **Historical Context**: Stories, photos, documentation from 1990s/2000s computing
2. **Verification**: Additional SPARK/Lean proofs
3. **Benchmarks**: Performance measurements on various hardware
4. **Documentation**: Tutorials, explanations, historical notes
5. **AI Annotations**: Contributions to the three books project

## Questions?

Open an issue: https://github.com/jimxzai/asicForTranAI/issues

## Code of Conduct

- Be respectful and inclusive
- Focus on technical merit
- Appreciate both historical and modern contributions
- Learn from 35 years of computational evolution

---

**Vision**: 7 years to aviation-grade AI safety on edge devices. Every contribution matters.
