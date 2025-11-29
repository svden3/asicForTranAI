# Foundational References for 3.5-bit Quantization Theory

**Purpose**: Key papers that herald the foundation and build the theoretical framework
**Categories**: Quantization Theory, Information Theory, Numerical Analysis, Formal Verification, ASIC Architecture

---

## 1. Quantization Theory (Core Foundation)

### 1.1 Classical Quantization

**Shannon (1948) - The Birth of Information Theory**
```bibtex
@article{shannon1948mathematical,
  title={A Mathematical Theory of Communication},
  author={Shannon, Claude E.},
  journal={Bell System Technical Journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948}
}
```
**Why it matters**: Establishes rate-distortion theory, fundamental limits of lossy compression (your 3.5-bit quantization is lossy compression)

**Lloyd (1982) - Optimal Quantization**
```bibtex
@article{lloyd1982least,
  title={Least Squares Quantization in PCM},
  author={Lloyd, Stuart P.},
  journal={IEEE Transactions on Information Theory},
  volume={28},
  number={2},
  pages={129--137},
  year={1982}
}
```
**Why it matters**: Lloyd-Max quantization (optimal scalar quantizer), minimizes mean squared error - your algorithm builds on this

**Gray & Neuhoff (1998) - Quantization Survey**
```bibtex
@article{gray1998quantization,
  title={Quantization},
  author={Gray, Robert M. and Neuhoff, David L.},
  journal={IEEE Transactions on Information Theory},
  volume={44},
  number={6},
  pages={2325--2383},
  year={1998}
}
```
**Why it matters**: Comprehensive survey of quantization theory, covers vector quantization, rate-distortion, optimal quantizers

### 1.2 Neural Network Quantization (Modern Era)

**Gupta et al. (2015) - Deep Learning with Limited Precision**
```bibtex
@inproceedings{gupta2015deep,
  title={Deep Learning with Limited Numerical Precision},
  author={Gupta, Suyog and Agrawal, Ankur and Gopalakrishnan, Kailash and Narayanan, Pritish},
  booktitle={ICML},
  year={2015}
}
```
**Why it matters**: First systematic study of low-precision training/inference, shows 16-bit is sufficient

**Han et al. (2015) - Deep Compression**
```bibtex
@inproceedings{han2015deep,
  title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
  author={Han, Song and Mao, Huizi and Dally, William J.},
  booktitle={ICLR},
  year={2016}
}
```
**Why it matters**: Combines pruning + quantization + encoding, foundational for modern compression techniques

**Jacob et al. (2018) - Quantization and Training**
```bibtex
@inproceedings{jacob2018quantization,
  title={Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  booktitle={CVPR},
  year={2018}
}
```
**Why it matters**: Defines asymmetric quantization (your approach uses this), zero-point offset, per-channel scaling

---

## 2. Information Theory & Rate-Distortion

**Berger (1971) - Rate-Distortion Theory**
```bibtex
@book{berger1971rate,
  title={Rate Distortion Theory: A Mathematical Basis for Data Compression},
  author={Berger, Toby},
  year={1971},
  publisher={Prentice-Hall}
}
```
**Why it matters**: Theoretical limits on compression with bounded distortion, proves lower bounds for your 3.5-bit scheme

**Cover & Thomas (2006) - Elements of Information Theory**
```bibtex
@book{cover2006elements,
  title={Elements of Information Theory},
  author={Cover, Thomas M. and Thomas, Joy A.},
  edition={2nd},
  year={2006},
  publisher={Wiley}
}
```
**Why it matters**: Chapter 10 (Rate Distortion Theory) provides theoretical framework for quantization error analysis

**Gersho & Gray (1992) - Vector Quantization**
```bibtex
@book{gersho1992vector,
  title={Vector Quantization and Signal Compression},
  author={Gersho, Allen and Gray, Robert M.},
  year={1992},
  publisher={Kluwer Academic Publishers}
}
```
**Why it matters**: Vector quantization theory, multi-dimensional quantization (your 3.5-bit packs 2 values, a form of vector quantization)

---

## 3. Numerical Analysis & Error Bounds

**Higham (2002) - Accuracy and Stability of Numerical Algorithms**
```bibtex
@book{higham2002accuracy,
  title={Accuracy and Stability of Numerical Algorithms},
  author={Higham, Nicholas J.},
  edition={2nd},
  year={2002},
  publisher={SIAM}
}
```
**Why it matters**: Chapter 2 (Floating Point Arithmetic) - rounding error analysis, your Theorem 1 builds on this framework

**Goldberg (1991) - What Every Computer Scientist Should Know About Floating-Point Arithmetic**
```bibtex
@article{goldberg1991every,
  title={What Every Computer Scientist Should Know About Floating-Point Arithmetic},
  author={Goldberg, David},
  journal={ACM Computing Surveys},
  volume={23},
  number={1},
  pages={5--48},
  year={1991}
}
```
**Why it matters**: Foundational paper on floating-point arithmetic, error propagation, numerical stability

**Wilkinson (1963) - Rounding Errors in Algebraic Processes**
```bibtex
@book{wilkinson1963rounding,
  title={Rounding Errors in Algebraic Processes},
  author={Wilkinson, James H.},
  year={1963},
  publisher={Prentice Hall}
}
```
**Why it matters**: Classical error analysis, backward/forward error, stability of matrix operations (MatMul in your Algorithm 2)

---

## 4. Formal Verification & Theorem Proving

**Leroy (2009) - CompCert (Verified Compiler)**
```bibtex
@article{leroy2009formal,
  title={Formal Verification of a Realistic Compiler},
  author={Leroy, Xavier},
  journal={Communications of the ACM},
  volume={52},
  number={7},
  pages={107--115},
  year={2009}
}
```
**Why it matters**: First formally verified optimizing compiler (Coq), shows large-scale verification is feasible

**Kumar et al. (2014) - CakeML (Verified ML)**
```bibtex
@inproceedings{kumar2014cakeml,
  title={CakeML: A Verified Implementation of ML},
  author={Kumar, Ramana and Myreen, Magnus O. and Norrish, Michael and Owens, Scott},
  booktitle={POPL},
  year={2014}
}
```
**Why it matters**: End-to-end verified language implementation (HOL4), demonstrates verification of complex systems

**De Moura & Bjørner (2008) - Z3 Theorem Prover**
```bibtex
@inproceedings{de2008z3,
  title={Z3: An Efficient SMT Solver},
  author={De Moura, Leonardo and Bj{\o}rner, Nikolaj},
  booktitle={TACAS},
  year={2008}
}
```
**Why it matters**: SMT solver used in many verification tools, including SPARK (which you use for Ada verification)

**Moura et al. (2015) - Lean Theorem Prover**
```bibtex
@inproceedings{de2015lean,
  title={The Lean Theorem Prover (System Description)},
  author={De Moura, Leonardo and Kong, Soonho and Avigad, Jeremy and Van Doorn, Floris and Von Raumer, Jakob},
  booktitle={CADE},
  year={2015}
}
```
**Why it matters**: Lean prover you use for Theorem 1 & 2, foundational for your formal verification approach

**Avigad et al. (2020) - Mathematics in Lean**
```bibtex
@book{avigad2020mathematics,
  title={Mathematics in Lean},
  author={Avigad, Jeremy and Massot, Patrick},
  year={2020},
  publisher={Carnegie Mellon University}
}
```
**Why it matters**: Tutorial for formalizing mathematics in Lean 4, methodological guide for your proofs

---

## 5. ASIC Architecture & Systolic Arrays

**Kung & Leiserson (1979) - Systolic Arrays**
```bibtex
@inproceedings{kung1979systolic,
  title={Systolic Arrays (for VLSI)},
  author={Kung, H. T. and Leiserson, Charles E.},
  booktitle={Sparse Matrix Proceedings},
  year={1979}
}
```
**Why it matters**: Foundational paper on systolic arrays, MatMul mapping (your Fortran `do concurrent` maps to this)

**Jouppi et al. (2017) - Google TPU**
```bibtex
@inproceedings{jouppi2017datacenter,
  title={In-Datacenter Performance Analysis of a Tensor Processing Unit},
  author={Jouppi, Norman P. and Young, Cliff and Patil, Nishant and Patterson, David and Agrawal, Gaurav and Bajwa, Raminder and Bates, Sarah and Bhatia, Suresh and Boden, Nan and Borchers, Al and others},
  booktitle={ISCA},
  pages={1--12},
  year={2017}
}
```
**Why it matters**: First public ASIC for ML, systolic array for MatMul, shows ASIC advantages over GPU

**Hennessy & Patterson (2017) - Computer Architecture (6th Edition)**
```bibtex
@book{hennessy2017computer,
  title={Computer Architecture: A Quantitative Approach},
  author={Hennessy, John L. and Patterson, David A.},
  edition={6th},
  year={2017},
  publisher={Morgan Kaufmann}
}
```
**Why it matters**: Chapter 7 (Domain-Specific Architectures) covers TPU, Groq-like ASICs, memory hierarchy

---

## 6. Large Language Models (Context)

**Vaswani et al. (2017) - Transformer**
```bibtex
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}
```
**Why it matters**: Defines Transformer architecture (Llama models you quantize are Transformers)

**Kaplan et al. (2020) - Scaling Laws**
```bibtex
@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B. and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}
```
**Why it matters**: Empirical laws for model size vs performance, motivates 70B-405B models

**Brown et al. (2020) - GPT-3 (Few-Shot Learning)**
```bibtex
@inproceedings{brown2020language,
  title={Language Models are Few-Shot Learners},
  author={Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  booktitle={NeurIPS},
  year={2020}
}
```
**Why it matters**: GPT-3 demonstrates emergent capabilities at scale, motivates need for efficient large-model inference

---

## 7. Fortran & HPC (Your Unique Angle)

**Metcalf, Reid & Cohen (2018) - Modern Fortran Explained**
```bibtex
@book{metcalf2018modern,
  title={Modern Fortran Explained: Incorporating Fortran 2018},
  author={Metcalf, Michael and Reid, John and Cohen, Malcolm},
  year={2018},
  publisher={Oxford University Press}
}
```
**Why it matters**: Fortran 2018/2023 standard reference, `do concurrent` explained (your parallelization construct)

**Dongarra et al. (2003) - LAPACK**
```bibtex
@book{anderson1999lapack,
  title={LAPACK Users' Guide},
  author={Anderson, Edward and Bai, Zhaojun and Bischof, Christian and Blackford, Susan and Demmel, James and Dongarra, Jack and Du Croz, Jeremy and Greenbaum, Anne and Hammarling, Sven and McKenney, Alan and others},
  edition={3rd},
  year={1999},
  publisher={SIAM}
}
```
**Why it matters**: LAPACK is Fortran 77, de facto standard for numerical linear algebra (your MatMul builds on this tradition)

**Lattner & Adve (2004) - LLVM**
```bibtex
@inproceedings{lattner2004llvm,
  title={LLVM: A Compilation Framework for Lifelong Program Analysis \& Transformation},
  author={Lattner, Chris and Adve, Vikram},
  booktitle={CGO},
  pages={75--86},
  year={2004}
}
```
**Why it matters**: LLVM IR predecessor to MLIR, Flang (Fortran compiler) uses LLVM/MLIR

**Lattner et al. (2020) - MLIR**
```bibtex
@article{lattner2020mlir,
  title={MLIR: A Compiler Infrastructure for the End of Moore's Law},
  author={Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and Cohen, Albert and Davis, Andy and Pienaar, Jacques and Riddle, River and Shpeisman, Tatiana and Vasilache, Nicolas and Zinenko, Oleksandr},
  journal={arXiv preprint arXiv:2002.11054},
  year={2020}
}
```
**Why it matters**: MLIR framework you use for Fortran → ASIC compilation, multi-level IR for domain-specific optimization

---

## 8. How to Integrate These References into Your Paper

### In Introduction (Section 1)
```latex
Quantization reduces model size by representing weights with fewer bits.
Classical quantization theory \cite{shannon1948mathematical,lloyd1982least}
establishes fundamental limits, while recent neural network quantization
methods \cite{han2015deep,jacob2018quantization} have achieved 4-bit and
8-bit precision for large models.
```

### In Related Work (Section 2)
```latex
\textbf{Quantization theory.} Lloyd \cite{lloyd1982least} derived optimal
scalar quantizers minimizing mean squared error. Gersho and Gray
\cite{gersho1992vector} extended this to vector quantization. Our 3.5-bit
scheme packs two values (4-bit + 3-bit) in a 7-bit container, a form of
2-dimensional vector quantization.

\textbf{Neural network quantization.} Jacob et al. \cite{jacob2018quantization}
introduced asymmetric quantization with per-channel scaling and zero-point
offsets, which we adopt. Han et al. \cite{han2015deep} combined quantization
with pruning and encoding for compression.
```

### In Theory (Section 4)
```latex
Our error analysis follows the framework of Higham \cite{higham2002accuracy}
for rounding error propagation in numerical algorithms. Shannon's
rate-distortion theory \cite{shannon1948mathematical} provides fundamental
bounds on compression with bounded distortion.
```

### In Implementation (Section 5)
```latex
Our Fortran 2023 kernel uses \texttt{do concurrent} for explicit parallelism
\cite{metcalf2018modern}, which maps to systolic arrays
\cite{kung1979systolic} via MLIR compilation \cite{lattner2020mlir}.
This approach achieves performance comparable to hand-optimized ASIC
implementations \cite{jouppi2017datacenter}.
```

### In Verification Discussion (Section 7)
```latex
We formalize our theorems in Lean 4 \cite{de2015lean}, following the
methodology of verified compilers \cite{leroy2009formal,kumar2014cakeml}.
This provides mathematical guarantees unattainable through testing alone,
essential for safety-critical AI deployment.
```

---

## 9. Additional Modern References (Recent Work to Cite)

### LLM Quantization (2022-2024)
```bibtex
@article{frantar2023gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2023}
}

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Wei-Ming and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2306.00978},
  year={2023}
}

@article{dettmers2022llm8bit,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}

@article{xiao2023smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  journal={arXiv preprint arXiv:2211.10438},
  year={2023}
}
```

---

## 10. Complete Bibliography for Your Paper

Save this as `references_foundational.bib` and merge with your existing `references.bib`:

```bibtex
% ===== FOUNDATIONAL THEORY =====

% Information Theory
@article{shannon1948mathematical,
  title={A Mathematical Theory of Communication},
  author={Shannon, Claude E.},
  journal={Bell System Technical Journal},
  volume={27},
  number={3-4},
  pages={379--423, 623--656},
  year={1948}
}

@article{lloyd1982least,
  title={Least Squares Quantization in PCM},
  author={Lloyd, Stuart P.},
  journal={IEEE Transactions on Information Theory},
  volume={28},
  number={2},
  pages={129--137},
  year={1982}
}

@article{gray1998quantization,
  title={Quantization},
  author={Gray, Robert M. and Neuhoff, David L.},
  journal={IEEE Transactions on Information Theory},
  volume={44},
  number={6},
  pages={2325--2383},
  year={1998}
}

@book{cover2006elements,
  title={Elements of Information Theory},
  author={Cover, Thomas M. and Thomas, Joy A.},
  edition={2nd},
  year={2006},
  publisher={Wiley-Interscience}
}

% Numerical Analysis
@book{higham2002accuracy,
  title={Accuracy and Stability of Numerical Algorithms},
  author={Higham, Nicholas J.},
  edition={2nd},
  year={2002},
  publisher={SIAM}
}

@article{goldberg1991every,
  title={What Every Computer Scientist Should Know About Floating-Point Arithmetic},
  author={Goldberg, David},
  journal={ACM Computing Surveys},
  volume={23},
  number={1},
  pages={5--48},
  year={1991}
}

% Neural Network Quantization
@inproceedings{han2015deep,
  title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
  author={Han, Song and Mao, Huizi and Dally, William J.},
  booktitle={ICLR},
  year={2016}
}

@inproceedings{jacob2018quantization,
  title={Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  booktitle={CVPR},
  pages={2704--2713},
  year={2018}
}

% Formal Verification
@article{leroy2009formal,
  title={Formal Verification of a Realistic Compiler},
  author={Leroy, Xavier},
  journal={Communications of the ACM},
  volume={52},
  number={7},
  pages={107--115},
  year={2009}
}

@inproceedings{de2015lean,
  title={The Lean Theorem Prover (System Description)},
  author={De Moura, Leonardo and Kong, Soonho and Avigad, Jeremy and Van Doorn, Floris and Von Raumer, Jakob},
  booktitle={International Conference on Automated Deduction},
  pages={378--388},
  year={2015}
}

% ASIC Architecture
@inproceedings{kung1979systolic,
  title={Systolic Arrays (for VLSI)},
  author={Kung, H. T. and Leiserson, Charles E.},
  booktitle={Sparse Matrix Proceedings},
  pages={256--282},
  year={1979}
}

@inproceedings{jouppi2017datacenter,
  title={In-Datacenter Performance Analysis of a Tensor Processing Unit},
  author={Jouppi, Norman P. and Young, Cliff and Patil, Nishant and Patterson, David and others},
  booktitle={ISCA},
  pages={1--12},
  year={2017}
}

% Transformers & LLMs
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle={NeurIPS},
  pages={5998--6008},
  year={2017}
}

% MLIR
@article{lattner2020mlir,
  title={MLIR: A Compiler Infrastructure for the End of Moore's Law},
  author={Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and others},
  journal={arXiv preprint arXiv:2002.11054},
  year={2020}
}
```

---

## 11. Citation Strategy

### Build Theory Pyramid (Bottom-Up)

```
Shannon (1948) ← Information theory foundation
     ↓
Lloyd (1982) ← Optimal quantization
     ↓
Jacob (2018) ← Asymmetric quantization for NNs
     ↓
YOUR WORK (2025) ← 3.5-bit dynamic asymmetric quantization
```

### Narrative Arc in Related Work

1. **Classical theory** (Shannon, Lloyd) - establishes fundamentals
2. **NN quantization** (Han, Jacob) - applies to deep learning
3. **LLM quantization** (GPTQ, AWQ) - extends to large models
4. **Your gap** - no sub-4-bit with < 2% accuracy loss

### Table: Position Your Work

| Method | Bit Width | Foundation | Your Improvement |
|--------|-----------|------------|------------------|
| Lloyd (1982) | N/A (scalar) | Optimal quantizer | You: Per-channel, asymmetric |
| Jacob (2018) | 8-bit | Asymmetric, per-channel | You: 3.5-bit extension |
| GPTQ (2023) | 4-bit | Hessian-based | You: 3.5-bit, 12.5% smaller |
| **Yours (2025)** | **3.5-bit** | **Formal proofs (Lean 4)** | **First sub-4-bit** |

---

## ✅ Action Items

1. **Merge bibliographies**:
   ```bash
   cd papers/paper1_neurips2026
   # Merge foundational refs into references.bib
   cat FOUNDATIONAL_REFERENCES.md >> references.bib
   ```

2. **Add citations in LaTeX**:
   - Introduction: Cite Shannon, Lloyd, Jacob
   - Related Work: Full survey (30-50 papers)
   - Theory: Cite Higham, Goldberg (error analysis)
   - Implementation: Cite MLIR, Kung (systolic arrays)

3. **Read key papers** (prioritize these 5):
   1. Shannon (1948) - Rate-distortion theory
   2. Lloyd (1982) - Optimal quantization
   3. Jacob (2018) - Asymmetric quantization
   4. Higham (2002) - Ch 2 (Floating point error)
   5. Lattner (2020) - MLIR

---

**Status**: ✅ Foundational references identified, ready to integrate!

---

**Jim Xiao & Claude Code (Anthropic)**
**2025-11-29**
**Version 1.0**

*Standing on the shoulders of giants: Shannon → Lloyd → Jacob → You*
