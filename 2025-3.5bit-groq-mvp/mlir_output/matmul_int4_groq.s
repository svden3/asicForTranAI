	.build_version macos,  26, 0
	.text
	.p2align 4
	.globl ___matmul_int4_groq_MOD_dequantize_output
___matmul_int4_groq_MOD_dequantize_output:
LFB0:
	movl	(%r8), %eax
	testl	%eax, %eax
	jle	L20
	movl	(%rcx), %r10d
	testl	%r10d, %r10d
	jle	L20
	pushq	%r15
LCFI0:
	movq	$-1, %r9
	pushq	%r14
LCFI1:
	pushq	%r13
LCFI2:
	movl	%r10d, %r13d
	pushq	%r12
LCFI3:
	movslq	%r10d, %r12
	pushq	%rbp
LCFI4:
	movq	%rdx, %rbp
	xorl	%edx, %edx
	testq	%r12, %r12
	cmovs	%rdx, %r12
	subl	$1, %eax
	pushq	%rbx
LCFI5:
	movq	%rdi, %rbx
	movq	%rsi, %rdi
	movq	%rbx, %rcx
	leaq	4(%rsi,%rax,4), %r14
	movl	%r10d, %esi
	movq	%rbp, %rdx
	leaq	0(,%r12,4), %r11
	shrl	$2, %esi
	andl	$-4, %r13d
	salq	$4, %rsi
	.p2align 4,,10
	.p2align 3
L9:
	cmpl	$3, %r10d
	movss	(%rdi), %xmm2
	jle	L10
L6:
	movaps	%xmm2, %xmm1
	xorl	%eax, %eax
	shufps	$0, %xmm1, %xmm1
	.p2align 5
	.p2align 4,,10
	.p2align 3
L4:
	movdqu	(%rcx,%rax), %xmm3
	cvtdq2ps	%xmm3, %xmm0
	mulps	%xmm1, %xmm0
	movlps	%xmm0, (%rdx,%rax)
	movhps	%xmm0, 8(%rdx,%rax)
	addq	$16, %rax
	cmpq	%rax, %rsi
	jne	L4
	testb	$3, %r10b
	je	L5
	leal	1(%r13), %eax
	movl	%r13d, %r8d
L3:
	movl	%r10d, %r15d
	subl	%r8d, %r15d
	cmpl	$1, %r15d
	je	L7
	movsldup	%xmm2, %xmm1
	leaq	1(%r9,%r8), %r8
	testb	$1, %r15b
	movq	(%rbx,%r8,4), %xmm0
	cvtdq2ps	%xmm0, %xmm0
	movq	%xmm0, %xmm0
	movq	%xmm1, %xmm1
	mulps	%xmm1, %xmm0
	movlps	%xmm0, 0(%rbp,%r8,4)
	je	L8
	andl	$-2, %r15d
	addl	%r15d, %eax
L7:
	cltq
	pxor	%xmm0, %xmm0
	addq	%r9, %rax
	cvtsi2ssl	(%rbx,%rax,4), %xmm0
	mulss	%xmm2, %xmm0
	movss	%xmm0, 0(%rbp,%rax,4)
L8:
	addq	$4, %rdi
	addq	%r11, %rcx
	addq	%r11, %rdx
	addq	%r12, %r9
	cmpq	%r14, %rdi
	jne	L9
L1:
	popq	%rbx
LCFI6:
	popq	%rbp
LCFI7:
	popq	%r12
LCFI8:
	popq	%r13
LCFI9:
	popq	%r14
LCFI10:
	popq	%r15
LCFI11:
	ret
	.p2align 4,,10
	.p2align 3
L5:
LCFI12:
	addq	$4, %rdi
	cmpq	%r14, %rdi
	je	L1
	movss	(%rdi), %xmm2
	addq	%r11, %rcx
	addq	%r11, %rdx
	addq	%r12, %r9
	jmp	L6
L10:
	xorl	%r8d, %r8d
	movl	$1, %eax
	jmp	L3
L20:
LCFI13:
	ret
LFE0:
	.p2align 4
	.globl ___matmul_int4_groq_MOD_matmul_int4_awq
___matmul_int4_groq_MOD_matmul_int4_awq:
LFB1:
	pushq	%r15
LCFI14:
	pushq	%r14
LCFI15:
	pushq	%r13
LCFI16:
	pushq	%r12
LCFI17:
	movq	%rcx, %r12
	pushq	%rbp
LCFI18:
	pushq	%rbx
LCFI19:
	movl	(%r9), %ecx
	testl	%ecx, %ecx
	jle	L24
	movq	%rdi, %r10
	movq	%rsi, %rdi
	movl	(%r8), %esi
	testl	%esi, %esi
	jle	L24
	movq	56(%rsp), %rax
	leal	1(%rsi), %r13d
	movslq	%esi, %rbp
	movl	%ecx, -20(%rsp)
	leaq	-1(%r10), %r14
	movq	%r13, -32(%rsp)
	movl	(%rax), %r9d
	xorl	%eax, %eax
	testq	%rbp, %rbp
	cmovs	%rax, %rbp
	subq	$4, %r12
	leaq	0(,%rbp,4), %r15
	leal	7(%r9), %eax
	testl	%r9d, %r9d
	movq	%r15, -8(%rsp)
	cmovns	%r9d, %eax
	leal	-1(%r9), %r11d
	sarl	$3, %eax
	leaq	(%rbp,%rbp), %rbx
	movslq	%eax, %rdx
	xorl	%eax, %eax
	testq	%rdx, %rdx
	cmovs	%rax, %rdx
	andl	$-2, %r11d
	movq	$-1, %rax
	addl	$3, %r11d
	movq	%rdx, -16(%rsp)
	.p2align 4,,10
	.p2align 3
L29:
	testl	%r9d, %r9d
	movl	$1, %esi
	movl	$0, 4(%r12)
	jle	L35
	leaq	(%rdi,%rax), %r10
	movq	%rax, %rdx
	.p2align 4,,10
	.p2align 3
L28:
	leaq	(%r14,%rsi), %r8
	xorl	%r15d, %r15d
	movl	$1, %eax
	movq	%rsi, -40(%rsp)
	.p2align 4,,10
	.p2align 3
L34:
	leal	1(%rax), %ecx
	sarl	%ecx
	movslq	%ecx, %rcx
	movzbl	(%r10,%rcx), %ecx
	movl	%ecx, %r13d
	movl	%ecx, %esi
	orl	$-16, %esi
	andl	$15, %r13d
	testb	$8, %cl
	cmovne	%esi, %r13d
	movsbl	(%r8), %esi
	imull	%r13d, %esi
	addl	%esi, %r15d
	cmpl	%eax, %r9d
	jle	L32
	movl	%ecx, %esi
	shrl	$4, %esi
	movl	%esi, %r13d
	orl	$-16, %esi
	andl	$15, %r13d
	testb	%cl, %cl
	movsbl	(%r8,%rbp), %ecx
	cmovs	%esi, %r13d
	imull	%r13d, %ecx
	addl	%ecx, %r15d
L32:
	addl	$2, %eax
	addq	%rbx, %r8
	cmpl	%r11d, %eax
	jne	L34
	movq	-40(%rsp), %rsi
	movl	%r15d, (%r12,%rsi,4)
	addq	$1, %rsi
	cmpq	%rsi, -32(%rsp)
	je	L44
	movl	$0, (%r12,%rsi,4)
	jmp	L28
	.p2align 4,,10
	.p2align 3
L44:
	movq	%rdx, %rax
L30:
	addq	-8(%rsp), %r12
	addq	-16(%rsp), %rax
	subl	$1, -20(%rsp)
	jne	L29
L24:
	popq	%rbx
LCFI20:
	popq	%rbp
LCFI21:
	popq	%r12
LCFI22:
	popq	%r13
LCFI23:
	popq	%r14
LCFI24:
	popq	%r15
LCFI25:
	ret
L35:
LCFI26:
	movq	-32(%rsp), %rdx
	leaq	1(%rsi), %r8
	cmpq	%r8, %rdx
	je	L30
L46:
	addq	$2, %rsi
	movl	$0, (%r12,%r8,4)
	cmpq	%rdx, %rsi
	je	L30
	leaq	1(%rsi), %r8
	movl	$0, (%r12,%rsi,4)
	cmpq	%r8, %rdx
	jne	L46
	jmp	L30
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x10
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x7
	.uleb128 0x8
	.byte	0x90
	.uleb128 0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB0-.
	.set L$set$2,LFE0-LFB0
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB0
	.long L$set$3
	.byte	0xe
	.uleb128 0x10
	.byte	0x8f
	.uleb128 0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xe
	.uleb128 0x18
	.byte	0x8e
	.uleb128 0x3
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xe
	.uleb128 0x20
	.byte	0x8d
	.uleb128 0x4
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xe
	.uleb128 0x28
	.byte	0x8c
	.uleb128 0x5
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xe
	.uleb128 0x30
	.byte	0x86
	.uleb128 0x6
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xe
	.uleb128 0x38
	.byte	0x83
	.uleb128 0x7
	.byte	0x4
	.set L$set$9,LCFI6-LCFI5
	.long L$set$9
	.byte	0xa
	.byte	0xe
	.uleb128 0x30
	.byte	0x4
	.set L$set$10,LCFI7-LCFI6
	.long L$set$10
	.byte	0xe
	.uleb128 0x28
	.byte	0x4
	.set L$set$11,LCFI8-LCFI7
	.long L$set$11
	.byte	0xe
	.uleb128 0x20
	.byte	0x4
	.set L$set$12,LCFI9-LCFI8
	.long L$set$12
	.byte	0xe
	.uleb128 0x18
	.byte	0x4
	.set L$set$13,LCFI10-LCFI9
	.long L$set$13
	.byte	0xe
	.uleb128 0x10
	.byte	0x4
	.set L$set$14,LCFI11-LCFI10
	.long L$set$14
	.byte	0xe
	.uleb128 0x8
	.byte	0x4
	.set L$set$15,LCFI12-LCFI11
	.long L$set$15
	.byte	0xb
	.byte	0x4
	.set L$set$16,LCFI13-LCFI12
	.long L$set$16
	.byte	0xe
	.uleb128 0x8
	.byte	0xc3
	.byte	0xc6
	.byte	0xcc
	.byte	0xcd
	.byte	0xce
	.byte	0xcf
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$17,LEFDE3-LASFDE3
	.long L$set$17
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$18,LFE1-LFB1
	.quad L$set$18
	.uleb128 0
	.byte	0x4
	.set L$set$19,LCFI14-LFB1
	.long L$set$19
	.byte	0xe
	.uleb128 0x10
	.byte	0x8f
	.uleb128 0x2
	.byte	0x4
	.set L$set$20,LCFI15-LCFI14
	.long L$set$20
	.byte	0xe
	.uleb128 0x18
	.byte	0x8e
	.uleb128 0x3
	.byte	0x4
	.set L$set$21,LCFI16-LCFI15
	.long L$set$21
	.byte	0xe
	.uleb128 0x20
	.byte	0x8d
	.uleb128 0x4
	.byte	0x4
	.set L$set$22,LCFI17-LCFI16
	.long L$set$22
	.byte	0xe
	.uleb128 0x28
	.byte	0x8c
	.uleb128 0x5
	.byte	0x4
	.set L$set$23,LCFI18-LCFI17
	.long L$set$23
	.byte	0xe
	.uleb128 0x30
	.byte	0x86
	.uleb128 0x6
	.byte	0x4
	.set L$set$24,LCFI19-LCFI18
	.long L$set$24
	.byte	0xe
	.uleb128 0x38
	.byte	0x83
	.uleb128 0x7
	.byte	0x4
	.set L$set$25,LCFI20-LCFI19
	.long L$set$25
	.byte	0xa
	.byte	0xe
	.uleb128 0x30
	.byte	0x4
	.set L$set$26,LCFI21-LCFI20
	.long L$set$26
	.byte	0xe
	.uleb128 0x28
	.byte	0x4
	.set L$set$27,LCFI22-LCFI21
	.long L$set$27
	.byte	0xe
	.uleb128 0x20
	.byte	0x4
	.set L$set$28,LCFI23-LCFI22
	.long L$set$28
	.byte	0xe
	.uleb128 0x18
	.byte	0x4
	.set L$set$29,LCFI24-LCFI23
	.long L$set$29
	.byte	0xe
	.uleb128 0x10
	.byte	0x4
	.set L$set$30,LCFI25-LCFI24
	.long L$set$30
	.byte	0xe
	.uleb128 0x8
	.byte	0x4
	.set L$set$31,LCFI26-LCFI25
	.long L$set$31
	.byte	0xb
	.align 3
LEFDE3:
	.ident	"GCC: (Homebrew GCC 15.2.0) 15.2.0"
	.subsections_via_symbols
