	.file	"mnist_inference_classify.c"
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"\n file opening failed "
.LC2:
	.string	","
	.text
	.globl	get_weights
	.type	get_weights, @function
get_weights:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$1112, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -1112(%rbp)
	movl	%esi, -1116(%rbp)
	movl	%edx, -1120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movl	$0, -1100(%rbp)
	movl	$0, -1096(%rbp)
	movq	-1112(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1080(%rbp)
	movl	-1116(%rbp), %eax
	movslq	%eax, %rdx
	movl	-1120(%rbp), %eax
	cltq
	imulq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -1072(%rbp)
	movl	$0, -1092(%rbp)
	jmp	.L2
.L3:
	movl	-1120(%rbp), %eax
	cltq
	salq	$2, %rax
	movl	-1092(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-1072(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, (%rbx)
	addl	$1, -1092(%rbp)
.L2:
	cmpl	$999, -1092(%rbp)
	jle	.L3
	cmpq	$0, -1080(%rbp)
	jne	.L5
	leaq	.LC1(%rip), %rdi
	movl	$0, %eax
	call	printf@PLT
	jmp	.L5
.L8:
	movq	-1064(%rbp), %rax
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -1088(%rbp)
	jmp	.L6
.L7:
	movq	-1088(%rbp), %rax
	movq	%rax, %rdi
	call	atof@PLT
	movl	-1100(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-1072(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-1096(%rbp), %eax
	leal	1(%rax), %edx
	movl	%edx, -1096(%rbp)
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	leaq	.LC2(%rip), %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -1088(%rbp)
.L6:
	cmpq	$0, -1088(%rbp)
	jne	.L7
	addl	$1, -1100(%rbp)
	movl	$0, -1096(%rbp)
.L5:
	movq	-1080(%rbp), %rdx
	leaq	-1056(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -1064(%rbp)
	cmpq	$0, -1064(%rbp)
	jne	.L8
	movq	-1072(%rbp), %rax
	movq	-24(%rbp), %rbx
	xorq	%fs:40, %rbx
	je	.L10
	call	__stack_chk_fail@PLT
.L10:
	addq	$1112, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	get_weights, .-get_weights
	.section	.rodata
.LC3:
	.string	";"
	.text
	.globl	get_biases
	.type	get_biases, @function
get_biases:
.LFB6:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1104, %rsp
	movq	%rdi, -1096(%rbp)
	movl	%esi, -1100(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -1076(%rbp)
	movl	-1100(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -1064(%rbp)
	movq	-1096(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1056(%rbp)
	jmp	.L12
.L15:
	movq	-1048(%rbp), %rax
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -1072(%rbp)
	jmp	.L13
.L14:
	movq	-1072(%rbp), %rax
	movq	%rax, %rdi
	call	atof@PLT
	movl	-1076(%rbp), %eax
	leal	1(%rax), %edx
	movl	%edx, -1076(%rbp)
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-1064(%rbp), %rax
	addq	%rdx, %rax
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	leaq	.LC3(%rip), %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -1072(%rbp)
.L13:
	cmpq	$0, -1072(%rbp)
	jne	.L14
.L12:
	movq	-1056(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -1048(%rbp)
	cmpq	$0, -1048(%rbp)
	jne	.L15
	movq	-1064(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L17
	call	__stack_chk_fail@PLT
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	get_biases, .-get_biases
	.section	.rodata
.LC7:
	.string	"Oh no x = %f\n"
	.text
	.globl	quantize_value
	.type	quantize_value, @function
quantize_value:
.LFB7:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movss	%xmm2, -28(%rbp)
	movl	%edi, -32(%rbp)
	movss	-24(%rbp), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jbe	.L33
	movss	-24(%rbp), %xmm0
	jmp	.L18
.L33:
	cvtsi2ss	-32(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	ucomiss	%xmm0, %xmm1
	jbe	.L34
	cvtsi2ss	-32(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	jmp	.L18
.L34:
	movl	$0, -12(%rbp)
	jmp	.L24
.L28:
	cvtsi2ss	-12(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -8(%rbp)
	movss	-28(%rbp), %xmm0
	movss	.LC4(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	-8(%rbp), %xmm1
	subss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1
	ucomiss	%xmm0, %xmm1
	jbe	.L25
	movss	-28(%rbp), %xmm0
	movss	.LC4(%rip), %xmm1
	divss	%xmm1, %xmm0
	addss	-8(%rbp), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jbe	.L25
	movss	-8(%rbp), %xmm1
	movss	.LC5(%rip), %xmm0
	mulss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	call	round@PLT
	movapd	%xmm0, %xmm1
	movsd	.LC6(%rip), %xmm0
	divsd	%xmm0, %xmm1
	movapd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, -4(%rbp)
	movss	-4(%rbp), %xmm0
	jmp	.L18
.L25:
	addl	$1, -12(%rbp)
.L24:
	movl	-12(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jl	.L28
	cvtss2sd	-20(%rbp), %xmm0
	leaq	.LC7(%rip), %rdi
	movl	$1, %eax
	call	printf@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	quantize_value, .-quantize_value
	.globl	get_quantize_step
	.type	get_quantize_step, @function
get_quantize_step:
.LFB8:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movss	%xmm2, -28(%rbp)
	movl	%edi, -32(%rbp)
	movss	-24(%rbp), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jbe	.L50
	movl	$0, %eax
	jmp	.L35
.L50:
	cvtsi2ss	-32(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	ucomiss	%xmm0, %xmm1
	jbe	.L51
	movl	-32(%rbp), %eax
	jmp	.L35
.L51:
	movl	$0, -8(%rbp)
	jmp	.L41
.L45:
	cvtsi2ss	-8(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -4(%rbp)
	movss	-28(%rbp), %xmm0
	movss	.LC4(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	-4(%rbp), %xmm1
	subss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1
	ucomiss	%xmm0, %xmm1
	jbe	.L42
	movss	-28(%rbp), %xmm0
	movss	.LC4(%rip), %xmm1
	divss	%xmm1, %xmm0
	addss	-4(%rbp), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jbe	.L42
	movl	-8(%rbp), %eax
	jmp	.L35
.L42:
	addl	$1, -8(%rbp)
.L41:
	movl	-8(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jl	.L45
	cvtss2sd	-20(%rbp), %xmm0
	leaq	.LC7(%rip), %rdi
	movl	$1, %eax
	call	printf@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	get_quantize_step, .-get_quantize_step
	.globl	calc_activations
	.type	calc_activations, @function
calc_activations:
.LFB9:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	%ecx, -60(%rbp)
	movl	-60(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movl	$0, -20(%rbp)
	jmp	.L53
.L56:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -16(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L54
.L55:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm1, %xmm0
	movss	-16(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -16(%rbp)
	addl	$1, -12(%rbp)
.L54:
	cmpl	$783, -12(%rbp)
	jle	.L55
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-16(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -16(%rbp)
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movss	-16(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -20(%rbp)
.L53:
	movl	-20(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L56
	movq	-8(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	calc_activations, .-calc_activations
	.globl	calc_multiplication
	.type	calc_multiplication, @function
calc_multiplication:
.LFB10:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	$0, -4(%rbp)
	cmpl	$15, -20(%rbp)
	ja	.L59
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L61(%rip), %rax
	movl	(%rdx,%rax), %eax
	movslq	%eax, %rdx
	leaq	.L61(%rip), %rax
	addq	%rdx, %rax
	jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L61:
	.long	.L60-.L61
	.long	.L62-.L61
	.long	.L63-.L61
	.long	.L64-.L61
	.long	.L65-.L61
	.long	.L66-.L61
	.long	.L67-.L61
	.long	.L68-.L61
	.long	.L69-.L61
	.long	.L70-.L61
	.long	.L71-.L61
	.long	.L72-.L61
	.long	.L73-.L61
	.long	.L74-.L61
	.long	.L75-.L61
	.long	.L76-.L61
	.text
.L60:
	pxor	%xmm0, %xmm0
	jmp	.L58
.L62:
	cvtsi2ss	-24(%rbp), %xmm0
	jmp	.L58
.L63:
	movl	-24(%rbp), %eax
	addl	%eax, %eax
	movl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L64:
	movl	-24(%rbp), %eax
	addl	%eax, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L65:
	movl	-24(%rbp), %eax
	sall	$2, %eax
	movl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L66:
	movl	-24(%rbp), %eax
	sall	$2, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L67:
	movl	-24(%rbp), %eax
	addl	%eax, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	sall	-4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L68:
	movl	-24(%rbp), %eax
	sall	$3, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L69:
	movl	-24(%rbp), %eax
	sall	$3, %eax
	movl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L70:
	movl	-24(%rbp), %eax
	sall	$3, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L71:
	movl	-24(%rbp), %eax
	sall	$2, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	sall	-4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L72:
	movl	-24(%rbp), %eax
	sall	$3, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L73:
	movl	-24(%rbp), %eax
	addl	%eax, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -4(%rbp)
	sall	$2, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L74:
	movl	-24(%rbp), %eax
	sall	$4, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L75:
	movl	-24(%rbp), %eax
	sall	$4, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L76:
	movl	-24(%rbp), %eax
	sall	$4, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	subl	%eax, -4(%rbp)
	cvtsi2ss	-4(%rbp), %xmm0
	jmp	.L58
.L59:
.L58:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	calc_multiplication, .-calc_multiplication
	.globl	fixed_point_quantize
	.type	fixed_point_quantize, @function
fixed_point_quantize:
.LFB11:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -4(%rbp)
	cvtss2sd	-4(%rbp), %xmm0
	movsd	.LC9(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	jbe	.L89
	movl	$14, %eax
	jmp	.L81
.L89:
	cvtss2sd	-4(%rbp), %xmm0
	movsd	.LC10(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	jbe	.L90
	movl	$-1, %eax
	jmp	.L81
.L90:
	cvtss2sd	-4(%rbp), %xmm0
	ucomisd	.LC11(%rip), %xmm0
	jbe	.L91
	movl	$1, %eax
	jmp	.L81
.L91:
	movl	$0, %eax
.L81:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	fixed_point_quantize, .-fixed_point_quantize
	.globl	calc_Q
	.type	calc_Q, @function
calc_Q:
.LFB12:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movss	%xmm0, -52(%rbp)
	movq	%rdi, -64(%rbp)
	movl	%esi, -56(%rbp)
	movl	-56(%rbp), %eax
	cltq
	imulq	$3136, %rax, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movl	$0, -40(%rbp)
	jmp	.L93
.L96:
	movl	-56(%rbp), %eax
	cltq
	salq	$2, %rax
	movl	-40(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-24(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, (%rbx)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -32(%rbp)
	movl	$0, -36(%rbp)
	jmp	.L94
.L95:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-36(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-52(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -68(%rbp)
	movss	-68(%rbp), %xmm0
	call	fixed_point_quantize
	sall	$4, %eax
	sarb	$4, %al
	andl	$15, %eax
	movl	%eax, %edx
	movzbl	-44(%rbp), %eax
	andl	$-16, %eax
	orl	%edx, %eax
	movb	%al, -44(%rbp)
	movzbl	-44(%rbp), %eax
	sall	$4, %eax
	sarb	$4, %al
	movl	-40(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rdx), %rdx
	movl	-36(%rbp), %ecx
	movslq	%ecx, %rcx
	salq	$2, %rcx
	addq	%rcx, %rdx
	movsbl	%al, %eax
	movl	%eax, (%rdx)
	addl	$1, -36(%rbp)
.L94:
	movl	-36(%rbp), %eax
	cmpl	-56(%rbp), %eax
	jl	.L95
	addl	$1, -40(%rbp)
.L93:
	cmpl	$783, -40(%rbp)
	jle	.L96
	movq	-24(%rbp), %rax
	addq	$72, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	calc_Q, .-calc_Q
	.globl	calc_activations_fixed_point
	.type	calc_activations_fixed_point, @function
calc_activations_fixed_point:
.LFB13:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	%ecx, -60(%rbp)
	movl	-60(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movl	$0, -24(%rbp)
	jmp	.L99
.L102:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	jmp	.L100
.L101:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	calc_multiplication
	movaps	%xmm0, %xmm1
	cvtsi2ss	-20(%rbp), %xmm0
	addss	%xmm1, %xmm0
	cvttss2si	%xmm0, %eax
	movl	%eax, -20(%rbp)
	addl	$1, -16(%rbp)
.L100:
	cmpl	$783, -16(%rbp)
	jle	.L101
	cvtsi2ss	-20(%rbp), %xmm0
	movss	%xmm0, -12(%rbp)
	movss	-12(%rbp), %xmm0
	movss	.LC12(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -12(%rbp)
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-12(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -12(%rbp)
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movss	-12(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -24(%rbp)
.L99:
	movl	-24(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L102
	movq	-8(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	calc_activations_fixed_point, .-calc_activations_fixed_point
	.section	.rodata
	.align 8
.LC13:
	.string	"../input_reshaped_quantized.csv"
.LC14:
	.string	"../biases.csv"
.LC15:
	.string	"../weights.csv"
	.align 8
.LC17:
	.string	"Normal Activation:\t\tFixed Point Multiplication Free Activation: "
.LC18:
	.string	"%f\t\t\t%f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB14:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$104, %rsp
	.cfi_offset 3, -24
	movl	$10, -76(%rbp)
	movl	$784, %esi
	leaq	.LC13(%rip), %rdi
	call	get_biases
	movq	%rax, -72(%rbp)
	movl	$10, %esi
	leaq	.LC14(%rip), %rdi
	call	get_biases
	movq	%rax, -64(%rbp)
	movl	$10, %edx
	movl	$784, %esi
	leaq	.LC15(%rip), %rdi
	call	get_weights
	movq	%rax, -56(%rbp)
	movl	-76(%rbp), %ecx
	movq	-56(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	calc_activations
	movq	%rax, -48(%rbp)
	movl	$3136, %edi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movl	$0, -84(%rbp)
	jmp	.L105
.L106:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	-84(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movl	$16, %edi
	movss	.LC16(%rip), %xmm2
	pxor	%xmm1, %xmm1
	movl	%eax, -100(%rbp)
	movss	-100(%rbp), %xmm0
	call	get_quantize_step
	movl	%eax, (%rbx)
	addl	$1, -84(%rbp)
.L105:
	cmpl	$783, -84(%rbp)
	jle	.L106
	movq	-56(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	movss	.LC16(%rip), %xmm0
	call	calc_Q
	movq	%rax, -32(%rbp)
	movl	-76(%rbp), %ecx
	movq	-64(%rbp), %rdx
	movq	-32(%rbp), %rsi
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	calc_activations_fixed_point
	movq	%rax, -24(%rbp)
	leaq	.LC17(%rip), %rdi
	call	puts@PLT
	movl	$0, -80(%rbp)
	jmp	.L107
.L108:
	movl	-80(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	cvtss2sd	%xmm0, %xmm1
	movl	-80(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	cvtss2sd	%xmm0, %xmm0
	leaq	.LC18(%rip), %rdi
	movl	$2, %eax
	call	printf@PLT
	addl	$1, -80(%rbp)
.L107:
	cmpl	$9, -80(%rbp)
	jle	.L108
	movl	$0, %eax
	addq	$104, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC4:
	.long	1073741824
	.align 4
.LC5:
	.long	1203982336
	.align 8
.LC6:
	.long	0
	.long	1090021888
	.align 8
.LC9:
	.long	1992864825
	.long	-1080385602
	.align 8
.LC10:
	.long	1202590843
	.long	-1081836831
	.align 8
.LC11:
	.long	1202590843
	.long	1065646817
	.align 4
.LC12:
	.long	1115684864
	.align 4
.LC16:
	.long	1032358069
	.ident	"GCC: (Ubuntu 7.2.0-8ubuntu3.2) 7.2.0"
	.section	.note.GNU-stack,"",@progbits
