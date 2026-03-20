#ifndef _GEMM_DATA_H_
#define _GEMM_DATA_H_

#include <gemm_exec.h>

#include <./input_matrix.h>
#include <./gemm_header_0.h>
#include <./gemm_header_1.h>

#define N_GEMM_LAYERS 2

static const gemm_t gemm_0 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_0,
	.output_size = OUTPUT_SIZE_0,
	.n_words_row = N_WORDS_ROW_0
};

static const gemm_t gemm_1 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_1,
	.output_size = OUTPUT_SIZE_1,
	.n_words_row = N_WORDS_ROW_1
};

#endif
