#ifndef _GEMM_DATA_H_
#define _GEMM_DATA_H_

#include <gemm_exec.h>

#include <./input_matrix.h>
#include <./gemm_header_0.h>
#include <./gemm_header_1.h>
#include <./gemm_header_2.h>
#include <./gemm_header_3.h>
#include <./gemm_header_4.h>
#include <./gemm_header_5.h>
#include <./gemm_header_6.h>
#include <./gemm_header_7.h>
#include <./gemm_header_8.h>
#include <./gemm_header_9.h>
#include <./gemm_header_10.h>
#include <./gemm_header_11.h>
#include <./gemm_header_12.h>
#include <./gemm_header_13.h>
#include <./gemm_header_14.h>
#include <./gemm_header_15.h>
#include <./gemm_header_16.h>
#include <./gemm_header_17.h>
#include <./gemm_header_18.h>
#include <./gemm_header_19.h>
#include <./gemm_header_20.h>
#include <./gemm_header_21.h>
#include <./gemm_header_22.h>
#include <./gemm_header_23.h>
#include <./gemm_header_24.h>
#include <./gemm_header_25.h>
#include <./gemm_header_26.h>

#define N_GEMM_LAYERS 27

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

static const gemm_t gemm_2 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_2,
	.output_size = OUTPUT_SIZE_2,
	.n_words_row = N_WORDS_ROW_2
};

static const gemm_t gemm_3 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_3,
	.output_size = OUTPUT_SIZE_3,
	.n_words_row = N_WORDS_ROW_3
};

static const gemm_t gemm_4 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_4,
	.output_size = OUTPUT_SIZE_4,
	.n_words_row = N_WORDS_ROW_4
};

static const gemm_t gemm_5 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_5,
	.output_size = OUTPUT_SIZE_5,
	.n_words_row = N_WORDS_ROW_5
};

static const gemm_t gemm_6 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_6,
	.output_size = OUTPUT_SIZE_6,
	.n_words_row = N_WORDS_ROW_6
};

static const gemm_t gemm_7 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_7,
	.output_size = OUTPUT_SIZE_7,
	.n_words_row = N_WORDS_ROW_7
};

static const gemm_t gemm_8 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_8,
	.output_size = OUTPUT_SIZE_8,
	.n_words_row = N_WORDS_ROW_8
};

static const gemm_t gemm_9 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_9,
	.output_size = OUTPUT_SIZE_9,
	.n_words_row = N_WORDS_ROW_9
};

static const gemm_t gemm_10 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_10,
	.output_size = OUTPUT_SIZE_10,
	.n_words_row = N_WORDS_ROW_10
};

static const gemm_t gemm_11 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_11,
	.output_size = OUTPUT_SIZE_11,
	.n_words_row = N_WORDS_ROW_11
};

static const gemm_t gemm_12 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_12,
	.output_size = OUTPUT_SIZE_12,
	.n_words_row = N_WORDS_ROW_12
};

static const gemm_t gemm_13 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_13,
	.output_size = OUTPUT_SIZE_13,
	.n_words_row = N_WORDS_ROW_13
};

static const gemm_t gemm_14 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_14,
	.output_size = OUTPUT_SIZE_14,
	.n_words_row = N_WORDS_ROW_14
};

static const gemm_t gemm_15 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_15,
	.output_size = OUTPUT_SIZE_15,
	.n_words_row = N_WORDS_ROW_15
};

static const gemm_t gemm_16 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_16,
	.output_size = OUTPUT_SIZE_16,
	.n_words_row = N_WORDS_ROW_16
};

static const gemm_t gemm_17 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_17,
	.output_size = OUTPUT_SIZE_17,
	.n_words_row = N_WORDS_ROW_17
};

static const gemm_t gemm_18 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_18,
	.output_size = OUTPUT_SIZE_18,
	.n_words_row = N_WORDS_ROW_18
};

static const gemm_t gemm_19 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_19,
	.output_size = OUTPUT_SIZE_19,
	.n_words_row = N_WORDS_ROW_19
};

static const gemm_t gemm_20 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_20,
	.output_size = OUTPUT_SIZE_20,
	.n_words_row = N_WORDS_ROW_20
};

static const gemm_t gemm_21 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_21,
	.output_size = OUTPUT_SIZE_21,
	.n_words_row = N_WORDS_ROW_21
};

static const gemm_t gemm_22 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_22,
	.output_size = OUTPUT_SIZE_22,
	.n_words_row = N_WORDS_ROW_22
};

static const gemm_t gemm_23 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_23,
	.output_size = OUTPUT_SIZE_23,
	.n_words_row = N_WORDS_ROW_23
};

static const gemm_t gemm_24 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_24,
	.output_size = OUTPUT_SIZE_24,
	.n_words_row = N_WORDS_ROW_24
};

static const gemm_t gemm_25 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_25,
	.output_size = OUTPUT_SIZE_25,
	.n_words_row = N_WORDS_ROW_25
};

static const gemm_t gemm_26 = {
	.seq_len = GEMM_M,
	.input_size = INPUT_SIZE_26,
	.output_size = OUTPUT_SIZE_26,
	.n_words_row = N_WORDS_ROW_26
};

#endif
