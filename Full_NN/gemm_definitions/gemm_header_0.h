#ifndef _GEMM_HEADER_0_H_
#define _GEMM_HEADER_0_H_


#include <inttypes.h>
#include <arm_sve.h>
#include <codebooks_def.h>

#define INPUT_SIZE_0      8
#define OUTPUT_SIZE_0     4

// Number of 32-bits words needed to store all the indexes
// #define N_WORDS_ROW_0 ((INPUT_SIZE_0) / (IDXS_PER_WORD))
#define N_WORDS_ROW_0 (((INPUT_SIZE_0) + (IDXS_PER_WORD) - 1) / (IDXS_PER_WORD))


static const int8_t codebooks_[N_LEARNERS][CB_SIZE] = {
	{
		0,
		0,
		-1,
		1,
		1,
		1,
		1,
		0,
	},
};



static const int8_t codebook_interleaved_[CB_SIZE * N_LEARNERS] = {
	0,
	0,
	-1,
	1,
	1,
	1,
	1,
	0,
};



static uint32_t weight_idx_compact_0[OUTPUT_SIZE_0 * N_WORDS_ROW_0] = {
	0b011110011010101110101101,
	0b000100110010100010000111,
	0b001001101011000011001110,
	0b011110011011001100001000,
};










static float bias_0[N_LEARNERS][OUTPUT_SIZE_0] = {
	{
		0.27381801866358396,
		0.47228078846901406,
		0.3887505167779042,
		0.26210622617823776,
	},
};


static float bias_interleaved_0[N_LEARNERS * OUTPUT_SIZE_0] = {
	0.27381801866358396,
	0.47228078846901406,
	0.3887505167779042,
	0.26210622617823776,
};

#endif
