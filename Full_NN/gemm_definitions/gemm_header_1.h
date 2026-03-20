#ifndef _GEMM_HEADER_1_H_
#define _GEMM_HEADER_1_H_


#include <inttypes.h>
#include <arm_sve.h>
#include <codebooks_def.h>

#define INPUT_SIZE_1      4
#define OUTPUT_SIZE_1     8

// Number of 32-bits words needed to store all the indexes
// #define N_WORDS_ROW_1 ((INPUT_SIZE_1) / (IDXS_PER_WORD))
#define N_WORDS_ROW_1 (((INPUT_SIZE_1) + (IDXS_PER_WORD) - 1) / (IDXS_PER_WORD))


static const int8_t codebooks_[N_LEARNERS][CB_SIZE] = {
	{
		-1,
		1,
		-1,
		-2,
		0,
		-1,
		-1,
		1,
	},
};



static const int8_t codebook_interleaved_[CB_SIZE * N_LEARNERS] = {
	-1,
	1,
	-1,
	-2,
	0,
	-1,
	-1,
	1,
};



static uint32_t weight_idx_compact_1[OUTPUT_SIZE_1 * N_WORDS_ROW_1] = {
	0b011101001001,
	0b110111110101,
	0b011110101111,
	0b100111101000,
	0b110001100111,
	0b000001111100,
	0b100011011011,
	0b100110100000,
};










static float bias_1[N_LEARNERS][OUTPUT_SIZE_1] = {
	{
		0.10983441136030095,
		0.44565921910014317,
		0.030949669003696556,
		0.8183883618709039,
		0.23290198344001523,
		0.5962700559185838,
		0.28053996848046986,
		0.46806121906002973,
	},
};


static float bias_interleaved_1[N_LEARNERS * OUTPUT_SIZE_1] = {
	0.10983441136030095,
	0.44565921910014317,
	0.030949669003696556,
	0.8183883618709039,
	0.23290198344001523,
	0.5962700559185838,
	0.28053996848046986,
	0.46806121906002973,
};

#endif
