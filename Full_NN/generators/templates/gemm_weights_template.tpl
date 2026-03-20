#ifndef _GEMM_HEADER_${ID}_H_
#define _GEMM_HEADER_${ID}_H_


#include <inttypes.h>
#include <arm_sve.h>
#include <codebooks_def.h>

#define INPUT_SIZE_${ID}      ${input_size}
#define OUTPUT_SIZE_${ID}     ${otuput_size}

// Number of 32-bits words needed to store all the indexes
// #define N_WORDS_ROW_${ID} ((INPUT_SIZE_${ID}) / (IDXS_PER_WORD))
#define N_WORDS_ROW_${ID} (((INPUT_SIZE_${ID}) + (IDXS_PER_WORD) - 1) / (IDXS_PER_WORD))


${codebook_string}


${codebook_string_interleaved}


${indexes_packed_string}


${indexes_packed_tiled_string}


${no_cb_weights_string}


${bias_strings}

#endif
