#ifndef _TRANSFORMER_DEF_H_
#define _TRANSFORMER_DEF_H_

#include <gemm_exec.h>
/* #include <feedforward.h> */

#include <./../gemm_definitions/input_matrix.h>
#include <./../gemm_definitions/gemm_header_0.h>

#define TRANSFORMER_N_GEMM_LAYERS 1
#define TRANSFORMER_BITS_PER_CB   BITS_PER_CB

#define TRANSFORMER_SEQ_LEN     GEMM_M
#define TRANSFORMER_INPUT_SIZE  INPUT_SIZE_0
#define TRANSFORMER_OUTPUT_SIZE OUTPUT_SIZE_0

static const gemm_t transformer_gemm_0 = {
    .seq_len = TRANSFORMER_SEQ_LEN,
    .input_size = TRANSFORMER_INPUT_SIZE,
    .output_size = TRANSFORMER_OUTPUT_SIZE,
    .n_words_row = N_WORDS_ROW_0
};

/*
static const feedforward_t transformer_ffn = {
    .fc1 = transformer_gemm_0,
    .fc2 = {
        .seq_len = 0,
        .input_size = 0,
        .output_size = 0,
        .n_words_row = 0
    }
};
*/

#endif
