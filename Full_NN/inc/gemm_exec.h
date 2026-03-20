#ifndef _GEMM_EXEC_H_
#define _GEMM_EXEC_H_

#include <inttypes.h>

typedef struct gemm_struct {
    uint16_t seq_len;
    uint16_t input_size;
    uint16_t output_size;

    uint16_t n_words_row;   // Number of packed-index words per output row
} gemm_t;

void gemm_exec_noCB(gemm_t gemm_layer,
                    const float *in,
                    const float *weights,
                    const float *bias,
                    float *out);

void gemm_exec_compact(gemm_t gemm_layer,
                       const float *in,
                       const uint32_t *weight_idx,
                       const float *codebook,
                       const float *bias,
                       float *out,
                       uint8_t bits_per_cb);

#endif
