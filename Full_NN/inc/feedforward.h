#ifndef _FEEDFORWARD_H_
#define _FEEDFORWARD_H_

#include <gemm_exec.h>

typedef struct feedforward_struct {
    gemm_t fc1;
    gemm_t fc2;
} feedforward_t;

void feedforward_exec_noCB(feedforward_t ff_layer,
                           const float *in,
                           const float *weights_0,
                           const float *bias_0,
                           const float *weights_1,
                           const float *bias_1,
                           float *hidden,
                           float *out);

void feedforward_exec_compact(feedforward_t ff_layer,
                              const float *in,
                              const uint32_t *weight_idx_0,
                              const float *codebook_0,
                              const float *bias_0,
                              const uint32_t *weight_idx_1,
                              const float *codebook_1,
                              const float *bias_1,
                              float *hidden,
                              float *out,
                              uint8_t bits_per_cb);

#endif
