#ifndef _GEMM_EXEC_H_
#define _GEMM_EXEC_H_

#include <inttypes.h>
#include <stdint.h>


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

void gemm_exec_noCB_int(gemm_t gemm_layer,
                        const int8_t *in,
                        const int8_t *weights,
                        const int32_t *bias,
                        int32_t *out);

void gemm_exec_compact_int(gemm_t gemm_layer,
                           const int8_t *in,
                           const uint32_t *weight_idx,
                           const int8_t *codebook,
                           const int32_t *bias,
                           int32_t *out,
                           uint8_t bits_per_cb);

void gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb);

#ifdef SIMD
void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb);

void gemm_exec_compact_int_sve_interleaved_4D_diff_seq(gemm_t gemm_layer,
                                                       const int8_t *in_interleaved,
                                                       const uint32_t *weight_idx_interleaved,
                                                       const int8_t *codebook_interleaved,
                                                       const int32_t *bias_interleaved,
                                                       int32_t *out_interleaved,
                                                       uint8_t bits_per_cb);
#endif

#endif
