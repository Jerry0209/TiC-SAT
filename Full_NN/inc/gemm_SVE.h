#ifndef _GEMM_SVE_H_
#define _GEMM_SVE_H_

#include <stdint.h>

#include <gemm_exec.h>

#ifdef __cplusplus
extern "C" {
#endif

void sve_gemm_row_compact_int8(const uint32_t *packed_row,
                               uint32_t n_words_row,
                               uint32_t k_elems,
                               const int8_t *in_mat,
                               uint32_t seq_tile,
                               uint32_t ld_in,
                               const int32_t *codebook_i32,
                               int32_t *out_mat,
                               uint32_t out_col,
                               uint32_t ld_out,
                               int32_t bias_val,
                               int add_bias,
                               int accumulate,
                               uint8_t bits_per_cb);

#ifdef __cplusplus
}
#endif

#endif
