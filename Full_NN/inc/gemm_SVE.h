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

void sve_gemm_row_compact_fp32(const uint32_t *packed_row,
                               uint32_t n_words_row,
                               uint32_t k_elems,
                               const float *in_mat,
                               uint32_t seq_tile,
                               uint32_t ld_in,
                               const float *codebook,
                               float *out_mat,
                               uint32_t out_col,
                               uint32_t ld_out,
                               float bias_val,
                               int add_bias,
                               int accumulate,
                               uint8_t bits_per_cb);

void sve_gemm_row_compact_fp32_interleaved_4D_diff_seq(
    const uint32_t *packed_rows_interleaved,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_row_compact_fp32_interleaved_2D_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_row_compact_fp32_interleaved_4D_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_row_compact_int8_interleaved_4D_diff_seq(
    const uint32_t *packed_rows_interleaved,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_row_compact_int8_interleaved_2D_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_row_compact_int8_interleaved_4D_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb);

void sve_gemm_dense_int8_interleaved_4D(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved);

void sve_gemm_dense_int8_interleaved_2D(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved);

#ifdef __cplusplus
}
#endif

#endif
