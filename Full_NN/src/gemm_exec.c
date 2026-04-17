#include <stdio.h>
#include <stdint.h>

#include <gemm_exec.h>
#ifdef SIMD
#include <gemm_SVE.h>
#endif

static uint32_t get_packed_index(const uint32_t *packed_row,
                                 uint32_t elem_idx,
                                 uint8_t bits_per_cb) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;

    return (packed_row[word_idx] >> offset) & idx_mask;
}


void gemm_exec_noCB(gemm_t gemm_layer,
                    const float *in,
                    const float *weights,
                    const float *bias,
                    float *out) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            float acc = bias == NULL ? 0.0f : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                acc += in[(seq * gemm_layer.input_size) + in_idx] *
                       weights[(out_idx * gemm_layer.input_size) + in_idx];
            }

            out[(seq * gemm_layer.output_size) + out_idx] = acc;
        }
    }
}

void gemm_exec_compact(gemm_t gemm_layer,
                       const float *in,
                       const uint32_t *weight_idx,
                       const float *codebook,
                       const float *bias,
                       float *out,
                       uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float acc = bias == NULL ? 0.0f : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                float weight = codebook[cb_idx];
                acc += in[(seq * gemm_layer.input_size) + in_idx] * weight;
            }

            out[(seq * gemm_layer.output_size) + out_idx] = acc;
        }
    }
}



void gemm_exec_noCB_int(gemm_t gemm_layer,
                        const int8_t *in,
                        const int8_t *weights,
                        const int32_t *bias,
                        int32_t *out) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            int32_t acc = (bias == NULL) ? 0 : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                int32_t in_val = (int32_t)in[seq * gemm_layer.input_size + in_idx];
                int32_t w_val  = (int32_t)weights[out_idx * gemm_layer.input_size + in_idx];
                acc += in_val * w_val;
            }

            out[seq * gemm_layer.output_size + out_idx] = acc;
        }
    }
}

void gemm_exec_compact_int(gemm_t gemm_layer,
                           const int8_t *in,
                           const uint32_t *weight_idx,
                           const int8_t *codebook,
                           const int32_t *bias,
                           int32_t *out,
                           uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t acc = (bias == NULL) ? 0 : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                int32_t in_val = (int32_t)in[seq * gemm_layer.input_size + in_idx];
                int32_t w_val  = (int32_t)codebook[cb_idx];
                acc += in_val * w_val;
            }

            out[seq * gemm_layer.output_size + out_idx] = acc;
        }
    }
}

#ifdef SIMD
void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0 : bias[out_idx];
            }
        }
        return;
    }

    if ((gemm_layer.input_size == 0u) || (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0 : bias[out_idx];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_int(gemm_layer,
                              in,
                              weight_idx,
                              codebook,
                              bias,
                              out,
                              bits_per_cb);
        return;
    }

    int32_t codebook_i32[256];
    const uint32_t codebook_size = 1u << bits_per_cb;
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        codebook_i32[cb_idx] = (int32_t)codebook[cb_idx];
    }

    /*
     * Non-tiled path: the tile sizes cover the full GEMM dimensions.
     * Reducing these two values later turns this into an L1/L2 tiled version
     * while preserving the same micro-kernel and partial-sum contract.
     */
    const uint32_t tile_seq = gemm_layer.seq_len;
    const uint32_t tile_k_words = gemm_layer.n_words_row;

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t bias_val = (bias == NULL) ? 0 : bias[out_idx];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                sve_gemm_row_compact_int8(&packed_row[w0],
                                          tile_words,
                                          k_tile,
                                          &in[seq0 * gemm_layer.input_size +
                                              processed_k],
                                          seq_tile,
                                          gemm_layer.input_size,
                                          codebook_i32,
                                          &out[seq0 * gemm_layer.output_size],
                                          out_idx,
                                          gemm_layer.output_size,
                                          bias_val,
                                          (w0 == 0u),
                                          (w0 != 0u),
                                          bits_per_cb);

                processed_k += k_tile;
            }
        }
    }
}
#endif
