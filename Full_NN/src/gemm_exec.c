#include <stdio.h>
#include <stdint.h>

#include <gemm_exec.h>

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