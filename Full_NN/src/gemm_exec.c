#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gemm_exec.h>
#ifdef SIMD
#include <codebooks_def.h>
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

static uint32_t get_packed_index_interleaved_4d(
    const uint32_t *packed_rows_interleaved,
    uint32_t elem_idx,
    uint8_t bits_per_cb,
    uint32_t learner) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    uint32_t packed_word = packed_rows_interleaved[word_idx * 4u + learner];

    return (packed_word >> offset) & idx_mask;
}

static uint32_t get_packed_index_interleaved_nd(
    const uint32_t *packed_rows_interleaved,
    uint32_t elem_idx,
    uint8_t bits_per_cb,
    uint32_t learner,
    uint32_t learner_count) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    uint32_t packed_word = packed_rows_interleaved[word_idx * learner_count + learner];

    return (packed_word >> offset) & idx_mask;
}

#ifdef SIMD
#ifndef TILE_L1_SIZE
#define TILE_L1_SIZE 0
#endif

static uint32_t gemm_sve_l1_tile_or_full(uint32_t full_size) {
    const uint32_t tile_size = (uint32_t)TILE_L1_SIZE;
    if ((tile_size <= 1u) || (tile_size >= full_size)) {
        return full_size;
    }
    return tile_size;
}

static uint32_t gemm_sve_codebook_capacity(void) {
#if defined(N_SVE_REG_CB_4)
    return N_SVE_LANES * 4u; // Four SVE registers are available for each learner codebook.
#elif defined(N_SVE_REG_CB_2)
    return N_SVE_LANES * 2u; // Two SVE registers are available for each learner codebook.
#elif defined(N_SVE_REG_CB_1)
    return N_SVE_LANES; // One SVE register is available for each learner codebook.
#else
    return 0u; // No Mentor-style codebook register mode was selected.
#endif
}

static int gemm_sve_codebook_fits_registers(uint32_t codebook_size) {
    const uint32_t capacity = gemm_sve_codebook_capacity(); // Match the same N_SVE_REG_CB_* policy used by the SVE row kernels.
    return (capacity != 0u) && (codebook_size <= capacity); // Fall back when the codebook cannot be fully cached in SVE registers.
}

static void gemm_sve_unpack_int8_interleaved_tile(const int8_t *src_interleaved,
                                                  int32_t *dst_interleaved,
                                                  uint32_t seq_tile,
                                                  uint32_t k_tile,
                                                  uint32_t ld_src_interleaved,
                                                  uint32_t learner_count) {
    const uint32_t tile_row_elems = k_tile * learner_count;

    for (uint32_t row = 0; row < seq_tile; row++) {
        const int8_t *src_row = &src_interleaved[row * ld_src_interleaved];
        int32_t *dst_row = &dst_interleaved[row * tile_row_elems];

        for (uint32_t idx = 0; idx < tile_row_elems; idx++) {
            dst_row[idx] = (int32_t)src_row[idx];
        }
    }
}

static void gemm_exec_compact_int_interleaved_2D_same_seq_to_int8_scalar(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    const int do_gemm =
        (bits_per_cb != 0u) && (gemm_layer.input_size != 0u) &&
        (gemm_layer.n_words_row != 0u);

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int8_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
            int32_t acc0 =
                (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 0u];
            int32_t acc1 =
                (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 1u];

            if (do_gemm) {
                for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                    const int8_t *in_vals =
                        &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 2u];
                    const uint32_t cb_idx =
                        get_packed_index(packed_row, in_idx, bits_per_cb);

                    acc0 += (int32_t)in_vals[0] *
                            (int32_t)codebook_interleaved[cb_idx * 2u + 0u];
                    acc1 += (int32_t)in_vals[1] *
                            (int32_t)codebook_interleaved[cb_idx * 2u + 1u];
                }
            }

            out_slot[0] = (int8_t)acc0;
            out_slot[1] = (int8_t)acc1;
        }
    }
}

static void gemm_exec_compact_int_interleaved_4D_same_seq_to_int8_scalar(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    const int do_gemm =
        (bits_per_cb != 0u) && (gemm_layer.input_size != 0u) &&
        (gemm_layer.n_words_row != 0u);

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int8_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            int32_t acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + learner];
            }

            if (do_gemm) {
                for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                    const int8_t *in_vals =
                        &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                    const uint32_t cb_idx =
                        get_packed_index(packed_row, in_idx, bits_per_cb);

                    for (uint32_t learner = 0; learner < 4u; learner++) {
                        acc[learner] +=
                            (int32_t)in_vals[learner] *
                            (int32_t)codebook_interleaved[cb_idx * 4u + learner];
                    }
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = (int8_t)acc[learner];
            }
        }
    }
}

static void gemm_exec_compact_int_interleaved_4D_diff_seq_to_int8_scalar(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    const int do_gemm =
        (bits_per_cb != 0u) && (gemm_layer.input_size != 0u) &&
        (gemm_layer.n_words_row != 0u);

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_rows =
                &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
            int8_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            int32_t acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + learner];
            }

            if (do_gemm) {
                for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                    const int8_t *in_vals =
                        &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];

                    for (uint32_t learner = 0; learner < 4u; learner++) {
                        const uint32_t cb_idx =
                            get_packed_index_interleaved_4d(
                                packed_rows, in_idx, bits_per_cb, learner);
                        acc[learner] +=
                            (int32_t)in_vals[learner] *
                            (int32_t)codebook_interleaved[cb_idx * 4u + learner];
                    }
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = (int8_t)acc[learner];
            }
        }
    }
}
#endif

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


// Scalar version of the compact gemm exec, used as a fallback when the codebook cannot be fully cached in SVE registers.
void gemm_exec_compact_fp32_interleaved_2D_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
            float acc0 = (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 2u + 0u];
            float acc1 = (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 2u + 1u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 2u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                acc0 += in_vals[0] * codebook_interleaved[cb_idx * 2u + 0u];
                acc1 += in_vals[1] * codebook_interleaved[cb_idx * 2u + 1u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

void gemm_exec_compact_fp32_interleaved_4D_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            float acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 4u + learner];
            }

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                for (uint32_t learner = 0; learner < 4u; learner++) {
                    acc[learner] += in_vals[learner] *
                                    codebook_interleaved[cb_idx * 4u + learner];
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = acc[learner];
            }
        }
    }
}

void gemm_exec_compact_fp32_interleaved_4D_diff_seq(gemm_t gemm_layer,
                                                   const float *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const float *codebook_interleaved,
                                                   const float *bias_interleaved,
                                                   float *out_interleaved,
                                                   uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_rows =
                &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            float acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 4u + learner];
            }

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                for (uint32_t learner = 0; learner < 4u; learner++) {
                    uint32_t cb_idx =
                        get_packed_index_interleaved_nd(packed_rows, in_idx, bits_per_cb, learner, 4u);
                    acc[learner] += in_vals[learner] *
                                    codebook_interleaved[cb_idx * 4u + learner];
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = acc[learner];
            }
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

// Scalar reference/fallback kernel for the grouped 4-learner path.
// Each output element stores 4 parallel accumulators, one per learner, so the
// caller can process 4 different sequences/weight sets in a single traversal.
void gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_rows =
                &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
            int32_t acc2 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
            int32_t acc3 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx0 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 0u);
                uint32_t cb_idx1 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 1u);
                uint32_t cb_idx2 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 2u);
                uint32_t cb_idx3 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 3u);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx0 * 4u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx1 * 4u + 1u];
                acc2 += (int32_t)in_vals[2] * (int32_t)codebook_interleaved[cb_idx2 * 4u + 2u];
                acc3 += (int32_t)in_vals[3] * (int32_t)codebook_interleaved[cb_idx3 * 4u + 3u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

void gemm_exec_compact_int_interleaved_2D_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 1u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 2u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx * 2u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx * 2u + 1u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

void gemm_exec_compact_int_interleaved_4D_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
            int32_t acc2 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
            int32_t acc3 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx * 4u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx * 4u + 1u];
                acc2 += (int32_t)in_vals[2] * (int32_t)codebook_interleaved[cb_idx * 4u + 2u];
                acc3 += (int32_t)in_vals[3] * (int32_t)codebook_interleaved[cb_idx * 4u + 3u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

#ifdef SIMD
void gemm_exec_compact_sve(gemm_t gemm_layer,
                           const float *in,
                           const uint32_t *weight_idx,
                           const float *codebook,
                           const float *bias,
                           float *out,
                           uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0.0f : bias[out_idx];
            }
        }
        return;
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float bias_val = (bias == NULL) ? 0.0f : bias[out_idx];

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

                sve_gemm_row_compact_fp32(&packed_row[w0],
                                          tile_words,
                                          k_tile,
                                          &in[seq0 * gemm_layer.input_size +
                                              processed_k],
                                          seq_tile,
                                          gemm_layer.input_size,
                                          codebook,
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

void gemm_exec_compact_sve_fp32_interleaved_2D_same_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                float *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
                out_slot[0] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 2u + 0u];
                out_slot[1] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 2u + 1u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_fp32_interleaved_2D_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_fp32_interleaved_2D_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 2u];

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

                sve_gemm_row_compact_fp32_interleaved_2D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 2u],
                    seq_tile,
                    gemm_layer.input_size * 2u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 2u],
                    out_idx,
                    gemm_layer.output_size * 2u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }
}

void gemm_exec_compact_sve_fp32_interleaved_4D_same_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                float *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_fp32_interleaved_4D_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_fp32_interleaved_4D_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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

                sve_gemm_row_compact_fp32_interleaved_4D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }
}

void gemm_exec_compact_sve_fp32_interleaved_4D_diff_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                float *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL)
                                  ? 0.0f
                                  : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_fp32_interleaved_4D_diff_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx_interleaved,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_fp32_interleaved_4D_diff_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx_interleaved,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_rows =
            &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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

                sve_gemm_row_compact_fp32_interleaved_4D_diff_seq(
                    &packed_rows[w0 * 4u],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }
}

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

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

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

// SVE version of the same grouped 4-learner kernel. The input/output layout
// matches gemm_exec_compact_int_interleaved_4D_diff_seq(); only the inner math
// changes, so higher layers select between them purely with #ifdef SIMD.
void gemm_exec_compact_int_sve_interleaved_4D_diff_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if ((gemm_layer.input_size == 0u) || (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx_interleaved,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx_interleaved,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner]; // Preserve [codebook index][learner] layout for svld4_s32.
        }
    }

    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 4u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx_interleaved,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_rows =
            &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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

                sve_gemm_row_compact_int8_interleaved_4D_diff_seq(
                    &packed_rows[w0 * 4u],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }

    free(input_i32_interleaved);
}

void gemm_exec_compact_int_sve_interleaved_4D_diff_seq_to_int8(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u) || (bits_per_cb > 8u)) {
        gemm_exec_compact_int_interleaved_4D_diff_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx_interleaved,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_4D_diff_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx_interleaved,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner];
        }
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t max_k_in_tile = tile_k_words * idxs_per_word;
    const uint32_t max_k_tile =
        (max_k_in_tile < gemm_layer.input_size) ? max_k_in_tile : gemm_layer.input_size;
    const size_t input_tile_count =
        (size_t)tile_seq * (size_t)max_k_tile * 4u;
    const size_t out_tile_count = (size_t)tile_seq * 4u;
    int32_t *input_i32_tile =
        (int32_t *)malloc(input_tile_count * sizeof(int32_t));
    int32_t *out_col_tile =
        (int32_t *)malloc(out_tile_count * sizeof(int32_t));
    if ((input_i32_tile == NULL) || (out_col_tile == NULL)) {
        free(input_i32_tile);
        free(out_col_tile);
        gemm_exec_compact_int_interleaved_4D_diff_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx_interleaved,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_rows =
            &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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
                const uint32_t max_k_in_current_tile = tile_words * idxs_per_word;
                const uint32_t k_tile =
                    ((processed_k + max_k_in_current_tile) <= gemm_layer.input_size)
                        ? max_k_in_current_tile
                        : (gemm_layer.input_size - processed_k);

                // Keep the SVE kernel's [k][learner] layout, but only widen the
                // active tile so the full activation matrix is not staged as int32.
                gemm_sve_unpack_int8_interleaved_tile(
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    input_i32_tile,
                    seq_tile,
                    k_tile,
                    gemm_layer.input_size * 4u,
                    4u);

                sve_gemm_row_compact_int8_interleaved_4D_diff_seq(
                    &packed_rows[w0 * 4u],
                    tile_words,
                    k_tile,
                    input_i32_tile,
                    seq_tile,
                    k_tile * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    out_col_tile,
                    0u,
                    4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }

            for (uint32_t row = 0; row < seq_tile; row++) {
                int8_t *out_slot =
                    &out_interleaved[(((seq0 + row) * gemm_layer.output_size) + out_idx) * 4u];
                const int32_t *acc_slot = &out_col_tile[row * 4u];
                out_slot[0] = (int8_t)acc_slot[0];
                out_slot[1] = (int8_t)acc_slot[1];
                out_slot[2] = (int8_t)acc_slot[2];
                out_slot[3] = (int8_t)acc_slot[3];
            }
        }
    }

    free(input_i32_tile);
    free(out_col_tile);
}

void gemm_exec_compact_int_sve_interleaved_2D_same_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 1u];
            }
        }
        return;
    }

    if ((gemm_layer.input_size == 0u) || (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 1u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_int_interleaved_2D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_2D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[2u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 2u; learner++) {
            codebook_i32_interleaved[cb_idx * 2u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 2u + learner]; // Preserve [codebook index][learner] layout for svld2_s32.
        }
    }

    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 2u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        gemm_exec_compact_int_interleaved_2D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 2u];

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

                sve_gemm_row_compact_int8_interleaved_2D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 2u],
                    seq_tile,
                    gemm_layer.input_size * 2u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 2u],
                    out_idx,
                    gemm_layer.output_size * 2u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }

    free(input_i32_interleaved);
}

void gemm_exec_compact_int_sve_interleaved_2D_same_seq_to_int8(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u) || (bits_per_cb > 8u)) {
        gemm_exec_compact_int_interleaved_2D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_2D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[2u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 2u; learner++) {
            codebook_i32_interleaved[cb_idx * 2u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 2u + learner];
        }
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t max_k_in_tile = tile_k_words * idxs_per_word;
    const uint32_t max_k_tile =
        (max_k_in_tile < gemm_layer.input_size) ? max_k_in_tile : gemm_layer.input_size;
    const size_t input_tile_count =
        (size_t)tile_seq * (size_t)max_k_tile * 2u;
    const size_t out_tile_count = (size_t)tile_seq * 2u;
    int32_t *input_i32_tile =
        (int32_t *)malloc(input_tile_count * sizeof(int32_t));
    int32_t *out_col_tile =
        (int32_t *)malloc(out_tile_count * sizeof(int32_t));
    if ((input_i32_tile == NULL) || (out_col_tile == NULL)) {
        free(input_i32_tile);
        free(out_col_tile);
        gemm_exec_compact_int_interleaved_2D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 2u];

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
                const uint32_t max_k_in_current_tile = tile_words * idxs_per_word;
                const uint32_t k_tile =
                    ((processed_k + max_k_in_current_tile) <= gemm_layer.input_size)
                        ? max_k_in_current_tile
                        : (gemm_layer.input_size - processed_k);

                // The compact scratch mirrors Mentor's tiled kernels: tile-local
                // storage is compact, while the final store still uses full stride.
                gemm_sve_unpack_int8_interleaved_tile(
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 2u],
                    input_i32_tile,
                    seq_tile,
                    k_tile,
                    gemm_layer.input_size * 2u,
                    2u);

                sve_gemm_row_compact_int8_interleaved_2D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    input_i32_tile,
                    seq_tile,
                    k_tile * 2u,
                    codebook_i32_interleaved,
                    codebook_size,
                    out_col_tile,
                    0u,
                    2u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }

            for (uint32_t row = 0; row < seq_tile; row++) {
                int8_t *out_slot =
                    &out_interleaved[(((seq0 + row) * gemm_layer.output_size) + out_idx) * 2u];
                const int32_t *acc_slot = &out_col_tile[row * 2u];
                out_slot[0] = (int8_t)acc_slot[0];
                out_slot[1] = (int8_t)acc_slot[1];
            }
        }
    }

    free(input_i32_tile);
    free(out_col_tile);
}

void gemm_exec_compact_int_sve_interleaved_4D_same_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if ((gemm_layer.input_size == 0u) || (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                int32_t *out_slot =
                    &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
                out_slot[0] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
                out_slot[1] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
                out_slot[2] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
                out_slot[3] = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        gemm_exec_compact_int_interleaved_4D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_4D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner]; // Preserve [codebook index][learner] layout for svld4_s32.
        }
    }

    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 4u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        gemm_exec_compact_int_interleaved_4D_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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

                sve_gemm_row_compact_int8_interleaved_4D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }
        }
    }

    free(input_i32_interleaved);
}

void gemm_exec_compact_int_sve_interleaved_4D_same_seq_to_int8(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int8_t *out_interleaved,
    uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u) || (bits_per_cb > 8u)) {
        gemm_exec_compact_int_interleaved_4D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        gemm_exec_compact_int_interleaved_4D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner];
        }
    }

    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t max_k_in_tile = tile_k_words * idxs_per_word;
    const uint32_t max_k_tile =
        (max_k_in_tile < gemm_layer.input_size) ? max_k_in_tile : gemm_layer.input_size;
    const size_t input_tile_count =
        (size_t)tile_seq * (size_t)max_k_tile * 4u;
    const size_t out_tile_count = (size_t)tile_seq * 4u;
    int32_t *input_i32_tile =
        (int32_t *)malloc(input_tile_count * sizeof(int32_t));
    int32_t *out_col_tile =
        (int32_t *)malloc(out_tile_count * sizeof(int32_t));
    if ((input_i32_tile == NULL) || (out_col_tile == NULL)) {
        free(input_i32_tile);
        free(out_col_tile);
        gemm_exec_compact_int_interleaved_4D_same_seq_to_int8_scalar(
            gemm_layer,
            in_interleaved,
            weight_idx,
            codebook_interleaved,
            bias_interleaved,
            out_interleaved,
            bits_per_cb);
        return;
    }

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

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
                const uint32_t max_k_in_current_tile = tile_words * idxs_per_word;
                const uint32_t k_tile =
                    ((processed_k + max_k_in_current_tile) <= gemm_layer.input_size)
                        ? max_k_in_current_tile
                        : (gemm_layer.input_size - processed_k);

                // Keep widening local to the current K/sequence tile; the full
                // int32 activation matrix was the main source of cache pressure.
                gemm_sve_unpack_int8_interleaved_tile(
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    input_i32_tile,
                    seq_tile,
                    k_tile,
                    gemm_layer.input_size * 4u,
                    4u);

                sve_gemm_row_compact_int8_interleaved_4D_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    input_i32_tile,
                    seq_tile,
                    k_tile * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    out_col_tile,
                    0u,
                    4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                processed_k += k_tile;
            }

            for (uint32_t row = 0; row < seq_tile; row++) {
                int8_t *out_slot =
                    &out_interleaved[(((seq0 + row) * gemm_layer.output_size) + out_idx) * 4u];
                const int32_t *acc_slot = &out_col_tile[row * 4u];
                out_slot[0] = (int8_t)acc_slot[0];
                out_slot[1] = (int8_t)acc_slot[1];
                out_slot[2] = (int8_t)acc_slot[2];
                out_slot[3] = (int8_t)acc_slot[3];
            }
        }
    }

    free(input_i32_tile);
    free(out_col_tile);
}
#endif
