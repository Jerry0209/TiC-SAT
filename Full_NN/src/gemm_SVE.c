#include <stdint.h>

#include <arm_sve.h>

#include <gemm_SVE.h>

static uint32_t gemm_sve_idx_mask(uint8_t bits_per_cb) {
    return (bits_per_cb >= 32u) ? UINT32_MAX : ((1u << bits_per_cb) - 1u);
}

static uint32_t gemm_sve_get_packed_index(const uint32_t *packed_row,
                                          uint32_t elem_idx,
                                          uint8_t bits_per_cb) {
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t word_idx = elem_idx / idxs_per_word;
    const uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;

    return (packed_row[word_idx] >> offset) & gemm_sve_idx_mask(bits_per_cb);
}

static int32_t gemm_sve_dot_scalar_fallback(const int8_t *in_row,
                                            const uint32_t *packed_row,
                                            const int8_t *codebook,
                                            uint32_t input_size,
                                            uint8_t bits_per_cb) {
    int32_t acc = 0;

    for (uint32_t in_idx = 0; in_idx < input_size; in_idx++) {
        const uint32_t cb_idx =
            gemm_sve_get_packed_index(packed_row, in_idx, bits_per_cb);
        acc += (int32_t)in_row[in_idx] * (int32_t)codebook[cb_idx];
    }

    return acc;
}

static int32_t gemm_sve_dot_compact_int8(const int8_t *in_row,
                                         const uint32_t *packed_row,
                                         const int32_t *codebook_i32,
                                         uint32_t input_size,
                                         uint8_t bits_per_cb) {
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);
    svint32_t acc_v = svdup_s32(0);

    uint32_t input_idx = 0;
    for (uint32_t word_idx = 0; input_idx < input_size; word_idx++) {
        const uint32_t packed_word = packed_row[word_idx];
        uint32_t idx_in_word = 0;

        while ((idx_in_word < idxs_per_word) && (input_idx < input_size)) {
            const uint32_t remaining_word = idxs_per_word - idx_in_word;
            const uint32_t remaining_total = input_size - input_idx;
            const uint32_t active_lanes =
                (remaining_word < remaining_total) ? remaining_word : remaining_total;

            const svbool_t pg =
                svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
            const svuint32_t shifts =
                svindex_u32(idx_in_word * bits_per_cb, bits_per_cb);

            svuint32_t cb_idxs = svlsr_u32_z(pg, svdup_u32(packed_word), shifts);
            cb_idxs = svand_u32_z(pg, idx_mask_v, cb_idxs);

            const svint32_t in_vals = svld1sb_s32(pg, &in_row[input_idx]);
            const svint32_t weights =
                svld1_gather_u32index_s32(pg, codebook_i32, cb_idxs);

            acc_v = svmla_s32_m(pg, acc_v, in_vals, weights);

            const uint32_t processed = (uint32_t)svcntp_b32(pg, pg);
            input_idx += processed;
            idx_in_word += processed;
        }
    }

    return svaddv_s32(svptrue_b32(), acc_v);
}

void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb) {
    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == 0) ? 0 : bias[out_idx];
            }
        }
        return;
    }

    if (bits_per_cb > 8u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            const int8_t *in_row = &in[seq * gemm_layer.input_size];
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                const uint32_t *packed_row =
                    &weight_idx[out_idx * gemm_layer.n_words_row];
                const int32_t dot =
                    gemm_sve_dot_scalar_fallback(in_row,
                                                 packed_row,
                                                 codebook,
                                                 gemm_layer.input_size,
                                                 bits_per_cb);
                out[seq * gemm_layer.output_size + out_idx] =
                    ((bias == 0) ? 0 : bias[out_idx]) + dot;
            }
        }
        return;
    }

    int32_t codebook_i32[256];
    const uint32_t codebook_size = 1u << bits_per_cb;
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        codebook_i32[cb_idx] = (int32_t)codebook[cb_idx];
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        const int8_t *in_row = &in[seq * gemm_layer.input_size];

        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            const int32_t dot =
                gemm_sve_dot_compact_int8(in_row,
                                          packed_row,
                                          codebook_i32,
                                          gemm_layer.input_size,
                                          bits_per_cb);

            out[seq * gemm_layer.output_size + out_idx] =
                ((bias == 0) ? 0 : bias[out_idx]) + dot;
        }
    }
}
