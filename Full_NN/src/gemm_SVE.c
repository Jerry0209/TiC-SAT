#include <stdint.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include <gemm_SVE.h>

static uint32_t gemm_sve_idx_mask(uint8_t bits_per_cb) {
    return (bits_per_cb >= 32u) ? UINT32_MAX : ((1u << bits_per_cb) - 1u);
}

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
                               uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t acc = add_bias ? bias_val : 0;
            int32_t *out_slot = &out_mat[row * ld_out + out_col];
            if (accumulate) {
                *out_slot += acc;
            } else {
                *out_slot = acc;
            }
        }
        return;
    }

#if defined(__ARM_FEATURE_SVE)
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();

    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v = svdup_s32(0);
        uint32_t input_idx = 0;

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg =
                svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    svint32_t in_vals =
                        svld1sb_s32(bits_pg, &in_mat[row * ld_in + input_idx]);
                    svint32_t weights =
                        svld1_gather_u32index_s32(bits_pg, codebook_i32, cb_idxs);

                    acc_v = svmla_s32_m(bits_pg, acc_v, in_vals, weights);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc = svaddv_s32(svptrue_b32(), acc_v);
        if (add_bias) {
            acc += bias_val;
        }

        int32_t *out_slot = &out_mat[row * ld_out + out_col];
        if (accumulate) {
            *out_slot += acc;
        } else {
            *out_slot = acc;
        }
    }
#else
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);

    for (uint32_t row = 0; row < seq_tile; row++) {
        int32_t acc = add_bias ? bias_val : 0;

        for (uint32_t in_idx = 0; in_idx < k_elems; in_idx++) {
            uint32_t word_idx = in_idx / idxs_per_word;
            uint32_t offset = (in_idx % idxs_per_word) * bits_per_cb;
            uint32_t cb_idx = (packed_row[word_idx] >> offset) & idx_mask;
            acc += (int32_t)in_mat[row * ld_in + in_idx] *
                   codebook_i32[cb_idx];
        }

        int32_t *out_slot = &out_mat[row * ld_out + out_col];
        if (accumulate) {
            *out_slot += acc;
        } else {
            *out_slot = acc;
        }
    }
#endif
}
