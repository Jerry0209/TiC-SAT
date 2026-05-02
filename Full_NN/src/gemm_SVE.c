#include <stdint.h>
#include <stddef.h>

#include <arm_sve.h>

#include <codebooks_def.h>
#include <gemm_SVE.h>

static uint32_t gemm_sve_idx_mask(uint8_t bits_per_cb) {
    return (bits_per_cb >= 32u) ? UINT32_MAX : ((1u << bits_per_cb) - 1u);
}

#if defined(N_SVE_REG_CB_2)
static svint32_t gemm_extract_weightsx2_s32(svbool_t pg,
                                            svuint32_t idxs,
                                            svint32x2_t cb_regs_x2) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svint32_t weights_0 = svtbl_s32(svget2_s32(cb_regs_x2, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svint32_t weights_1 = svtbl_s32(svget2_s32(cb_regs_x2, 1), idxs_1);

    return svsel_s32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights_0);
}

static svfloat32_t gemm_extract_weightsx2_f32(svbool_t pg,
                                              svuint32_t idxs,
                                              svfloat32x2_t cb_regs_x2) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svfloat32_t weights_0 = svtbl_f32(svget2_f32(cb_regs_x2, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svfloat32_t weights_1 = svtbl_f32(svget2_f32(cb_regs_x2, 1), idxs_1);

    return svsel_f32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights_0);
}
#endif

#if defined(N_SVE_REG_CB_4)
static svint32_t gemm_extract_weightsx4_s32(svbool_t pg,
                                            svuint32_t idxs,
                                            svint32x4_t cb_regs_x4) {
    const uint32_t n_lanes = (uint32_t)svcntw(); // Each register stores one contiguous slice of the codebook.
    svint32_t weights_0 = svtbl_s32(svget4_s32(cb_regs_x4, 0), idxs); // Lookup values from codebook slice 0.
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes); // Rebase indexes for codebook slice 1.
    svuint32_t idxs_2 = svsub_n_u32_x(pg, idxs, 2u * n_lanes); // Rebase indexes for codebook slice 2.
    svuint32_t idxs_3 = svsub_n_u32_x(pg, idxs, 3u * n_lanes); // Rebase indexes for codebook slice 3.
    svint32_t weights_1 = svtbl_s32(svget4_s32(cb_regs_x4, 1), idxs_1); // Lookup values from codebook slice 1.
    svint32_t weights_2 = svtbl_s32(svget4_s32(cb_regs_x4, 2), idxs_2); // Lookup values from codebook slice 2.
    svint32_t weights_3 = svtbl_s32(svget4_s32(cb_regs_x4, 3), idxs_3); // Lookup values from codebook slice 3.
    svint32_t weights = weights_0; // Start with the first slice, then overwrite lanes that belong to later slices.

    weights = svsel_s32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights); // Keep slice 1 for indexes >= one register.
    weights = svsel_s32(svcmpge_n_u32(pg, idxs, 2u * n_lanes), weights_2, weights); // Keep slice 2 for indexes >= two registers.
    weights = svsel_s32(svcmpge_n_u32(pg, idxs, 3u * n_lanes), weights_3, weights); // Keep slice 3 for indexes >= three registers.

    return weights; // Return the same logical result as a wider table lookup.
}

static svfloat32_t gemm_extract_weightsx4_f32(svbool_t pg,
                                              svuint32_t idxs,
                                              svfloat32x4_t cb_regs_x4) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svfloat32_t weights_0 = svtbl_f32(svget4_f32(cb_regs_x4, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svuint32_t idxs_2 = svsub_n_u32_x(pg, idxs, 2u * n_lanes);
    svuint32_t idxs_3 = svsub_n_u32_x(pg, idxs, 3u * n_lanes);
    svfloat32_t weights_1 = svtbl_f32(svget4_f32(cb_regs_x4, 1), idxs_1);
    svfloat32_t weights_2 = svtbl_f32(svget4_f32(cb_regs_x4, 2), idxs_2);
    svfloat32_t weights_3 = svtbl_f32(svget4_f32(cb_regs_x4, 3), idxs_3);
    svfloat32_t weights = weights_0;

    weights = svsel_f32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights);
    weights = svsel_f32(svcmpge_n_u32(pg, idxs, 2u * n_lanes), weights_2, weights);
    weights = svsel_f32(svcmpge_n_u32(pg, idxs, 3u * n_lanes), weights_3, weights);

    return weights;
}
#endif

// bits_per_cb = 1 → mask = 0b1
// bits_per_cb = 2 → mask = 0b11
// bits_per_cb = 4 → mask = 0b1111
// bits_per_cb = 8 → mask = 0b11111111

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
    if ((bits_per_cb == 0u) || (k_elems == 0u)) { // No input elements, so output is just bias (if add_bias) or zero
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

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw(); // Auto-detect number of 32-bit lanes in SVE vector

    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v = svdup_s32(0);
        uint32_t input_idx = 0;

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg =
                svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row); // n_words_row is the number of uint32_t words in the row
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane); // duplicate the current lane's packed indices across the vector for processing

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
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb); // shifts for extracting each index from the packed word

                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v); // Get the actual codebook indices for this set of lanes

                    svint32_t in_vals =
                        svld1sb_s32(bits_pg, &in_mat[row * ld_in + input_idx]);
                    svint32_t weights =
                        svld1_gather_u32index_s32(bits_pg, codebook_i32, cb_idxs); // Get the corresponding weights from the codebook for these indices
                        // weights = [codebook_i32[2], codebook_i32[0], codebook_i32[3], codebook_i32[1]]

                    acc_v = svmla_s32_m(bits_pg, acc_v, in_vals, weights); // Add them together into the accumulator vector

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc = svaddv_s32(svptrue_b32(), acc_v); // Horizontally add the vector accumulator to get the final dot product for this output element
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
}

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
                               uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float acc = add_bias ? bias_val : 0.0f;
            float *out_slot = &out_mat[row * ld_out + out_col];
            if (accumulate) {
                *out_slot += acc;
            } else {
                *out_slot = acc;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
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

                    svbool_t bits_pg = svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts = svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);
                    svuint32_t cb_idxs = svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    svfloat32_t in_vals = svld1_f32(bits_pg, &in_mat[row * ld_in + input_idx]);
                    svfloat32_t weights =
                        svld1_gather_u32index_f32(bits_pg, codebook, cb_idxs);

                    acc_v = svmla_f32_m(bits_pg, acc_v, in_vals, weights);
                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc = svaddv_f32(svptrue_b32(), acc_v);
        if (add_bias) {
            acc += bias_val;
        }

        float *out_slot = &out_mat[row * ld_out + out_col];
        if (accumulate) {
            *out_slot += acc;
        } else {
            *out_slot = acc;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;
            const float bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0.0f;
            const float bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        svfloat32_t acc_v2 = svdup_f32(0.0f);
        svfloat32_t acc_v3 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32x4_t packed_idxs_4d =
                svld4_u32(load_pg, &packed_rows_interleaved[cw * 4u]);
            svuint32_t packed_idxs0 = svget4_u32(packed_idxs_4d, 0);
            svuint32_t packed_idxs1 = svget4_u32(packed_idxs_4d, 1);
            svuint32_t packed_idxs2 = svget4_u32(packed_idxs_4d, 2);
            svuint32_t packed_idxs3 = svget4_u32(packed_idxs_4d, 3);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed0 = svdup_lane_u32(packed_idxs0, lane);
                svuint32_t dup_idxs_packed1 = svdup_lane_u32(packed_idxs1, lane);
                svuint32_t dup_idxs_packed2 = svdup_lane_u32(packed_idxs2, lane);
                svuint32_t dup_idxs_packed3 = svdup_lane_u32(packed_idxs3, lane);

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

                    svuint32_t cb_idxs0 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed0, shifts);
                    svuint32_t cb_idxs1 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed1, shifts);
                    svuint32_t cb_idxs2 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed2, shifts);
                    svuint32_t cb_idxs3 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed3, shifts);

                    cb_idxs0 = svand_u32_z(bits_pg, cb_idxs0, idx_mask_v);
                    cb_idxs1 = svand_u32_z(bits_pg, cb_idxs1, idx_mask_v);
                    cb_idxs2 = svand_u32_z(bits_pg, cb_idxs2, idx_mask_v);
                    cb_idxs3 = svand_u32_z(bits_pg, cb_idxs3, idx_mask_v);

                    svfloat32x4_t in_vals =
                        svld4_f32(bits_pg, &row_in[input_idx * 4u]);
                    svfloat32_t in0 = svget4_f32(in_vals, 0);
                    svfloat32_t in1 = svget4_f32(in_vals, 1);
                    svfloat32_t in2 = svget4_f32(in_vals, 2);
                    svfloat32_t in3 = svget4_f32(in_vals, 3);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);
                    svfloat32_t weights2 = svdup_n_f32(0);
                    svfloat32_t weights3 = svdup_n_f32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs0);
                    weights1 = svtbl_f32(cb1, cb_idxs1);
                    weights2 = svtbl_f32(cb2, cb_idxs2);
                    weights3 = svtbl_f32(cb3, cb_idxs3);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs0, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs1, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs2, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs3, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs0, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs1, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs2, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs3, cb3_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_f32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_f32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);
        float acc2 = svaddv_f32(svptrue_b32(), acc_v2);
        float acc3 = svaddv_f32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget2_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget2_f32(codebooks_loaded, 1);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 2u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget2_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget2_f32(codebooks_loaded, 1));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 2u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget2_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget2_f32(codebooks_loaded, 1));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
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

                    svfloat32x2_t in_vals =
                        svld2_f32(bits_pg, &row_in[input_idx * 2u]);
                    svfloat32_t in0 = svget2_f32(in_vals, 0);
                    svfloat32_t in1 = svget2_f32(in_vals, 1);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs);
                    weights1 = svtbl_f32(cb1, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb1_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb1_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;
            const float bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0.0f;
            const float bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        svfloat32_t acc_v2 = svdup_f32(0.0f);
        svfloat32_t acc_v3 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
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

                    svfloat32x4_t in_vals =
                        svld4_f32(bits_pg, &row_in[input_idx * 4u]);
                    svfloat32_t in0 = svget4_f32(in_vals, 0);
                    svfloat32_t in1 = svget4_f32(in_vals, 1);
                    svfloat32_t in2 = svget4_f32(in_vals, 2);
                    svfloat32_t in3 = svget4_f32(in_vals, 3);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);
                    svfloat32_t weights2 = svdup_n_f32(0);
                    svfloat32_t weights3 = svdup_n_f32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs);
                    weights1 = svtbl_f32(cb1, cb_idxs);
                    weights2 = svtbl_f32(cb2, cb_idxs);
                    weights3 = svtbl_f32(cb3, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb3_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_f32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_f32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);
        float acc2 = svaddv_f32(svptrue_b32(), acc_v2);
        float acc3 = svaddv_f32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;
            const int32_t bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0;
            const int32_t bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget4_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget4_s32(codebooks_loaded, 1);
    svint32_t cb2 = svget4_s32(codebooks_loaded, 2);
    svint32_t cb3 = svget4_s32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    svint32x2_t cb2_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    svint32x2_t cb3_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget4_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget4_s32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_s32(svget2_s32(cb2_2regs, 0), svget4_s32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_s32(svget2_s32(cb3_2regs, 0), svget4_s32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb2_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb3_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget4_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget4_s32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(cb2_4regs, 2), svget4_s32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(cb3_4regs, 2), svget4_s32(codebooks_loaded, 3));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        svint32_t acc_v2 = svdup_s32(0);
        svint32_t acc_v3 = svdup_s32(0);
        uint32_t input_idx = 0;

        const int32_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32x4_t packed_idxs_4d =
                svld4_u32(load_pg, &packed_rows_interleaved[cw * 4u]);
            svuint32_t packed_idxs0 = svget4_u32(packed_idxs_4d, 0);
            svuint32_t packed_idxs1 = svget4_u32(packed_idxs_4d, 1);
            svuint32_t packed_idxs2 = svget4_u32(packed_idxs_4d, 2);
            svuint32_t packed_idxs3 = svget4_u32(packed_idxs_4d, 3);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed0 = svdup_lane_u32(packed_idxs0, lane);
                svuint32_t dup_idxs_packed1 = svdup_lane_u32(packed_idxs1, lane);
                svuint32_t dup_idxs_packed2 = svdup_lane_u32(packed_idxs2, lane);
                svuint32_t dup_idxs_packed3 = svdup_lane_u32(packed_idxs3, lane);

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

                    svuint32_t cb_idxs0 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed0, shifts);
                    svuint32_t cb_idxs1 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed1, shifts);
                    svuint32_t cb_idxs2 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed2, shifts);
                    svuint32_t cb_idxs3 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed3, shifts);

                    cb_idxs0 = svand_u32_z(bits_pg, cb_idxs0, idx_mask_v);
                    cb_idxs1 = svand_u32_z(bits_pg, cb_idxs1, idx_mask_v);
                    cb_idxs2 = svand_u32_z(bits_pg, cb_idxs2, idx_mask_v);
                    cb_idxs3 = svand_u32_z(bits_pg, cb_idxs3, idx_mask_v);

                    svint32x4_t in_vals =
                        svld4_s32(bits_pg, &row_in[input_idx * 4u]);
                    svint32_t in0 = svget4_s32(in_vals, 0);
                    svint32_t in1 = svget4_s32(in_vals, 1);
                    svint32_t in2 = svget4_s32(in_vals, 2);
                    svint32_t in3 = svget4_s32(in_vals, 3);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);
                    svint32_t weights2 = svdup_n_s32(0);
                    svint32_t weights3 = svdup_n_s32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs0);
                    weights1 = svtbl_s32(cb1, cb_idxs1);
                    weights2 = svtbl_s32(cb2, cb_idxs2);
                    weights3 = svtbl_s32(cb3, cb_idxs3);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs0, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs1, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs2, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs3, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs0, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs1, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs2, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs3, cb3_4regs);
#endif

                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_s32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_s32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        int32_t acc2 = svaddv_s32(svptrue_b32(), acc_v2);
        int32_t acc3 = svaddv_s32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget2_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget2_s32(codebooks_loaded, 1);

#elif defined(N_SVE_REG_CB_2)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 2u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget2_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget2_s32(codebooks_loaded, 1), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget2_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget2_s32(codebooks_loaded, 1));

#elif defined(N_SVE_REG_CB_4)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 2u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget2_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget2_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget2_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget2_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget2_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget2_s32(codebooks_loaded, 1), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget2_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget2_s32(codebooks_loaded, 1));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        uint32_t input_idx = 0;

        const int32_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
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

                    svint32x2_t in_vals =
                        svld2_s32(bits_pg, &row_in[input_idx * 2u]);
                    svint32_t in0 = svget2_s32(in_vals, 0);
                    svint32_t in1 = svget2_s32(in_vals, 1);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs);
                    weights1 = svtbl_s32(cb1, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb1_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb1_4regs);
#endif

                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
        }

        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

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
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;
            const int32_t bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0;
            const int32_t bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

#if defined(N_SVE_REG_CB_1)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget4_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget4_s32(codebooks_loaded, 1);
    svint32_t cb2 = svget4_s32(codebooks_loaded, 2);
    svint32_t cb3 = svget4_s32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    svint32x2_t cb2_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    svint32x2_t cb3_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget4_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget4_s32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_s32(svget2_s32(cb2_2regs, 0), svget4_s32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_s32(svget2_s32(cb3_2regs, 0), svget4_s32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb2_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb3_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget4_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget4_s32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(cb2_4regs, 2), svget4_s32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(cb3_4regs, 2), svget4_s32(codebooks_loaded, 3));
#endif

    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        svint32_t acc_v2 = svdup_s32(0);
        svint32_t acc_v3 = svdup_s32(0);
        uint32_t input_idx = 0;

        const int32_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
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

                    svint32x4_t in_vals =
                        svld4_s32(bits_pg, &row_in[input_idx * 4u]);
                    svint32_t in0 = svget4_s32(in_vals, 0);
                    svint32_t in1 = svget4_s32(in_vals, 1);
                    svint32_t in2 = svget4_s32(in_vals, 2);
                    svint32_t in3 = svget4_s32(in_vals, 3);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);
                    svint32_t weights2 = svdup_n_s32(0);
                    svint32_t weights3 = svdup_n_s32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs);
                    weights1 = svtbl_s32(cb1, cb_idxs);
                    weights2 = svtbl_s32(cb2, cb_idxs);
                    weights3 = svtbl_s32(cb3, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb3_4regs);
#endif

                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_s32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_s32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        int32_t acc2 = svaddv_s32(svptrue_b32(), acc_v2);
        int32_t acc3 = svaddv_s32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

void sve_gemm_dense_int8_interleaved_4D(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved) {
    const uint32_t n_lanes = (uint32_t)svcntw();

    for (uint32_t row = 0; row < lhs_rows; row++) {
        const int32_t *lhs_row = &lhs_interleaved[row * k_elems * 4u];

        for (uint32_t col = 0; col < rhs_cols; col++) {
            const int32_t *rhs_col = &rhs_by_col_interleaved[col * k_elems * 4u];
            svint32_t acc_v0 = svdup_s32(0);
            svint32_t acc_v1 = svdup_s32(0);
            svint32_t acc_v2 = svdup_s32(0);
            svint32_t acc_v3 = svdup_s32(0);

            for (uint32_t k = 0; k < k_elems; k += n_lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)k, (uint64_t)k_elems);
                svint32x4_t lhs_vals = svld4_s32(pg, &lhs_row[k * 4u]);
                svint32x4_t rhs_vals = svld4_s32(pg, &rhs_col[k * 4u]);

                acc_v0 = svmla_s32_m(pg, acc_v0, svget4_s32(lhs_vals, 0), svget4_s32(rhs_vals, 0));
                acc_v1 = svmla_s32_m(pg, acc_v1, svget4_s32(lhs_vals, 1), svget4_s32(rhs_vals, 1));
                acc_v2 = svmla_s32_m(pg, acc_v2, svget4_s32(lhs_vals, 2), svget4_s32(rhs_vals, 2));
                acc_v3 = svmla_s32_m(pg, acc_v3, svget4_s32(lhs_vals, 3), svget4_s32(rhs_vals, 3));
            }

            int32_t *out_slot = &out_interleaved[(row * rhs_cols + col) * 4u];
            out_slot[0] = svaddv_s32(svptrue_b32(), acc_v0);
            out_slot[1] = svaddv_s32(svptrue_b32(), acc_v1);
            out_slot[2] = svaddv_s32(svptrue_b32(), acc_v2);
            out_slot[3] = svaddv_s32(svptrue_b32(), acc_v3);
        }
    }
}

void sve_gemm_dense_int8_interleaved_2D(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved) {
    const uint32_t n_lanes = (uint32_t)svcntw();

    for (uint32_t row = 0; row < lhs_rows; row++) {
        const int32_t *lhs_row = &lhs_interleaved[row * k_elems * 2u];

        for (uint32_t col = 0; col < rhs_cols; col++) {
            const int32_t *rhs_col = &rhs_by_col_interleaved[col * k_elems * 2u];
            svint32_t acc_v0 = svdup_s32(0);
            svint32_t acc_v1 = svdup_s32(0);

            for (uint32_t k = 0; k < k_elems; k += n_lanes) {
                svbool_t pg = svwhilelt_b32((uint64_t)k, (uint64_t)k_elems);
                svint32x2_t lhs_vals = svld2_s32(pg, &lhs_row[k * 2u]);
                svint32x2_t rhs_vals = svld2_s32(pg, &rhs_col[k * 2u]);

                acc_v0 = svmla_s32_m(pg, acc_v0, svget2_s32(lhs_vals, 0), svget2_s32(rhs_vals, 0));
                acc_v1 = svmla_s32_m(pg, acc_v1, svget2_s32(lhs_vals, 1), svget2_s32(rhs_vals, 1));
            }

            int32_t *out_slot = &out_interleaved[(row * rhs_cols + col) * 2u];
            out_slot[0] = svaddv_s32(svptrue_b32(), acc_v0);
            out_slot[1] = svaddv_s32(svptrue_b32(), acc_v1);
        }
    }
}
