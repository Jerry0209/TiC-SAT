#ifndef _SVE_IMPLEMENTATIONS_H_
#define _SVE_IMPLEMENTATIONS_H_

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <arm_sve.h>

#include <sve_prints.h>

#include <codebooks_def.h>
#include <conv_def.h>


void duplicate_interleave(float *in, uint32_t len, uint8_t n_reps, float *out);


void conv3D_staticPatch_compact_tiled_L1L2_SVE(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out);
void conv3D_staticPatch_compact_tiled_L1L2_SVE_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint16_t *kernel, const float16_t *codebook, const float16_t *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_f16_t *out);

void conv3D_staticPatch_compactSVE_interleavedND(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, uint8_t interl_factor, uint8_t tile_out, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_no_lanes_loop(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel_interl, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint16_t *kernel_interl, const float16_t *codebook_interl, const float16_t *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_f16_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_mem(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out);
void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint32_t *kernel, const float16_t *codebook_interl, const float16_t *bias_interl, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_f16_t *out);
void sve_vect_mul_compact_interleaved4D(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t mat_cols, const float *codebook_interl, tensor3D_t *res, uint32_t out_index);

svfloat32_t extract_weightsx2(svbool_t pg, svuint32_t idxs, svfloat32x2_t cb_resg_x2);
svfloat32_t extract_weightsx4(svbool_t pg, svuint32_t idxs, svfloat32x4_t cb_regs_x4);

svfloat32_t get_weights_f32(svfloat16_t codebook_f16, svuint32_t idxs);
svfloat16_t extract_weightsx2_f16(svbool_t pg, svuint16_t idxs, svfloat16x2_t cb_resg_x2);


#define CEIL_DIV(x, y)  ((x + y - 1) / (y))

#endif