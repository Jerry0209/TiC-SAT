
#ifndef _CONV_EXEC_H_
#define _CONV_EXEC_H_

#include <conv_def.h>

// void get_3Dpatch(float *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float *res);

// void get_3Dpatch_f16(float16_t *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float16_t *res);

tensor3D_t* check_padding(tensor3D_t *raw_input, uint32_t padding);
tensor3D_f16_t* check_padding_f16(tensor3D_f16_t *raw_input, uint32_t padding);

void check_padding_interleavedND(tensor3D_t *raw_in_interl, uint32_t padding, uint8_t interl_factor, tensor3D_t* res);

void check_padding_interleavedND_f16(tensor3D_f16_t *raw_in_interl, uint32_t padding, uint8_t interl_factor, tensor3D_f16_t* res);


void conv3D_staticPatch_compact(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, tensor3D_t *out);

void conv3D_staticPatch_compact_tiled(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, uint32_t tile_out, tensor3D_t *out);

void conv3D_staticPatch_compact_tiled_L1L2_noCB(conv_t conv_layer, tensor3D_t *input, const float *kernel, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out);

void conv3D_staticPatch_compact_tiled_L1L2(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out);

tensor3D_t conv_layer(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *codebook);

tensor3D_t conv_layer_tiled_l2(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, uint32_t tile_L2_size, const float *codebook);

tensor3D_t conv_layer_tiled_l2l1_SVE(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float *codebook);

tensor3D_f16_t conv_layer_tiled_l2l1_SVE_f16(conv_t conv, tensor3D_f16_t *in_tensor, const uint16_t *kernel_perCH, const float16_t *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float16_t *codebook);

tensor3D_t conv_layer_tiled_l2l1_noCB(conv_t conv, tensor3D_t *in_tensor, const float *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size);

tensor3D_t conv_layer_tiled_l2l1(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float *codebook);

tensor3D_t conv_layer_interl(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias,  uint8_t interl_factor);

tensor3D_t conv_layer_interl_tiled_l2l1(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);

tensor3D_t conv_layer_interl_tiled_l2l1_no_lanes_loop(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);

tensor3D_t conv_layer_interl_tiled_l2l1_diff_seq(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH_interl, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);

tensor3D_f16_t conv_layer_interl_tiled_l2l1_diff_seq_f16(conv_t conv, tensor3D_f16_t *in_interl, const uint16_t *kernel_perCH_interl, const float16_t *codebook_interl, const float16_t *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);

tensor3D_t conv_layer_interl_tiled_l2l1_mem(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);


/**
 * Performs a convolutional layer of interleaved ensemble of learners with tiled computation for L1D and L2 cache and codebooks quantized in fp16.
 */
tensor3D_f16_t conv_layer_interl_tiled_f16(conv_t conv, tensor3D_f16_t *in_interl, const uint32_t *kernel_perCH, const float16_t *codebook_interl, const float16_t *bias_interl, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor);

#endif