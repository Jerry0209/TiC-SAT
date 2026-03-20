#ifndef _MAX_POOL_H_
#define _MAX_POOL_H_

#include <conv_def.h>

void maxPool(tensor3D_t *input, uint32_t nelems_channel, uint8_t pool_size, uint8_t stride, tensor3D_t *out);

void maxPool_f16(tensor3D_f16_t *input, uint32_t nelems_channel, uint8_t pool_size, uint8_t stride, tensor3D_f16_t *out);

void maxPool_interleavedND(tensor3D_t *input_interl, uint8_t pool_size, uint8_t stride, uint8_t interl_factor, tensor3D_t *out);

void maxPool_interleavedND_f16(tensor3D_f16_t *input_interl, uint8_t pool_size, uint8_t stride, uint8_t interl_factor, tensor3D_f16_t *out);



void glob_avg_pool(tensor3D_t *input, tensor3D_t *out);

void glob_avg_pool_f16(tensor3D_f16_t *input, tensor3D_f16_t *out);

void glob_avg_pool_interleavedND(tensor3D_t *input_interl, uint8_t interl_factor, tensor3D_t *out);

void glob_avg_pool_interleavedND_f16(tensor3D_f16_t *input_interl, uint8_t interl_factor, tensor3D_f16_t *out);

#endif