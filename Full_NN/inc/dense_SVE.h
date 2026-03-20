#ifndef _DENSE_SVE_H_
#define _DENSE_SVE_H_

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>


#include <arm_sve.h>

#include <codebooks_def.h>
#include <dense_exec.h>

void exec_sve_compact_interleaved_2D(dense_t dense_layer, const float *in_interl, const float *cb_interl, const uint32_t *weight_indexes, const float *bias, float *out_interl);
void exec_sve_compact_interleaved_4D_SVE(dense_t dense_layer, const float *in, const float *cb, const uint32_t *weight_indexes, const float *bias, float *out);
void exec_sve_compact_SVE_f16(dense_t dense_layer, const float16_t *in, const float16_t *cb, const uint16_t *weight_indexes, const float16_t *bias, float16_t *out);
void exec_sve_compact_interleaved_4D(dense_t dense_layer, const float *in_interl, const float *cb_interl, const uint32_t *weight_indexes, const float *bias_interl, float *out_interl);
void exec_sve_compact_interleaved_4D_no_lanes_loop(dense_t dense_layer, const float *in_interl, const float *cb_interl, const uint32_t *weight_indexes, const float *bias_interl, float *out_interl);
void exec_sve_compact_interleaved_4D_diff_seq(dense_t dense_layer, const float *in_interl, const float *cb_interl, const uint32_t *weight_indexes_interl, const float *bias, float *out_interl);
void exec_sve_compact_interleaved_4D_diff_seq_f16(dense_t dense_layer, const float16_t *in_interl, const float16_t *cb_interl, const uint16_t *weight_indexes_interl, const float16_t *bias, float16_t *out_interl);
void exec_sve_compact_interleaved_4D_f16(dense_t dense_layer, const float16_t *in_interl, const float16_t *cb_interl, const uint32_t *weight_indexes, const float16_t *bias_interl, float16_t *out_interl);
void exec_sve_compact_interleaved_4D_f16_optim(dense_t dense_layer, const float16_t *in_interl, const float16_t *cb_interl, const uint32_t *weight_indexes, const float16_t *bias_interl, float16_t *out_interl);


#endif