#ifndef _GEMM_SVE_H_
#define _GEMM_SVE_H_

#include <stdint.h>

#include <gemm_exec.h>

#ifdef __cplusplus
extern "C" {
#endif

void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb);

#ifdef __cplusplus
}
#endif

#endif
