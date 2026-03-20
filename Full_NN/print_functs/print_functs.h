#ifndef _PRINT_FUNCTS_H_
#define _PRINT_FUNCTS_H_

#include <inttypes.h>
#include <arm_sve.h>

/**
 * Prints n 1D-arrays stored in an interleaved way.
 * The length of each arrays is specified by `len`.
 * The number of interleaved array is specified by `n_interl`
 */
void print_flattened_interleaved(float *interl_vals, uint16_t len, uint8_t n_interl);

void print_flattened_interleaved_f16(float16_t *interl_vals, uint16_t len, uint8_t n_interl);

#endif