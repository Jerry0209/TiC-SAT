#ifndef _RESIDUAL_H_
#define _RESIDUAL_H_

#include <inttypes.h>


/**
 * Adds the residual coming from a skip connection.
 * Basically, it just expects the tensors flattened in a 1D shape and 
 * adds them element-wise.
 * The result is computed in-place in the x input tensor.
 * 
 * @param *x        : input tensor to which the residual will be added
 * @param *residual : residual tensor
 * @param len       : length of the tensors as in a 1D flattened form
 */
void add_residual(float *x, float *residual, uint32_t len);


#endif
