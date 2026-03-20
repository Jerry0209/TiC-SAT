#ifndef _RELU_H_
#define _RELU_H_

#include <arm_sve.h>

void relu(float *x, int len);

void relu_f16(float16_t *x, int len);

#endif