#ifndef _DENSE_EXEC_H_
#define _DENSE_EXEC_H_

#include <inttypes.h>

typedef struct dense_struct {
    uint16_t in_size;
    uint16_t out_size;

    uint16_t n_words_row;   // Number of words of packed indexes per row of weights

} dense_t;


void exec_compact_noCB(dense_t dense_layer, const float *in, const float *weights, float *bias, float *out);

void exec_compact(dense_t dense_layer, const float *in, const uint32_t *indexes, const float* codebook, float *bias, float *out);



#endif