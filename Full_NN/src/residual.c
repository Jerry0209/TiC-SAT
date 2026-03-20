#include <residual.h>


void add_residual(float *x, float *residual, uint32_t len){

    for(int i=0; i<len; i++){
        x[i] = x[i] + residual[i];
    }
}