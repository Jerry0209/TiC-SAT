
#include <stdio.h>

#include <relu.h>


float _relu(float x){
    return (x > 0) ? x : 0;
}


void relu(float *x, int len){

    for(int i=0; i<len; i++){
        x[i] = _relu(x[i]);
    }
}








float16_t _relu_f16(float16_t x){
    return (x > 0) ? x : 0;
}


void relu_f16(float16_t *x, int len){

    for(int i=0; i<len; i++){
        x[i] = _relu(x[i]);
    }
}