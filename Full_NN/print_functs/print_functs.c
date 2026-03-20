#include <stdio.h>

#include <print_functs.h>
#include <arm_sve.h>


void print_flattened_interleaved(float *interl_vals, uint16_t len, uint8_t n_interl){

    for(uint8_t i=0; i<n_interl; i++){
        printf("\n---- Learner %d ----\n", i);
        for(uint16_t j=0; j<len; j++){
            printf("%f\n", interl_vals[(j*n_interl)+i]);
        }
    }

}

void print_flattened_interleaved_f16(float16_t *interl_vals, uint16_t len, uint8_t n_interl){

    for(uint8_t i=0; i<n_interl; i++){
        printf("\n---- Learner %d ----\n", i);
        for(uint16_t j=0; j<len; j++){
            printf("%f\n", interl_vals[(j*n_interl)+i]);
        }
    }

}