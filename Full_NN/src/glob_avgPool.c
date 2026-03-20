
#include <stdio.h>
#include <stdlib.h>

#include <pooling.h>


float avg_v(float *in, uint32_t len);

float16_t avg_v_f16(float16_t *in, uint32_t len);


void glob_avg_pool(tensor3D_t *input, tensor3D_t *out){

    uint8_t out_h = 1;
    uint8_t out_w = 1;

    dim3D_t out_dim = {
        .depth = input->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float *out_tensor = (float*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float));

    for(int d=0; d<out_dim.depth; d++){
        out_tensor[d] = avg_v(&input->tensor[d * (input->dim.height * input->dim.width)], (input->dim.height * input->dim.width));
    }

    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}


void glob_avg_pool_f16(tensor3D_f16_t *input, tensor3D_f16_t *out){

    uint8_t out_h = 1;
    uint8_t out_w = 1;

    dim3D_t out_dim = {
        .depth = input->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float16_t *out_tensor = (float16_t*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float16_t));

    for(int d=0; d<out_dim.depth; d++){
        out_tensor[d] = avg_v_f16(&input->tensor[d * (input->dim.height * input->dim.width)], (input->dim.height * input->dim.width));
    }

    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}



float avg_v(float *in, uint32_t len){

    float sum = 0.0;

    for(uint32_t i=0; i<len; i++){
        sum += in[i];
    }

    return (sum / len);
}



float16_t avg_v_f16(float16_t *in, uint32_t len){

    float16_t sum = 0.0;

    for(uint32_t i=0; i<len; i++){
        sum += in[i];
    }

    return (sum / len);
}




void avg_v_interl_2D(float *in_inter, uint32_t len, float *avg_2_vals);

void avg_v_interl_4D(float *in_inter, uint32_t len, float *avg_4_vals);


void avg_v_interl_4D_f16(float16_t *in_inter, uint32_t len, float16_t *avg_4_vals);



void glob_avg_pool_interleavedND(tensor3D_t *input_interl, uint8_t interl_factor, tensor3D_t *out){

    // uint32_t nelems_channel = input_interl->dim.height * input_interl->dim.width;

    // Compute the output size
    uint32_t out_h = 1 ;
    uint32_t out_w = 1;         // The width has to be computed starting from the non-interleaved one
    out_w *= interl_factor;     // Return to interleaved dimentions

    dim3D_t out_dim = {
        .depth = input_interl->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float *out_tensor = (float*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float));
    
    // Allocate space to return the avg values of the interleaved patch (channel)
    float *avg_values = (float*)malloc(interl_factor * sizeof(float));

    for(int d=0; d<out_dim.depth; d++){

        if(interl_factor == 2){
            avg_v_interl_2D(&input_interl->tensor[d * (input_interl->dim.height * input_interl->dim.width)], (input_interl->dim.height * input_interl->dim.width), avg_values);

            out_tensor[(d*2) + 0] = avg_values[0];
            out_tensor[(d*2) + 1] = avg_values[1];
        }
        else if(interl_factor == 4){
            avg_v_interl_4D(&input_interl->tensor[d * (input_interl->dim.height * input_interl->dim.width)], (input_interl->dim.height * input_interl->dim.width), avg_values);

            out_tensor[(d*4) + 0] = avg_values[0];
            out_tensor[(d*4) + 1] = avg_values[1];
            out_tensor[(d*4) + 2] = avg_values[2];
            out_tensor[(d*4) + 3] = avg_values[3];
        }
    }

    free(avg_values);

    // Free the original output tensor so to use the newly computed one
    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}





void glob_avg_pool_interleavedND_f16(tensor3D_f16_t *input_interl, uint8_t interl_factor, tensor3D_f16_t *out){

    // uint32_t nelems_channel = input_interl->dim.height * input_interl->dim.width;

    // Compute the output size
    uint32_t out_h = 1 ;
    uint32_t out_w = 1;         // The width has to be computed starting from the non-interleaved one
    out_w *= interl_factor;     // Return to interleaved dimentions

    dim3D_t out_dim = {
        .depth = input_interl->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float16_t *out_tensor = (float16_t*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float16_t));
    
    // Allocate space to return the avg values of the interleaved patch (channel)
    float16_t *avg_values = (float16_t*)malloc(interl_factor * sizeof(float16_t));

    for(int d=0; d<out_dim.depth; d++){

        if(interl_factor == 2){
            printf("ERROR: global avg pooling not implemented for interl = 2!\n");
            exit(1);
        }

        if(interl_factor == 4){
            avg_v_interl_4D_f16(&input_interl->tensor[d * (input_interl->dim.height * input_interl->dim.width)], (input_interl->dim.height * input_interl->dim.width), avg_values);

            out_tensor[(d*4) + 0] = avg_values[0];
            out_tensor[(d*4) + 1] = avg_values[1];
            out_tensor[(d*4) + 2] = avg_values[2];
            out_tensor[(d*4) + 3] = avg_values[3];
        }
    }

    free(avg_values);

    // Free the original output tensor so to use the newly computed one
    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}




void avg_v_interl_2D(float *in_inter, uint32_t len, float *avg_2_vals){

    float sum0 = 0.0;
    float sum1 = 0.0;

    uint32_t single_len = len / 2;

    for(uint32_t i=0; i<single_len; i++){
        
        sum0 += in_inter[(i*2) + 0];
        sum1 += in_inter[(i*2) + 1];

    }

    avg_2_vals[0] = sum0 / single_len;
    avg_2_vals[1] = sum1 / single_len;
}






void avg_v_interl_4D(float *in_inter, uint32_t len, float *avg_4_vals){

    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;

    uint32_t single_len = len / 4;

    for(uint32_t i=0; i<single_len; i++){
        
        sum0 += in_inter[(i*4) + 0];
        sum1 += in_inter[(i*4) + 1];
        sum2 += in_inter[(i*4) + 2];
        sum3 += in_inter[(i*4) + 3];

    }

    avg_4_vals[0] = sum0 / single_len;
    avg_4_vals[1] = sum1 / single_len;
    avg_4_vals[2] = sum2 / single_len;
    avg_4_vals[3] = sum3 / single_len;
}





void avg_v_interl_4D_f16(float16_t *in_inter, uint32_t len, float16_t *avg_4_vals){

    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;

    uint32_t single_len = len / 4;

    for(uint32_t i=0; i<single_len; i++){
        
        float in0_f = (float)in_inter[(i*4) + 0];
        float in1_f = (float)in_inter[(i*4) + 1];
        float in2_f = (float)in_inter[(i*4) + 2];
        float in3_f = (float)in_inter[(i*4) + 3];

        sum0 += in0_f;
        sum1 += in1_f;
        sum2 += in2_f;
        sum3 += in3_f;

    }

    avg_4_vals[0] = (float16_t)(sum0 / single_len);
    avg_4_vals[1] = (float16_t)(sum1 / single_len);
    avg_4_vals[2] = (float16_t)(sum2 / single_len);
    avg_4_vals[3] = (float16_t)(sum3 / single_len);
}