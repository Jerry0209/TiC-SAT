
#include <stdio.h>
#include <stdlib.h>

#include <pooling.h>

#include <conv_def.h>


float max_v(float *in, uint32_t len);
float16_t max_v_f16(float16_t *in, uint32_t len);


void maxPool(tensor3D_t *input, uint32_t nelems_channel, uint8_t pool_size, uint8_t stride, tensor3D_t *out){

    uint32_t out_h = ((input->dim.height - pool_size) / stride) + 1;
    uint32_t out_w = ((input->dim.width - pool_size) / stride) + 1;

    dim3D_t out_dim = {
        .depth = input->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float *out_tensor = (float*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float));


    float *flat_patches = (float*)malloc(input->dim.depth * pool_size * pool_size * sizeof(float));

    for(int r=0; r<out_h; r++){
        for(int c=0; c<out_w; c++){

            // dim3D_t patch_idx = {
            //     .depth=0,
            //     .height=(r * pool_size),    // Here I assume that the stride is always equal to the kernel for MaxPooling
            //     .width=(c * pool_size)
            // };
            dim3D_t patch_idx = {
                .depth=0,
                .height=(r * stride),    // Here I assume that the stride is always equal to the kernel for MaxPooling
                .width=(c * stride)
            };

            // printf("Patch idx: %d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);

            dim3D_t patch_size = {
                .depth = input->dim.depth,
                .height = pool_size,
                .width = pool_size
            };

            get_3Dpatch(input->tensor, &input->dim, &patch_idx, &patch_size, nelems_channel, flat_patches);
            
            // print_volume(flat_patches, &patch_size);
            
            for(int ch=0; ch<input->dim.depth; ch++){
                float max = max_v(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width);
                uint32_t index = (ch * out_h * out_w) + (r * out_w) + c;
                out_tensor[index] = max;
            }
        }
    }

    free(flat_patches);

    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}


float max_v(float *in, uint32_t len){
    float max = in[0];

    for(uint32_t i=0; i<len; i++){
        if(in[i] >= max){
            max = in[i];
        }
    }
    return max;
}


float16_t max_v_f16(float16_t *in, uint32_t len){
    float16_t max = in[0];

    for(uint32_t i=0; i<len; i++){
        if(in[i] >= max){
            max = in[i];
        }
    }
    return max;
}




void maxPool_f16(tensor3D_f16_t *input, uint32_t nelems_channel, uint8_t pool_size, uint8_t stride, tensor3D_f16_t *out){

    uint32_t out_h = ((input->dim.height - pool_size) / stride) + 1;
    uint32_t out_w = ((input->dim.width - pool_size) / stride) + 1;

    dim3D_t out_dim = {
        .depth = input->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // Allocate a new output tensor
    float16_t *out_tensor = (float16_t*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float16_t));


    float16_t *flat_patches = (float16_t*)malloc(input->dim.depth * pool_size * pool_size * sizeof(float16_t));

    for(int r=0; r<out_h; r++){
        for(int c=0; c<out_w; c++){

            // dim3D_t patch_idx = {
            //     .depth=0,
            //     .height=(r * pool_size),    // Here I assume that the stride is always equal to the kernel for MaxPooling
            //     .width=(c * pool_size)
            // };
            dim3D_t patch_idx = {
                .depth=0,
                .height=(r * stride),    // Here I assume that the stride is always equal to the kernel for MaxPooling
                .width=(c * stride)
            };

            // printf("Patch idx: %d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);

            dim3D_t patch_size = {
                .depth = input->dim.depth,
                .height = pool_size,
                .width = pool_size
            };

            get_3Dpatch_f16(input->tensor, &input->dim, &patch_idx, &patch_size, nelems_channel, flat_patches);
            
            // print_volume(flat_patches, &patch_size);
            
            for(int ch=0; ch<input->dim.depth; ch++){
                float16_t max = max_v_f16(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width);
                uint32_t index = (ch * out_h * out_w) + (r * out_w) + c;
                out_tensor[index] = max;
            }
        }
    }

    free(flat_patches);

    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}












/**
 * Computes the maximum values over an array of 2 interleaved values.
 * 
 * Considering the following input:
 * [A0, A1, A2, A3, B0, B1, B2, B3]
 * 
 * The input length is considered to be 2 * 2 = 4.
 * That is 2 values (A and B) for 2 inteleaved arrays (0, 1, 2, 3)
 * 
 * Then the function returns the maximum 2 values among the 2 values
 */
void max_v_interl_2D(float *in_interl, uint32_t len, float* max_2_vals);


void max_v_interl_2D_f16(float16_t *in_interl, uint32_t len, float16_t* max_2_vals);

/**
 * Computes the maximum values over an array of 4 interleaved values.
 * 
 * Considering the following input:
 * [A0, A1, A2, A3, B0, B1, B2, B3]
 * 
 * The input length is considered to be 2 * 4 = 8.
 * That is 2 values (A and B) for 4 inteleaved arrays (0, 1, 2, 3)
 * 
 * Then the function returns the maximum 4 values among the 2 values
 */
void max_v_interl_4D(float *in_interl, uint32_t len, float* max_4_vals);


void max_v_interl_4D_f16(float16_t *in_interl, uint32_t len, float16_t* max_4_vals);



void maxPool_interleavedND(tensor3D_t *input_interl, uint8_t pool_size, uint8_t stride, uint8_t interl_factor, tensor3D_t *out){

    uint32_t nelems_channel = input_interl->dim.height * input_interl->dim.width;

    // Compute the output size
    uint32_t out_h = ((input_interl->dim.height - pool_size) / stride) + 1 ;
    uint32_t out_w = (((input_interl->dim.width / interl_factor) - pool_size) / stride) + 1; // The width has to be computed starting from the non-interleaved one
    out_w *= interl_factor;     // Return to interleaved dimentions

    dim3D_t out_dim = {
        .depth = input_interl->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // printf("In dims: %d %d %d\n", input_interl->dim.depth, input_interl->dim.height, input_interl->dim.width);
    // printf("Out dims: %d %d %d\n", out_dim.depth, out_dim.height, out_dim.width);

    // Allocate a new output tensor
    float *out_tensor = (float*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float));

    // Allocate the patch (multi-channel)
    float *flat_patches = (float*)malloc(input_interl->dim.depth * pool_size * (pool_size * interl_factor) * sizeof(float));

    // Allocate space to return the maximum values of the interleaved patch (channel)
    float *max_values = (float*)malloc(interl_factor * sizeof(float));

    for(int r=0; r<out_h; r++){

        // Here the upperbound is the width/interl_factor since so that I have a good loop-variable to compute index
        for(int c=0; c<(out_w / interl_factor); c++){

            dim3D_t patch_idx = {
                .depth=0,
                .height=(r * stride),    // Here I assume that the stride is always equal to the kernel for MaxPooling
                .width=(c * stride) * interl_factor
            };

            // printf("PAtch idx: %d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);

            dim3D_t patch_size = {
                .depth = input_interl->dim.depth,
                .height = pool_size,
                .width = pool_size * interl_factor
            };

            get_3Dpatch(input_interl->tensor, 
                        &input_interl->dim, 
                        &patch_idx, 
                        &patch_size, 
                        nelems_channel, 
                        flat_patches);
            
            // print_volume(flat_patches, &patch_size);


            if(interl_factor == 2){
                for(int ch=0; ch<input_interl->dim.depth; ch++){
                    max_v_interl_2D(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width, max_values);


                    uint32_t index = (ch * out_h * out_w) + (r * out_w) + (c * interl_factor);

                    out_tensor[index + 0] = max_values[0];
                    out_tensor[index + 1] = max_values[1];
                }

            } else if(interl_factor == 4){
                for(int ch=0; ch<input_interl->dim.depth; ch++){
                    max_v_interl_4D(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width, max_values);


                    uint32_t index = (ch * out_h * out_w) + (r * out_w) + (c * interl_factor);

                    // printf("(%d * %d * %d) + (%d * %d) + %d = %d\n",    ch, out_h, out_w,
                    //                                                     r, out_w,
                    //                                                     (c * 4),
                    //                                                     index);
                    // printf("index : %d\n", index);

                    out_tensor[index + 0] = max_values[0];
                    out_tensor[index + 1] = max_values[1];
                    out_tensor[index + 2] = max_values[2];
                    out_tensor[index + 3] = max_values[3];
                }
            }
        }
    }

    free(flat_patches);
    free(max_values);

    // Free the original output tensor so to use the newly computed one
    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}






void maxPool_interleavedND_f16(tensor3D_f16_t *input_interl, uint8_t pool_size, uint8_t stride, uint8_t interl_factor, tensor3D_f16_t *out){

    uint32_t nelems_channel = input_interl->dim.height * input_interl->dim.width;

    // Compute the output size
    uint32_t out_h = ((input_interl->dim.height - pool_size) / stride) + 1 ;
    uint32_t out_w = (((input_interl->dim.width / interl_factor) - pool_size) / stride) + 1; // The width has to be computed starting from the non-interleaved one
    out_w *= interl_factor;     // Return to interleaved dimentions

    dim3D_t out_dim = {
        .depth = input_interl->dim.depth,
        .height = out_h,
        .width = out_w
    };

    // printf("In dims: %d %d %d\n", input_interl->dim.depth, input_interl->dim.height, input_interl->dim.width);
    // printf("Out dims: %d %d %d\n", out_dim.depth, out_dim.height, out_dim.width);

    // Allocate a new output tensor
    float16_t *out_tensor = (float16_t*)malloc(out_dim.depth * out_dim.height * out_dim.width * sizeof(float16_t));

    // Allocate the patch (multi-channel)
    float16_t *flat_patches = (float16_t*)malloc(input_interl->dim.depth * pool_size * (pool_size * interl_factor) * sizeof(float16_t));

    // Allocate space to return the maximum values of the interleaved patch (channel)
    float16_t *max_values = (float16_t*)malloc(interl_factor * sizeof(float16_t));

    for(int r=0; r<out_h; r++){

        // Here the upperbound is the width/interl_factor since so that I have a good loop-variable to compute index
        for(int c=0; c<(out_w / interl_factor); c++){

            dim3D_t patch_idx = {
                .depth=0,
                .height=(r * stride),    // Here I assume that the stride is always equal to the kernel for MaxPooling
                .width=(c * stride) * interl_factor
            };

            // printf("PAtch idx: %d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);

            dim3D_t patch_size = {
                .depth = input_interl->dim.depth,
                .height = pool_size,
                .width = pool_size * interl_factor
            };

            get_3Dpatch_f16(input_interl->tensor, 
                        &input_interl->dim, 
                        &patch_idx, 
                        &patch_size, 
                        nelems_channel, 
                        flat_patches);
            
            // print_volume(flat_patches, &patch_size);


            if(interl_factor == 2){
                for(int ch=0; ch<input_interl->dim.depth; ch++){
                    max_v_interl_2D_f16(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width, max_values);


                    uint32_t index = (ch * out_h * out_w) + (r * out_w) + (c * interl_factor);

                    out_tensor[index + 0] = max_values[0];
                    out_tensor[index + 1] = max_values[1];
                }

            } else if(interl_factor == 4){
                for(int ch=0; ch<input_interl->dim.depth; ch++){
                    max_v_interl_4D_f16(&flat_patches[ch * patch_size.height * patch_size.width], patch_size.height * patch_size.width, max_values);


                    uint32_t index = (ch * out_h * out_w) + (r * out_w) + (c * interl_factor);

                    // printf("(%d * %d * %d) + (%d * %d) + %d = %d\n",    ch, out_h, out_w,
                    //                                                     r, out_w,
                    //                                                     (c * 4),
                    //                                                     index);
                    // printf("index : %d\n", index);

                    out_tensor[index + 0] = max_values[0];
                    out_tensor[index + 1] = max_values[1];
                    out_tensor[index + 2] = max_values[2];
                    out_tensor[index + 3] = max_values[3];
                }
            }
        }
    }

    free(flat_patches);
    free(max_values);

    // Free the original output tensor so to use the newly computed one
    free(out->tensor);

    out->tensor = out_tensor;
    out->dim = out_dim;
}












void max_v_interl_2D(float *in_interl, uint32_t len, float* max_2_vals){

    max_2_vals[0] = in_interl[0];
    max_2_vals[1] = in_interl[1];

    uint32_t single_len = len / 2;

    for(uint32_t i=0; i<single_len; i++){
        for(uint8_t n=0; n<2; n++){
            if(in_interl[(i*2)+n] >= max_2_vals[n]){
                max_2_vals[n] = in_interl[(i*2)+n];
            }
        }
    }
}


void max_v_interl_2D_f16(float16_t *in_interl, uint32_t len, float16_t* max_2_vals){

    max_2_vals[0] = in_interl[0];
    max_2_vals[1] = in_interl[1];

    uint32_t single_len = len / 2;

    for(uint32_t i=0; i<single_len; i++){
        for(uint8_t n=0; n<2; n++){
            if(in_interl[(i*2)+n] >= max_2_vals[n]){
                max_2_vals[n] = in_interl[(i*2)+n];
            }
        }
    }
}




void max_v_interl_4D(float *in_interl, uint32_t len, float* max_4_vals){

    max_4_vals[0] = in_interl[0];
    max_4_vals[1] = in_interl[1];
    max_4_vals[2] = in_interl[2];
    max_4_vals[3] = in_interl[3];

    uint32_t single_len = len / 4;

    for(uint32_t i=0; i<single_len; i++){
        for(uint8_t n=0; n<4; n++){
            if(in_interl[(i*4)+n] >= max_4_vals[n]){
                max_4_vals[n] = in_interl[(i*4)+n];
            }
        }
    }
}





void max_v_interl_4D_f16(float16_t *in_interl, uint32_t len, float16_t* max_4_vals){

    max_4_vals[0] = in_interl[0];
    max_4_vals[1] = in_interl[1];
    max_4_vals[2] = in_interl[2];
    max_4_vals[3] = in_interl[3];

    uint32_t single_len = len / 4;

    for(uint32_t i=0; i<single_len; i++){
        for(uint8_t n=0; n<4; n++){
            if(in_interl[(i*4)+n] >= max_4_vals[n]){
                max_4_vals[n] = in_interl[(i*4)+n];
            }
        }
    }
}