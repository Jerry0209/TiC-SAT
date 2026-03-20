#ifndef _CONV_DEF_H_
#define _CONV_DEF_H_

#include <inttypes.h>
#include <arm_sve.h>


#define CONV_OUT_DIM(dim, k_size, stride, padding)  (((dim + 2 * padding) < k_size) ? 0 : (((dim - k_size + (2 * padding)) / stride) + 1))



/*
    Struct type to define 3D dimensions of a tensor
*/
typedef struct dim3D_struc {
    uint32_t depth;
    uint32_t height;
    uint32_t width;
} dim3D_t;



typedef struct tensor3D_struct {
    dim3D_t dim;
    float *tensor;
} tensor3D_t;


typedef struct tensor3D_f16_t_struct {
    dim3D_t dim;
    float16_t *tensor;
} tensor3D_f16_t;



/*
    Struct type defining the characteristics of a convolution
*/
typedef struct conv_struct {
    uint32_t in_ch;         // Number of input channels
    uint32_t out_ch;        // Number of output channels
    
    dim3D_t input_dim;      // Dimensions of the input
    dim3D_t output_dim;     // Dimensions of the output

    dim3D_t kernel_dim;     // Dimensions of the kernel

    // uint32_t n_filters;     // Number of kernels
    uint32_t stride;        // Stride
    uint32_t padding;       // Padding

    uint32_t n_elems_k_channel; // How many elements per each kernel channel
    uint32_t n_elems_in_channel;  // How many elements per each input channel

    uint32_t n_idxs_word_channel;   // How many words of packed indexes are there per channel
} conv_t;





void get_3Dpatch(float *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float *res);

void get_3Dpatch_f16(float16_t *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float16_t *res);


tensor3D_t* check_padding(tensor3D_t *raw_input, uint32_t padding);


void pad_input2D_old(float *in, dim3D_t *in_dim, uint32_t padding, float *res);
void pad_input2D(float *in, dim3D_t *in_dim, uint32_t padding, float *res, dim3D_t *res_dim);
void pad_input2D_interleavedND(float *in, dim3D_t *in_dim, uint32_t padding, uint8_t interl_factor, float *res, dim3D_t *res_dim);

float vect_vect_mult(float *v0, float *v1, int size);


void print_volume(float *vol, dim3D_t *dims);
void print_volume_f16(float16_t *vol, dim3D_t *dims);

void print_interleaved_out(tensor3D_t *out_interleaved, uint8_t n_interleaved);
void print_interleaved_out_f16(tensor3D_f16_t *out_interleaved, uint8_t n_interleaved);

void merge_interleaved_2D(float *in0, float *in1, uint32_t len, float *out);

void merge_interleaved_4D(float *in0, float *in1, float *in2, float *in3, uint32_t len, float *out);

void merge_interleaved_4D_uint32(uint32_t *in0, uint32_t *in1, uint32_t *in2, uint32_t *in3, uint32_t len, uint32_t *out);

void merge_interleaved_4D_uint16(uint16_t *in0, uint16_t *in1, uint16_t *in2, uint16_t *in3, uint16_t len, uint16_t *out);

void merge_interleaved_4D_f16(float16_t *in0, float16_t *in1, float16_t *in2, float16_t *in3, uint32_t len, float16_t *out);

#endif