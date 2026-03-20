#ifndef _LENET_DEF_H_
#define _LENET_DEF_H_

#include <conv_def.h>
#include <dense_exec.h>

#include <./../lenet_definitions/input_image.h>

#include <./../lenet_definitions/conv_header_0.h>
#include <./../lenet_definitions/conv_header_2.h>
#include <./../lenet_definitions/dense_header_4.h>
#include <./../lenet_definitions/dense_header_5.h>
#include <./../lenet_definitions/dense_header_6.h>


#define PAD_DIMS(dim, padding)  (dim + (2 * padding))

// #define CONV_OUT_DIM(dim, k_size, stride, padding)  (((dim - k_size + (2 * padding)) / stride) + 1)
#define MAX_POOL_DIM(dim, size) (int)(dim / size)

//////////////////////////////////////////////////
//      CONV 0                                  //
//////////////////////////////////////////////////

static const dim3D_t input_dim_0 = {
    .depth = IN_D,
    .height = IN_H,
    .width = IN_W
};

tensor3D_t input_0 = {
    .dim = input_dim_0,
    .tensor = input_flat
};


static const dim3D_t kernel_dim_0 = {
    .depth = K_D_0,
    .height = K_H_0,
    .width = K_W_0
};

static const dim3D_t output_dims_0 = {
        .depth = OUT_CH_0,
        .height = CONV_OUT_DIM(input_dim_0.height, K_H_0, STRIDE_0, PADDING_0),
        .width = CONV_OUT_DIM(input_dim_0.width, K_W_0, STRIDE_0, PADDING_0),
};

static conv_t conv_0 = {
    .in_ch = IN_CH_0,
    .out_ch = OUT_CH_0,
    .input_dim = input_dim_0,
    .output_dim = output_dims_0,
    .kernel_dim = kernel_dim_0,
    .stride = STRIDE_0,
    .padding = PADDING_0,
    .n_elems_k_channel = N_K_ELEMS_PER_CHANNEL_0,
    .n_elems_in_channel = PAD_DIMS(input_dim_0.height, PADDING_2) * PAD_DIMS(input_dim_0.width, PADDING_2),
    .n_idxs_word_channel = N_WORDS_IDX_PER_CH_0
};


//////////////////////////////////////////////////
//////////////////////////////////////////////////


#define POOL_SIZE_1 2
#define POOL_STRIDE_1 2


//////////////////////////////////////////////////
//      CONV 2                                  //
//////////////////////////////////////////////////

static const dim3D_t input_dim_2 = {
    .depth = MAX_POOL_DIM(output_dims_0.depth, POOL_SIZE_1),
    .height = MAX_POOL_DIM(output_dims_0.height, POOL_SIZE_1),
    .width = MAX_POOL_DIM(output_dims_0.width, POOL_SIZE_1)
};

static const dim3D_t kernel_dim_2 = {
    .depth = K_D_2,
    .height = K_H_2,
    .width = K_W_2
};

static const dim3D_t output_dims_2 = {
        .depth = OUT_CH_2,
        .height = CONV_OUT_DIM(input_dim_2.height, K_H_2, STRIDE_2, PADDING_2),
        .width = CONV_OUT_DIM(input_dim_2.width, K_W_2, STRIDE_2, PADDING_2),
};


static conv_t conv_2 = {
    .in_ch = IN_CH_2,
    .out_ch = OUT_CH_2,
    .input_dim = input_dim_2,
    .output_dim = output_dims_2,
    .kernel_dim = kernel_dim_2,
    .stride = STRIDE_2,
    .padding = PADDING_2,
    .n_elems_k_channel = N_K_ELEMS_PER_CHANNEL_2,
    .n_elems_in_channel = PAD_DIMS(input_dim_2.height, PADDING_2) * PAD_DIMS(input_dim_2.width, PADDING_2),
    .n_idxs_word_channel = N_WORDS_IDX_PER_CH_2
};


//////////////////////////////////////////////////
//////////////////////////////////////////////////


#define POOL_SIZE_3 2
#define POOL_STRIDE_3 2


//////////////////////////////////////////////////
//      DENSE 4                                 //
//////////////////////////////////////////////////


static const dense_t dense_4 = {
    .in_size = INPUT_SIZE_4,
    .out_size = OUTPUT_SIZE_4,
    .n_words_row = N_WORDS_ROW_4
};
//////////////////////////////////////////////////
//////////////////////////////////////////////////




//////////////////////////////////////////////////
//      DENSE 5                                 //
//////////////////////////////////////////////////

static const dense_t dense_5 = {
    .in_size = INPUT_SIZE_5,
    .out_size = OUTPUT_SIZE_5,
    .n_words_row = N_WORDS_ROW_5
};
//////////////////////////////////////////////////
//////////////////////////////////////////////////




//////////////////////////////////////////////////
//      DENSE 6                                 //
//////////////////////////////////////////////////

static const dense_t dense_6 = {
    .in_size = INPUT_SIZE_6,
    .out_size = OUTPUT_SIZE_6,
    .n_words_row = N_WORDS_ROW_6
};
//////////////////////////////////////////////////
//////////////////////////////////////////////////


#endif