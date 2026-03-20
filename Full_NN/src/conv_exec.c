#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SVE_implementations.h>

#include <codebooks_def.h>
#include <conv_exec.h>

float vect_vect_mult(float *v0, float *v1, int size);



/**
 * Extract a 3D patch from a given input volume
 * 
 * Both the inputs and resulting patch should be allocated as a 1D array!
 * 
 * @param *in           :   pointer to the input volume
 * @param *input_dim    :   pointer to the dimensions of the input volume
 * @param *patch_index  :   dimension indexes to the patch to extract
 * @param *patch_dim    :   pointer to the dimensions of the patch to extract
 * @param *res          :   pointer to the result (patch)
 */
void get_3Dpatch(float *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float *res){

    int index = 0;
    // printf("Dim %d %d %d\n", input_dim->depth, input_dim->height, input_dim->width);
    // printf("Patch idx: %d %d %d\n", patch_index->depth, patch_index->height, patch_index->width);
    // printf("Patch dim: %d %d %d\n", patch_dim->depth, patch_dim->height, patch_dim->width);

    for(int d=0; d<patch_dim->depth; d++){
        for(int h=0; h<patch_dim->height; h++){
            for(int w=0; w<patch_dim->width; w++){
                // printf(">> d: %d | h: %d | w: %d\n", d, h, w);
                index = (d * n_elems_slice);                                // Starting index (patch index depth is always 0 since we force input and kernel to have same depth)
                index += ((patch_index->height + h) * input_dim->width);    // Gets the correct index for the height
                index += (patch_index->width + w);                          // Gets the correct index for the width
                // printf(" --> res[%d] = in[%d + %d + %d = %d] = %f\n\n", (d * patch_dim->height * patch_dim->width) + (h * patch_dim->width) + w, 
                //                                                     (d * n_elems_slice),
                //                                                     ((patch_index->height + h) * input_dim->width),
                //                                                     (patch_index->width + w),
                //                                                     index,
                //                                                     in[index]);
                res[(d * patch_dim->height * patch_dim->width) + (h * patch_dim->width) + w] = in[index];
            }
        }
    }

    // print_volume(res, patch_dim);
    // exit(0);
}


void get_3Dpatch_f16(float16_t *in, dim3D_t *input_dim, dim3D_t *patch_index, dim3D_t *patch_dim, uint32_t n_elems_slice, float16_t *res){

    int index = 0;
    // printf("Dim %d %d %d\n", input_dim->depth, input_dim->height, input_dim->width);
    // printf("Patch idx: %d %d %d\n", patch_index->depth, patch_index->height, patch_index->width);
    // printf("Patch dim: %d %d %d\n", patch_dim->depth, patch_dim->height, patch_dim->width);

    for(int d=0; d<patch_dim->depth; d++){
        for(int h=0; h<patch_dim->height; h++){
            for(int w=0; w<patch_dim->width; w++){
                // printf(">> d: %d | h: %d | w: %d\n", d, h, w);
                index = ((patch_index->depth + d) * n_elems_slice);         // Starting index (patch index depth is always 0 since we force input and kernel to have same depth)
                index += ((patch_index->height + h) * input_dim->width);    // Gets the correct index for the height
                index += (patch_index->width + w);                          // Gets the correct index for the width
                // printf(" --> res[%d] = in[%d + %d + %d = %d] = %f\n\n", (d * patch_dim->height * patch_dim->width) + (h * patch_dim->width) + w, 
                //                                                     ((patch_index->depth + d) * n_elems_slice),
                //                                                     ((patch_index->height + h) * input_dim->width),
                //                                                     (patch_index->width + w),
                //                                                     index,
                //                                                     in[index]);
                // printf("Index: (%d * %d * %d) + (%d * %d) + %d = %d\n",
                //     d, patch_dim->height, patch_dim->width, h, patch_dim->width, + w,
                //     (d * patch_dim->height * patch_dim->width) + (h * patch_dim->width) + w
                // );
                res[(d * patch_dim->height * patch_dim->width) + (h * patch_dim->width) + w] = in[index];
            }
        }
    }
    // print_volume(res, patch_dim);
    // exit(0);
}




/**
 * Pads the input matrix with 0 values.
 * 
 * @param *in       :   pointer to a matrix stored in row-major order
 * @param *in_dim   :   pointer to the struct with the dimensions of the input
 * @param padding   :   amount of padding to perform
 * @param *res      :   pointer to the resulting padding array
 */
void pad_input2D_old(float *in, dim3D_t *in_dim, uint32_t padding, float *res){

    // printf("VOL:\n");
    // printf("DIM: %d %d\n", in_dim->height, in_dim->width);
    // print_volume(in, in_dim);

    // Used to index the next value to add
    uint32_t val_idx = 0;

    // printf("Padding %d\n", padding);
    // printf("%d %d\n", in_dim->height, in_dim->width);

    // Number of padded values to add at the beginning (and end) of the row-major input
    uint32_t n_start_pad_values = (padding * in_dim->width) + (2 * padding * padding);

    // printf("PAd values; %d\n", n_start_pad_values);

    // Pad the beginning and end of the matrix
    for(uint32_t i=0; i<n_start_pad_values; i++){
        res[i] = 0.0;
        res[(in_dim->height * in_dim->width) + i] = 0.0;
        // printf("[%d] and [%d]\n", i, (in_dim->height * in_dim->width) + i);
    }
    val_idx += n_start_pad_values;

    // Loop through the input rows
    for(uint32_t r=0; r<in_dim->height; r++){

        // Append pad values at the beginning and end of each row
        for(uint32_t i=0; i<padding; i++){
            res[val_idx + i] = 0.0;                 // Beginning
            res[val_idx + in_dim->width + i + 1] = 0.0;  // End
            // printf("val_idx: %d  |  [%d] | ... | [%d]  =  %f | ... | %f\n", val_idx, 
            //                                                             val_idx + i, 
            //                                                             val_idx + in_dim->width + i, 
            //                                                             res[val_idx + i], 
            //                                                             res[val_idx + in_dim->width + i]);
        }
        
        val_idx += padding;

        // printf("Row: %d, At %d, From %d, Size %d --> %f\n", r, val_idx, r*in_dim->width, in_dim->width, in[r*in_dim->width]);

        // Copy the original values
        memcpy(&res[val_idx], &in[r*in_dim->width], in_dim->width*sizeof(*in));

        val_idx += padding + in_dim->width;  // Adjust to the size of the padded row
    }

    // printf("RES:\n");
    // printf("DIM: %d %d %d\n", in_dim->height, in_dim->width);
    // print_volume(res, in_dim);
}



/**
 * Pads the input matrix with 0 values.
 * 
 * @param *in       :   pointer to a matrix stored in row-major order
 * @param *in_dim   :   pointer to the struct with the dimensions of the input
 * @param padding   :   amount of padding to perform
 * @param *res      :   pointer to the resulting padding array
 * @param *res_dim  :   pointer to the struct with the dimensions of the padded result
 */
void pad_input2D(float *in, dim3D_t *in_dim, uint32_t padding, float *res, dim3D_t *res_dim){
    
    // Set everything to 0.0
    memset(res, 0.0, (res_dim->height * res_dim->width) * sizeof(float));

    uint32_t res_row_idx = 0; // In a row, it index the position from which to copy the original values

    // Loop through the input rows
    for(uint32_t r=0; r<in_dim->height; r++){

        res_row_idx = (res_dim->width * (r + padding)) + padding;

        // Copy the original values
        memcpy(&res[res_row_idx], &in[r*in_dim->width], in_dim->width*sizeof(*in));

    }

}



void pad_input2D_f16(float16_t *in, dim3D_t *in_dim, uint32_t padding, float16_t *res, dim3D_t *res_dim){
    
    // Set everything to 0.0
    memset(res, 0.0, (res_dim->height * res_dim->width) * sizeof(float16_t));

    uint32_t res_row_idx = 0; // In a row, it index the position from which to copy the original values

    // Loop through the input rows
    for(uint32_t r=0; r<in_dim->height; r++){

        res_row_idx = (res_dim->width * (r + padding)) + padding;

        // Copy the original values
        memcpy(&res[res_row_idx], &in[r*in_dim->width], in_dim->width*sizeof(*in));

    }

}






/**
 * Pads the interleaved input matrix with 0 values.
 * Since it is interleaved, the height will be padded normally, but the 
 * width will have N=4 times padding (having N=interleaved factor)
 * 
 * @param *in           :   pointer to a matrix stored in row-major order
 * @param *in_dim       :   pointer to the struct with the dimensions of the input
 * @param padding       :   amount of padding to perform
 * @param interl_factor :   how many interleaved arrays are there
 * @param *res          :   pointer to the resulting padding array
 * @param *res_dim      :   pointer to the struct with the dimensions of the padded result
 */
void pad_input2D_interleavedND(float *in, dim3D_t *in_dim, uint32_t padding, uint8_t interl_factor, float *res, dim3D_t *res_dim){
    
    // Set everything to 0.0
    memset(res, 0.0, (res_dim->height * res_dim->width) * sizeof(float));

    uint32_t res_row_idx = 0;   // In a row, it index the position from which to copy the original values

    // Loop through the input rows
    for(uint32_t r=0; r<in_dim->height; r++){

        res_row_idx = (res_dim->width * (r + padding)) + (interl_factor * padding); // 4 since the interleaved factor is 4

        // Copy the original values
        memcpy(&res[res_row_idx], &in[r*in_dim->width], in_dim->width*sizeof(*in));

    }

}




void pad_input2D_interleavedND_f16(float16_t *in, dim3D_t *in_dim, uint32_t padding, uint8_t interl_factor, float16_t *res, dim3D_t *res_dim){
    
    // Set everything to 0.0
    memset(res, 0.0, (res_dim->height * res_dim->width) * sizeof(float16_t));

    uint32_t res_row_idx = 0;   // In a row, it index the position from which to copy the original values

    // Loop through the input rows
    for(uint32_t r=0; r<in_dim->height; r++){

        res_row_idx = (res_dim->width * (r + padding)) + (interl_factor * padding); // 4 since the interleaved factor is 4

        // Copy the original values
        memcpy(&res[res_row_idx], &in[r*in_dim->width], in_dim->width*sizeof(*in));

    }

}






/**
 * Checks if padding is required. 
 * 
 * If yes, it pads the matrix and returns the new padded input.
 * 
 * If no, it allocs a copy of the input array and returns it
 * 
 * 
 * @param *raw_input       :    poiter to the tensor type with the raw (non-padded) input
 * @param padding   :   amount of padding to perform
 * 
 * @returns a tensor type with the new padded tensor (if padding is required)
 */
tensor3D_t* check_padding(tensor3D_t *raw_input, uint32_t padding){

    float *new_in;
    dim3D_t new_dims;

    new_dims.depth = raw_input->dim.depth;

    if(padding != 0){
        // printf("%d %d\n", raw_input->dim.height, raw_input->dim.width);
        // print_volume(raw_input->tensor, &raw_input->dim);

        // Compute new dimension and allocate pointer for padded input
        uint32_t new_height = raw_input->dim.height + (2 * padding);
        uint32_t new_width = raw_input->dim.width + (2 * padding);
        new_in = (float*)malloc(raw_input->dim.depth * new_height * new_width * sizeof(float));

        new_dims.height = new_height;
        new_dims.width = new_width;

        // Pads the tensor
        for(uint32_t ch=0; ch<raw_input->dim.depth; ch++){
            // pad_input2D_old(&raw_input->tensor[ch*raw_input->dim.height*raw_input->dim.width], &raw_input->dim, padding, &new_in[ch * new_height * new_width]);
            pad_input2D(&raw_input->tensor[ch*raw_input->dim.height*raw_input->dim.width], &raw_input->dim, padding, &new_in[ch * new_height * new_width], &new_dims);
        }
        
        // new_dims.height = new_height;
        // new_dims.width = new_width;

        // printf("%d %d\n", new_dims.height, new_dims.width);
        // print_volume(new_in, &new_dims);
        
    } else {
        // Allocate same space as input (non padded) and copy the content
        new_in = (float*)malloc((raw_input->dim.depth * raw_input->dim.height * raw_input->dim.width) * sizeof(float));
        memcpy(new_in, raw_input->tensor, (raw_input->dim.depth * raw_input->dim.height * raw_input->dim.width) * sizeof(float)); 

        new_dims.height = raw_input->dim.height;
        new_dims.width = raw_input->dim.width;
    }

    tensor3D_t *res = (tensor3D_t*)malloc(sizeof(tensor3D_t));
    res->dim = new_dims;
    res->tensor = new_in;

    return res;
}




tensor3D_f16_t* check_padding_f16(tensor3D_f16_t *raw_input, uint32_t padding){

    float16_t *new_in;
    dim3D_t new_dims;

    new_dims.depth = raw_input->dim.depth;

    if(padding != 0){
        // printf("%d %d\n", raw_input->dim.height, raw_input->dim.width);
        // print_volume(raw_input->tensor, &raw_input->dim);

        // Compute new dimension and allocate pointer for padded input
        uint32_t new_height = raw_input->dim.height + (2 * padding);
        uint32_t new_width = raw_input->dim.width + (2 * padding);
        new_in = (float16_t*)malloc(raw_input->dim.depth * new_height * new_width * sizeof(float16_t));

        new_dims.height = new_height;
        new_dims.width = new_width;

        // Pads the tensor
        for(uint32_t ch=0; ch<raw_input->dim.depth; ch++){
            // pad_input2D_old(&raw_input->tensor[ch*raw_input->dim.height*raw_input->dim.width], &raw_input->dim, padding, &new_in[ch * new_height * new_width]);
            pad_input2D_f16(&raw_input->tensor[ch*raw_input->dim.height*raw_input->dim.width], &raw_input->dim, padding, &new_in[ch * new_height * new_width], &new_dims);
        }
        
        // new_dims.height = new_height;
        // new_dims.width = new_width;

        // printf("%d %d\n", new_dims.height, new_dims.width);
        // print_volume(new_in, &new_dims);
        
    } else {
        // Allocate same space as input (non padded) and copy the content
        new_in = (float16_t*)malloc((raw_input->dim.depth * raw_input->dim.height * raw_input->dim.width) * sizeof(float16_t));
        memcpy(new_in, raw_input->tensor, (raw_input->dim.depth * raw_input->dim.height * raw_input->dim.width) * sizeof(float16_t)); 

        new_dims.height = raw_input->dim.height;
        new_dims.width = raw_input->dim.width;
    }

    tensor3D_f16_t *res = (tensor3D_f16_t*)malloc(sizeof(tensor3D_f16_t));
    res->dim = new_dims;
    res->tensor = new_in;

    return res;
}







void check_padding_interleavedND(tensor3D_t *raw_in_interl, uint32_t padding, uint8_t interl_factor, tensor3D_t* res){
    
    float *new_in;
    dim3D_t new_dims;

    new_dims.depth = raw_in_interl->dim.depth;

    if(padding != 0){
        // Compute new dimension and allocate pointer for padded input
        uint32_t new_height = raw_in_interl->dim.height + (2 * padding);
        uint32_t new_width = raw_in_interl->dim.width + ((2 * padding) * interl_factor);
        new_in = (float*)malloc(raw_in_interl->dim.depth * new_height * new_width * sizeof(float));

        new_dims.height = new_height;
        new_dims.width = new_width;

        // Pads the tensor
        for(uint32_t ch=0; ch<raw_in_interl->dim.depth; ch++){
            pad_input2D_interleavedND(  &raw_in_interl->tensor[ch*raw_in_interl->dim.height*raw_in_interl->dim.width], 
                                        &raw_in_interl->dim, 
                                        padding, 
                                        interl_factor,
                                        &new_in[ch * new_height * new_width], &new_dims);
        }
    } else {

        // Allocate same space as input (non padded) and copy the content
        new_in = (float*)malloc((raw_in_interl->dim.depth * raw_in_interl->dim.height * raw_in_interl->dim.width) * sizeof(float));
        memcpy(new_in, raw_in_interl->tensor, (raw_in_interl->dim.depth * raw_in_interl->dim.height * raw_in_interl->dim.width) * sizeof(float)); 

        new_dims.height = raw_in_interl->dim.height;
        new_dims.width = raw_in_interl->dim.width;
    }

    res->dim = new_dims;
    res->tensor = new_in;
}



void check_padding_interleavedND_f16(tensor3D_f16_t *raw_in_interl, uint32_t padding, uint8_t interl_factor, tensor3D_f16_t* res){
    

    float16_t *new_in;
    dim3D_t new_dims;

    new_dims.depth = raw_in_interl->dim.depth;

    if(padding != 0){
        // Compute new dimension and allocate pointer for padded input
        uint32_t new_height = raw_in_interl->dim.height + (2 * padding);
        uint32_t new_width = raw_in_interl->dim.width + ((2 * padding) * interl_factor);
        new_in = (float16_t*)malloc(raw_in_interl->dim.depth * new_height * new_width * sizeof(float16_t));

        new_dims.height = new_height;
        new_dims.width = new_width;

        // Pads the tensor
        for(uint32_t ch=0; ch<raw_in_interl->dim.depth; ch++){
            pad_input2D_interleavedND_f16(  &raw_in_interl->tensor[ch*raw_in_interl->dim.height*raw_in_interl->dim.width], 
                                        &raw_in_interl->dim, 
                                        padding, 
                                        interl_factor,
                                        &new_in[ch * new_height * new_width], &new_dims);
        }

    } else {

        // Allocate same space as input (non padded) and copy the content
        new_in = (float16_t*)malloc((raw_in_interl->dim.depth * raw_in_interl->dim.height * raw_in_interl->dim.width) * sizeof(float16_t));
        memcpy(new_in, raw_in_interl->tensor, (raw_in_interl->dim.depth * raw_in_interl->dim.height * raw_in_interl->dim.width) * sizeof(float16_t)); 

        new_dims.height = raw_in_interl->dim.height;
        new_dims.width = raw_in_interl->dim.width;
    }

    res->dim = new_dims;
    res->tensor = new_in;
}






void conv3D_staticPatch_compact(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((kernel_d * (kernel_h * kernel_w) * (out_h * out_w)) * sizeof(float));

    // printf("Patches matrix: (%d) x (%d) = (%d)\n", kernel_d * (kernel_h * kernel_w), (out_h * out_w), (kernel_d * (kernel_h * kernel_w) * (out_h * out_w)));

    uint32_t patch_cnt = 0; // Counter to the index of the patch to extrac

    // Loop output rows
    for(int r=0; r<out_h; r++){

        // Loop output columns
        for(int c=0; c<out_w; c++){

            dim3D_t patch_idx = {
                .depth=0,
                .height=(r*conv_layer.stride),
                .width=(c*conv_layer.stride)
            };

            // printf("%d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);

            // printf("Index: %d\n", patch_cnt * (N_K_ELEMS_PER_CHANNEL));
            get_3Dpatch(input->tensor, 
                        &input->dim,
                        &patch_idx,
                        &conv_layer.kernel_dim,
                        conv_layer.n_elems_in_channel,
                        &flat_patches_matrix[patch_cnt * (conv_layer.n_elems_k_channel)]);

            // print_volume(&flat_patches_matrix[patch_cnt * (conv_layer.n_elems_k_channel)], &conv_layer.kernel_dim);

            patch_cnt++;
        }
    }

    // // Allocate space for the kernel with the float values extracted from the codebooks
    // float *kernel_values = (float*)malloc(kernel_ch * conv_layer.n_elems_k_channel * sizeof(float));

    uint8_t shamt = 0;
    uint8_t index = 0;

    // Loop through the kernel channels (output channels)
    for(int ch=0; ch<kernel_ch; ch++){

        // Allocate space for the kernel values (from the codebooks) of a channel
        float *kernel_channel_values = (float*)malloc(conv_layer.n_elems_k_channel * sizeof(float));

        // Keeps track of how many indexes have been extracted
        // Needed to stop at the right amount of indexes
        uint32_t idxs_used = 0; 

        // Extract the values from codebooks (one full channel of 3D kernel)
        for(int i=0; i<conv_layer.n_idxs_word_channel; i++){
            for(int j=0; j<IDXS_PER_WORD; j++){
                shamt = j * BITS_PER_CB;
                index = ((kernel[(ch * conv_layer.n_idxs_word_channel) + i] >> shamt) & IDX_MASK);
                // printf("%d  |  %d\n", kernel[(ch * N_WORDS_IDX_PER_CH) + i], index);
                kernel_channel_values[(i * IDXS_PER_WORD) + j] = codebook[index];
                // printf("Val[%d]: %f\n", index, codebook[index]);

                idxs_used++;
                if(idxs_used >= conv_layer.n_elems_k_channel){
                    break;
                }
            }
        }
        
        // Multiply all the patches by the kernel --> compute one full output channel
        for(int r=0; r<(out_h * out_w); r++){
            out->tensor[(ch * out_h * out_w) + r] = vect_vect_mult(kernel_channel_values,
                                                                    &flat_patches_matrix[(r * conv_layer.n_elems_k_channel)],
                                                                    (conv_layer.n_elems_k_channel)
                                                                  );
        }

        free(kernel_channel_values);
    }

    // free(kernel_values);
    free(flat_patches_matrix);
}





void conv3D_staticPatch_compact_tiled(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, uint32_t tile_out, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;

    uint32_t patch_cnt = 0; // Counter to the index of the patch to extrac

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in = tile_out * conv_layer.stride + in_tiles_overlap;
    int tile_in_step = tile_in - in_tiles_overlap;

    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // Matrix holding all the flattened patches extracted statically before the computations (only from a single input tile)
    float *flat_patches_matrix = (float*)malloc((kernel_d * (kernel_h * kernel_w) * (tile_out * tile_out)) * sizeof(float));

    // Allocate space for the kernel values (from the codebooks) of a channel
    float *kernel_channel_values = (float*)malloc(conv_layer.n_elems_k_channel * sizeof(float));

    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    
    // int out_idx = 0;
    int out_idx_so_far = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<conv_layer.input_dim.height; th+=tile_in_step){
        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in) >= conv_layer.input_dim.height ? (conv_layer.input_dim.height - th) : tile_in;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);

        if(out_tile_h_elems <= 0){
            continue;
        }
        
        for(int tw=0; tw<conv_layer.input_dim.width; tw+=tile_in_step){
        
            // system("m5 resetstats");
            next_free_idx = 0;

            uint32_t tile_w_elems = (tw + tile_in) >= conv_layer.input_dim.width ? (conv_layer.input_dim.width - tw) : tile_in;
            uint32_t out_tile_w_elems = CONV_OUT_DIM(tile_w_elems, kernel_w, conv_layer.stride, 0);
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            int in_tensor_idx = (th * conv_layer.input_dim.width) + tw;  // This is for the non-tiled input

            // Extract the patches for the current input tile

            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<out_tile_w_elems; c++){

                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride)
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &conv_layer.kernel_dim,
                                conv_layer.n_elems_in_channel,
                                &flat_patches_matrix[next_free_idx]);

                    next_free_idx += (conv_layer.kernel_dim.depth * conv_layer.kernel_dim.height * conv_layer.kernel_dim.width);
                    patch_cnt++;
                }
            }

            uint8_t shamt = 0;
            uint8_t index = 0;

            // Loop through the kernel channels (output channels)
            for(int ch=0; ch<kernel_ch; ch++){

                // Keeps track of how many indexes have been extracted
                // Needed to stop at the right amount of indexes
                uint32_t idxs_used = 0; 

                // Extract the values from codebooks (one full channel of 3D kernel)
                for(int i=0; i<conv_layer.n_idxs_word_channel; i++){
                    for(int j=0; j<IDXS_PER_WORD; j++){
                        shamt = j * BITS_PER_CB;
                        index = ((kernel[(ch * conv_layer.n_idxs_word_channel) + i] >> shamt) & IDX_MASK);

                        kernel_channel_values[idxs_used] = codebook[index];
                        
                        idxs_used++;
                        if(idxs_used >= conv_layer.n_elems_k_channel){
                            break;
                        }

                    }
                }

                ///////////////////////////////////////////
                //// OUTPUT STORED IN NON-TILED MANNER ////
                ///////////////////////////////////////////

                patch_cnt = 0;  // Reset to count the patches (columns) that are multiplied

                // Multiply all the patches by the kernel --> compute one full output channel //

                // Both loops iterate over the columns of the patches matrix (I use to loop to compute conveniently the output index)
                for(int r=0; r<out_tile_h_elems; r++){
                    for(int c=0; c<out_tile_w_elems; c++){
                        int out_idx = (ch * out_h * out_w) + (out_tile_h_idx * out_w) + (out_tile_w_idx) + (r * out_w) + c; 
                        
                        // printf("Out idx: %d = %f\n", out_idx, vect_vect_mult(kernel_channel_values, &flat_patches_matrix[(patch_cnt * conv_layer.n_elems_k_channel)], (conv_layer.n_elems_k_channel)));
                        // printf("In idx: %d\n", (patch_cnt * conv_layer.n_elems_k_channel));
                        out->tensor[out_idx] = vect_vect_mult(kernel_channel_values, &flat_patches_matrix[(patch_cnt * conv_layer.n_elems_k_channel)], (conv_layer.n_elems_k_channel));
                        patch_cnt++;
                    }
                }
            }

            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            out_tile_w_idx += out_tile_w_elems;
            tile_cnt++;
        }

        out_tile_h_idx += out_tile_h_elems;
    }
    
    free(flat_patches_matrix);
    free(kernel_channel_values);
}













void conv3D_staticPatch_compact_tiled_L1L2_noCB(conv_t conv_layer, tensor3D_t *input, const float *kernel, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out){

    // Workaround to make it work in noCB case
    conv_layer.n_idxs_word_channel = conv_layer.n_elems_k_channel;

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;

    uint32_t patch_cnt = 0; // Counter to the index of the patch to extrac

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in = tile_L2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step = tile_in - in_tiles_overlap;

    // Matrix holding all the flattened patches extracted statically before the computations (only from a single input tile)
    // printf("FIRST\n");
    float *flat_patches_matrix = (float*)malloc((kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)) * sizeof(float));
    // printf("SECOND\n");
    float *flat_patches_matrix_tiled_l1 = (float*)malloc((tile_L1 * tile_L1) * sizeof(float));   // Rearranged for L1 tiling

    // printf("FPM: %d\n", (kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)));
    // printf("FMP L1: %d\n", (tile_L1 * tile_L1 * IDXS_PER_WORD));

    // Allocate space for the kernel values (from the codebooks) of a channel
    // printf("THIRD\n");
    float *kernel_channel_values = (float*)malloc(tile_L1 * IDXS_PER_WORD * sizeof(float));
    // printf("FORTH\n");

    // printf("Kernel CH values: %d\n", tile_L1 * IDXS_PER_WORD);

    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<conv_layer.input_dim.height; th+=tile_in_step){
        // printf("TH: %d\n", th);
        int tile_h_elems = (th + tile_in) >= conv_layer.input_dim.height ? (conv_layer.input_dim.height - th) : tile_in;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<conv_layer.input_dim.width; tw+=tile_in_step){
            // printf("TW: %d\n", tw);
            next_free_idx = 0;

            uint32_t tile_w_elems = (tw + tile_in) >= conv_layer.input_dim.width ? (conv_layer.input_dim.width - tw) : tile_in;
            uint32_t out_tile_w_elems = CONV_OUT_DIM(tile_w_elems, kernel_w, conv_layer.stride, 0);
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\n=================\n");
            // printf("TILES: %d %d\n", th, tw);
            // printf("%d  >= %d ? %d : %d --> %d\n", (tw + tile_in), conv_layer.input_dim.width, (conv_layer.input_dim.width - tw), tile_in, tile_w_elems);
            // printf("IN  H elems: %d\n", tile_h_elems);
            // printf("IN  W elems: %d\n", tile_w_elems);
            // printf("OUT H elems: %d\n", out_tile_h_elems);
            // printf("OUT W elems: %d\n", out_tile_w_elems);

            int in_tensor_idx = (th * conv_layer.input_dim.width) + tw;  // This is for the non-tiled input

            // Extract the patches for the current input tile
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<out_tile_w_elems; c++){

                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride)
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &conv_layer.kernel_dim,
                                conv_layer.n_elems_in_channel,
                                &flat_patches_matrix[next_free_idx]);

                    next_free_idx += (conv_layer.kernel_dim.depth * conv_layer.kernel_dim.height * conv_layer.kernel_dim.width);
                    patch_cnt++;

                }
            }


            // Start tiling L1 //

            // uint8_t shamt = 0;
            // uint8_t index = 0;

            int start_idx_out_tile = (((th / tile_in_step) * tile_L2) * out_w) + ((tw / tile_in_step) * tile_L2);

            // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
            int n_cnt_rows_tile = 0;
            int n_cnt_cols_tile = 0;
            int tile_row_cnt = 0;
            int tile_col_cnt = 0;

            int n_proc_idxs = 0;    // Number of processed indexes per each output channel

            for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_L1){
                // printf("TWL1: %d\n", twl1);
                n_cnt_rows_tile = 0;
                n_cnt_cols_tile = 0;

                int tile_l1_w_elems = (twl1 + tile_L1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_L1;
                int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems)) >= conv_layer.n_elems_k_channel ? (conv_layer.n_elems_k_channel - n_proc_idxs) : (tile_l1_w_elems);
                
                for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*out_tile_w_elems); twl1_Tin+=tile_L1){
                    // printf("TWL1_TIN: %d\n", twl1_Tin);
                    int tile_l1_Tin_w_elems = (twl1_Tin + tile_L1) >= (out_tile_h_elems * out_tile_w_elems) ? ((out_tile_h_elems * out_tile_w_elems) - twl1_Tin) : tile_L1;
                    
                    // Define dimensions and extract the L1 tile from the matrix of flat patches
                    dim3D_t fpm_dim = {.depth=1, .height=(tile_L2 * tile_L2), .width=(kernel_d * (kernel_h * kernel_w))};
                    dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1};
                    dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile};
                    

                    get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
                    // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

                    for(int thl1=0; thl1<kernel_ch; thl1+=tile_L1){
                        // printf("THL1: %d\n", thl1);
                        int tile_l1_h_elems = (thl1 + tile_L1) >= kernel_ch ? (kernel_ch - thl1) : tile_L1;

                        // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                        // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                        // printf("N W elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                        // printf("N indexes in tile: %d\n", n_indexes_in_tile);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
                        // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                        int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);

                        // Loop over the output channels (the height of the tile)
                        for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
                            // printf("=================CH: %d\n", ch);

                            tile_row_cnt = n_cnt_rows_tile;
                            tile_col_cnt = n_cnt_cols_tile;
                            for(int patch_cnt=0; patch_cnt<tile_l1_Tin_w_elems; patch_cnt++){

                                int out_index = (ch * out_h * out_w) + start_idx_out_tile + (tile_row_cnt * out_w) + tile_col_cnt;

                                // printf(" ==> k_index = %d --> %f\n", k_index, kernel[k_index]);
                                out->tensor[out_index] += vect_vect_mult(&kernel[k_index], &flat_patches_matrix_tiled_l1[patch_cnt*n_indexes_in_tile], n_indexes_in_tile);


                                tile_col_cnt++;
                                if(tile_col_cnt >= out_tile_w_elems){
                                    tile_col_cnt=0;
                                    tile_row_cnt++;
                                }
                            }

                            k_index += n_indexes_in_tile;
                        }
                        // exit(0);
                    }

                    n_cnt_cols_tile = tile_col_cnt;
                    n_cnt_rows_tile = tile_row_cnt;
                }
                n_proc_idxs += n_indexes_in_tile;
            }
        }
    }
    

    #ifdef USE_BIAS
    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w] += bias[ch];
            }
        }
    }
    #endif


    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
    free(kernel_channel_values);
}
















void conv3D_staticPatch_compact_tiled_L1L2(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;

    uint32_t patch_cnt = 0; // Counter to the index of the patch to extrac

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in = tile_L2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step = tile_in - in_tiles_overlap;

    // Matrix holding all the flattened patches extracted statically before the computations (only from a single input tile)
    // printf("FIRST\n");
    float *flat_patches_matrix = (float*)malloc((kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)) * sizeof(float));
    // printf("SECOND\n");
    float *flat_patches_matrix_tiled_l1 = (float*)malloc((tile_L1 * tile_L1 * IDXS_PER_WORD) * sizeof(float));   // Rearranged for L1 tiling

    // printf("FPM: %d\n", (kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)));
    // printf("FMP L1: %d\n", (tile_L1 * tile_L1 * IDXS_PER_WORD));

    // Allocate space for the kernel values (from the codebooks) of a channel
    // printf("THIRD\n");
    float *kernel_channel_values = (float*)malloc(tile_L1 * IDXS_PER_WORD * sizeof(float));
    // printf("FORTH\n");

    // printf("Kernel CH values: %d\n", tile_L1 * IDXS_PER_WORD);

    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<conv_layer.input_dim.height; th+=tile_in_step){
        // printf("TH: %d\n", th);
        int tile_h_elems = (th + tile_in) >= conv_layer.input_dim.height ? (conv_layer.input_dim.height - th) : tile_in;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<conv_layer.input_dim.width; tw+=tile_in_step){
            // printf("TW: %d\n", tw);
            next_free_idx = 0;

            uint32_t tile_w_elems = (tw + tile_in) >= conv_layer.input_dim.width ? (conv_layer.input_dim.width - tw) : tile_in;
            uint32_t out_tile_w_elems = CONV_OUT_DIM(tile_w_elems, kernel_w, conv_layer.stride, 0);
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\n=================\n");
            // printf("TILES: %d %d\n", th, tw);
            // printf("%d  >= %d ? %d : %d --> %d\n", (tw + tile_in), conv_layer.input_dim.width, (conv_layer.input_dim.width - tw), tile_in, tile_w_elems);
            // printf("IN  H elems: %d\n", tile_h_elems);
            // printf("IN  W elems: %d\n", tile_w_elems);
            // printf("OUT H elems: %d\n", out_tile_h_elems);
            // printf("OUT W elems: %d\n", out_tile_w_elems);

            int in_tensor_idx = (th * conv_layer.input_dim.width) + tw;  // This is for the non-tiled input

            // Extract the patches for the current input tile
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<out_tile_w_elems; c++){

                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride)
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &conv_layer.kernel_dim,
                                conv_layer.n_elems_in_channel,
                                &flat_patches_matrix[next_free_idx]);

                    next_free_idx += (conv_layer.kernel_dim.depth * conv_layer.kernel_dim.height * conv_layer.kernel_dim.width);
                    patch_cnt++;

                }
            }


            // Start tiling L1 //

            uint8_t shamt = 0;
            uint8_t index = 0;

            int start_idx_out_tile = (((th / tile_in_step) * tile_L2) * out_w) + ((tw / tile_in_step) * tile_L2);

            // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
            int n_cnt_rows_tile = 0;
            int n_cnt_cols_tile = 0;
            int tile_row_cnt = 0;
            int tile_col_cnt = 0;

            int n_proc_idxs = 0;    // Number of processed indexes per each output channel

            for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_L1){
                // printf("TWL1: %d\n", twl1);
                n_cnt_rows_tile = 0;
                n_cnt_cols_tile = 0;

                int tile_l1_w_elems = (twl1 + tile_L1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_L1;
                int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel ? (conv_layer.n_elems_k_channel - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);
                
                for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*out_tile_w_elems); twl1_Tin+=tile_L1){
                    // printf("TWL1_TIN: %d\n", twl1_Tin);
                    int tile_l1_Tin_w_elems = (twl1_Tin + tile_L1) >= (out_tile_h_elems * out_tile_w_elems) ? ((out_tile_h_elems * out_tile_w_elems) - twl1_Tin) : tile_L1;
                    
                    // Define dimensions and extract the L1 tile from the matrix of flat patches
                    dim3D_t fpm_dim = {.depth=1, .height=(tile_L2 * tile_L2), .width=(kernel_d * (kernel_h * kernel_w))};
                    dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD};
                    dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile};
                    

                    get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
                    // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

                    for(int thl1=0; thl1<kernel_ch; thl1+=tile_L1){
                        // printf("THL1: %d\n", thl1);
                        int tile_l1_h_elems = (thl1 + tile_L1) >= kernel_ch ? (kernel_ch - thl1) : tile_L1;

                        // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                        // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                        // printf("N W elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                        // printf("N indexes in tile: %d\n", n_indexes_in_tile);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
                        // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                        int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);

                        // Loop over the output channels (the height of the tile)
                        for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
                            // printf("=================CH: %d\n", ch);
                            // Keeps track of how many indexes have been extracted
                            // Needed to stop at the right amount of indexes
                            int idxs_used = 0; 


                            // Loop over the words of indexes (the width of the tile)
                            // Extract the values from codebooks (one full channel of 3D kernel)
                            for(int i=0; i<tile_l1_w_elems; i++){
                                // printf("i: %d\n", i);
                                // printf("K index: %d --> %u\n", k_index, kernel[k_index]);
                                for(int j=0; j<IDXS_PER_WORD; j++){
                                    // printf("j: %d\n", j);
                                    shamt = j * BITS_PER_CB;
                                    index = ((kernel[k_index] >> shamt) & IDX_MASK);

                                    kernel_channel_values[idxs_used] = codebook[index];
                                    // printf("IDx used: %d\n", idxs_used);
                                    idxs_used++;
                                    if(idxs_used >= n_indexes_in_tile){
                                        break;
                                    }   
                                }
                                k_index++;
                            }


                            tile_row_cnt = n_cnt_rows_tile;
                            tile_col_cnt = n_cnt_cols_tile;
                            
                            for(int patch_cnt=0; patch_cnt<tile_l1_Tin_w_elems; patch_cnt++){
                                // printf("\nCh index: %d\n", (ch * out_h * out_w));

                                int out_index = (ch * out_h * out_w) + start_idx_out_tile + (tile_row_cnt * out_w) + tile_col_cnt;
                                // printf("\nMAt idx: %d\n", patch_cnt*n_indexes_in_tile);
                                // printf("OUT IDX: %d\n", out_index);
                                out->tensor[out_index] += vect_vect_mult(kernel_channel_values, &flat_patches_matrix_tiled_l1[patch_cnt*n_indexes_in_tile], n_indexes_in_tile);

                                tile_col_cnt++;
                                if(tile_col_cnt >= out_tile_w_elems){
                                    tile_col_cnt=0;
                                    tile_row_cnt++;
                                }
                            }
                        }
                    }

                    n_cnt_cols_tile = tile_col_cnt;
                    n_cnt_rows_tile = tile_row_cnt;
                }
                n_proc_idxs += n_indexes_in_tile;
            }
        }
    }
    

    #ifdef USE_BIAS

    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w] += bias[ch];
            }
        }
    }

    #endif



    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
    free(kernel_channel_values);
}







float vect_vect_mult(float *v0, float *v1, int size){
    float tmp=0;
    for(int i=0; i<size; i++){
        // printf("%f * %f\n", v0[i], v1[i]);
        tmp += v0[i] * v1[i];
    }

    return tmp;
}




tensor3D_t conv_layer(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *codebook){

    // Allocate an array of output tensors
    tensor3D_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float));

    tensor3D_t *padded_input = check_padding(in_tensor, conv.padding);
    free(in_tensor->tensor);
    
    // Conv 2
    conv3D_staticPatch_compact(conv, padded_input, kernel_perCH, codebook, &out_tensor);


    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}




tensor3D_t conv_layer_tiled_l2(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, uint32_t tile_L2_size, const float *codebook){

    // Allocate an array of output tensors
    tensor3D_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float));

    tensor3D_t *padded_input = check_padding(in_tensor, conv.padding);
    conv.input_dim = padded_input->dim;
    free(in_tensor->tensor);

    // Conv 2
    conv3D_staticPatch_compact_tiled(conv, padded_input, kernel_perCH, codebook, tile_L2_size, &out_tensor);

    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}






tensor3D_t conv_layer_tiled_l2l1_noCB(conv_t conv, tensor3D_t *in_tensor, const float *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size){

    // Allocate an array of output tensors
    tensor3D_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float));

    tensor3D_t *padded_input = check_padding(in_tensor, conv.padding);
    conv.input_dim = padded_input->dim;
    free(in_tensor->tensor);

    // Conv 2
    conv3D_staticPatch_compact_tiled_L1L2_noCB(conv, padded_input, kernel_perCH, bias, tile_L2_size, tile_L1_size, &out_tensor);

    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}







tensor3D_t conv_layer_tiled_l2l1(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float *codebook){

    // Allocate an array of output tensors
    tensor3D_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float));

    tensor3D_t *padded_input = check_padding(in_tensor, conv.padding);
    conv.input_dim = padded_input->dim;
    free(in_tensor->tensor);

    // Conv 2
    conv3D_staticPatch_compact_tiled_L1L2(conv, padded_input, kernel_perCH, codebook, bias, tile_L2_size, tile_L1_size, &out_tensor);

    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}



tensor3D_t conv_layer_tiled_l2l1_SVE(conv_t conv, tensor3D_t *in_tensor, const uint32_t *kernel_perCH, const float *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float *codebook){

    // Allocate an array of output tensors
    tensor3D_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float));

    tensor3D_t *padded_input = check_padding(in_tensor, conv.padding);
    conv.input_dim = padded_input->dim;
    free(in_tensor->tensor);

    // Conv 2
    conv3D_staticPatch_compact_tiled_L1L2_SVE(conv, padded_input, kernel_perCH, codebook, bias, tile_L2_size, tile_L1_size, &out_tensor);

    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}






tensor3D_f16_t conv_layer_tiled_l2l1_SVE_f16(conv_t conv, tensor3D_f16_t *in_tensor, const uint16_t *kernel_perCH, const float16_t *bias, uint32_t tile_L2_size, uint32_t tile_L1_size, const float16_t *codebook){

    // Allocate an array of output tensors
    tensor3D_f16_t out_tensor = {
        .dim = conv.output_dim,
        .tensor = (float16_t*)malloc((conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float16_t))
    };

    memset(out_tensor.tensor, 0.0, (conv.output_dim.depth * conv.output_dim.height * conv.output_dim.width) * sizeof(float16_t));

    tensor3D_f16_t *padded_input = check_padding_f16(in_tensor, conv.padding);
    conv.input_dim = padded_input->dim;
    free(in_tensor->tensor);

    // Conv 2
    conv3D_staticPatch_compact_tiled_L1L2_SVE_f16(conv, padded_input, kernel_perCH, codebook, bias, tile_L2_size, tile_L1_size, &out_tensor);

    // Deallocate the tensor
    free(padded_input->tensor);
    free(padded_input);

    return out_tensor;
}








tensor3D_t conv_layer_interl(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias,  uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float *out_interl = (float*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));
    tensor3D_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));

    tensor3D_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // Conv 2 //
    conv3D_staticPatch_compactSVE_interleavedND(conv, &input_interl_padded, kernel_perCH, codebook_interl, bias, interl_factor, &output_tensors_interl_4D);

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}



tensor3D_t conv_layer_interl_tiled_l2l1(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_ND = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float *out_interl = (float*)malloc((output_dims_interl_ND.depth * output_dims_interl_ND.height * output_dims_interl_ND.width) * sizeof(float));
    tensor3D_t output_tensors_interl_ND = {
        .dim = output_dims_interl_ND,
        .tensor = out_interl
    };

    memset(output_tensors_interl_ND.tensor, 0.0, (output_dims_interl_ND.depth * output_dims_interl_ND.height * output_dims_interl_ND.width) * sizeof(float));

    tensor3D_t input_interl_padded;

    check_padding_interleavedND(in_interl, conv.padding, interl_factor, &input_interl_padded);

    free(in_interl->tensor);

    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_ND);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2(conv, &input_interl_padded, kernel_perCH, codebook_interl, bias, interl_factor, tile_l2, tile_l1, &output_tensors_interl_ND);


    free(input_interl_padded.tensor);

    return output_tensors_interl_ND;

}







tensor3D_t conv_layer_interl_tiled_l2l1_no_lanes_loop(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float *out_interl = (float*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));
    tensor3D_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));

    tensor3D_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // system("m5 resetstats");
    // Conv 2 //
    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_4D);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_no_lanes_loop(conv, &input_interl_padded, kernel_perCH, codebook_interl, bias, interl_factor, tile_l2, tile_l1, &output_tensors_interl_4D);
    // system("m5 dumpresetstats");

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}









tensor3D_t conv_layer_interl_tiled_l2l1_diff_seq(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH_interl, const float *codebook_interl, const float *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float *out_interl = (float*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));
    tensor3D_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));

    tensor3D_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // system("m5 resetstats");
    // Conv 2 //
    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_4D);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq(conv, &input_interl_padded, kernel_perCH_interl, codebook_interl, bias, interl_factor, tile_l2, tile_l1, &output_tensors_interl_4D);
    // system("m5 dumpresetstats");

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}









tensor3D_f16_t conv_layer_interl_tiled_l2l1_diff_seq_f16(conv_t conv, tensor3D_f16_t *in_interl, const uint16_t *kernel_perCH_interl, const float16_t *codebook_interl, const float16_t *bias, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float16_t *out_interl = (float16_t*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float16_t));
    tensor3D_f16_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float16_t));

    tensor3D_f16_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND_f16(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // system("m5 resetstats");
    // Conv 2 //
    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_4D);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq_f16(conv, &input_interl_padded, kernel_perCH_interl, codebook_interl, bias, interl_factor, tile_l2, tile_l1, &output_tensors_interl_4D);
    // system("m5 dumpresetstats");

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}










tensor3D_t conv_layer_interl_tiled_l2l1_mem(conv_t conv, tensor3D_t *in_interl, const uint32_t *kernel_perCH, const float *codebook_interl, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float *out_interl = (float*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));
    tensor3D_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float));

    tensor3D_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // system("m5 resetstats");
    // Conv 2 //
    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_4D);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_mem(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_l2, tile_l1, &output_tensors_interl_4D);
    // system("m5 dumpresetstats");

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}




tensor3D_f16_t conv_layer_interl_tiled_f16(conv_t conv, tensor3D_f16_t *in_interl, const uint32_t *kernel_perCH, const float16_t *codebook_interl, const float16_t *bias_interl, uint32_t tile_l2, uint32_t tile_l1, uint8_t interl_factor){

    dim3D_t output_dims_interl_4D = {
        .depth = conv.output_dim.depth,
        .height = conv.output_dim.height,
        .width = conv.output_dim.width * interl_factor,
    };

    float16_t *out_interl = (float16_t*)malloc((output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float16_t));
    tensor3D_f16_t output_tensors_interl_4D = {
        .dim = output_dims_interl_4D,
        .tensor = out_interl
    };

    memset(output_tensors_interl_4D.tensor, 0.0, (output_dims_interl_4D.depth * output_dims_interl_4D.height * output_dims_interl_4D.width) * sizeof(float16_t));

    // printf("output_tensors_interl_4D memset done!\n");

    tensor3D_f16_t input_interl_padded;
    
    // print_interleaved_out(in_interl, 2);

    check_padding_interleavedND_f16(in_interl, conv.padding, interl_factor, &input_interl_padded);
    free(in_interl->tensor);

    // printf("PAdded: %d %d %d\n", input_interl_padded.dim.depth, input_interl_padded.dim.height, input_interl_padded.dim.width);
    
    // print_interleaved_out(&input_interl_padded, interl_factor);

    // Conv 2 //
    // conv3D_staticPatch_compactSVE_interleavedND_tiled(conv, &input_interl_padded, kernel_perCH, codebook_interl, interl_factor, tile_out, &output_tensors_interl_4D);
    conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_f16(conv, &input_interl_padded, kernel_perCH, codebook_interl, bias_interl, interl_factor, tile_l2, tile_l1, &output_tensors_interl_4D);

    // printf("(Conv2) Size %d %d %d\n", output_tensors_interl_4D.dim.depth, output_tensors_interl_4D.dim.height, output_tensors_interl_4D.dim.width);
    // print_interleaved_out(&output_tensors_interl_4D, 4);
    // exit(0);

    free(input_interl_padded.tensor);

    return output_tensors_interl_4D;

}




