
#include <SVE_implementations.h>



/**
 * Duplcates the input `n_reps` times and interleaves it in the output array
 */
void duplicate_interleave(float *in, uint32_t len, uint8_t n_reps, float *out){

    for(uint32_t i=0; i<len; i++){
        for(uint8_t j=0; j<n_reps; j++){
            out[(i*n_reps)+j] = in[i];
        }
    }
}




// // Defines the multiply factor of the interleaved 4D
// #define INTERL_4D_FACTOR    4

void sve_vect_mul_compact_interleaved2D(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t mat_cols, const float *codebook_interl, tensor3D_t *res, uint32_t out_index);
void sve_vect_mul_compact_interleaved4D(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t mat_cols, const float *codebook_interl, tensor3D_t *res, uint32_t out_index);
void sve_vect_mul_compact_interleaved4D_non_tiled_out(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_h, uint32_t tile_w, int full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
void sve_vect_mul_compact_non_tiled_out_l1l2(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook, tensor3D_t *res, uint32_t out_index, int base_res_idx);

void sve_vect_mul_compact_interleaved2D_non_tiled_out_l1l2(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_no_lanes_loop(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq(const uint32_t *vect_idxs_interl, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_mem(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx);
// void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(const uint16_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx);








void conv3D_staticPatch_compact_tiled_L1L2_SVE(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook, const float *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_t *out){

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


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // Matrix holding all the flattened patches extracted statically before the computations (only from a single input tile)
    float *flat_patches_matrix = (float*)malloc((kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)) * sizeof(float));
    float *flat_patches_matrix_tiled_l1 = (float*)malloc((tile_L1 * tile_L1 * IDXS_PER_WORD) * sizeof(float));   // Rearranged for L1 tiling

    // printf("FPM: %d\n", (kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)));
    // printf("FMP L1: %d\n", (tile_L1 * tile_L1 * IDXS_PER_WORD));

    // printf("Kernel CH values: %d\n", tile_L1 * IDXS_PER_WORD);

    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    int out_idx_so_far = 0;


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

            int start_idx_out_tile = (((th / tile_in_step) * tile_L2) * out_w) + ((tw / tile_in_step) * tile_L2);

            int n_proc_idxs = 0;    // Number of processed indexes per each output channel

            for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_L1){
                // printf("TWL1: %d\n", twl1);

                out_tile_w_idx=0;
                out_tile_h_idx=0;

                int tile_l1_w_elems = (twl1 + tile_L1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_L1;
                int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel ? (conv_layer.n_elems_k_channel - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);
                
                for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*out_tile_w_elems); twl1_Tin+=tile_L1){
                    // printf("TWL1_TIN: %d\n", twl1_Tin);
                    int tile_l1_Tin_w_elems = (twl1_Tin + tile_L1) >= (out_tile_h_elems * out_tile_w_elems) ? ((out_tile_h_elems * out_tile_w_elems) - twl1_Tin) : tile_L1;
                    

                    out_tile_w_idx = (twl1_Tin % (out_tile_w_elems));
                    out_tile_h_idx = twl1_Tin / (out_tile_w_elems);

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

                        // Loop over the output channels (the height of the tile)
                        for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){

                            out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;

                            sve_vect_mul_compact_non_tiled_out_l1l2(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                                    CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                    n_indexes_in_tile, 
                                                    flat_patches_matrix_tiled_l1, 
                                                    1, 
                                                    tile_l1_Tin_w_elems, 
                                                    out_tile_w_elems,
                                                    out_w, 
                                                    codebook, 
                                                    out, 
                                                    (ch*out_h*out_w), 
                                                    out_idx);
                        }
                    }
                }
                n_proc_idxs += n_indexes_in_tile;
            }
            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);
            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }


    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w] += bias[ch];
            }
        }
    }
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}









void conv3D_staticPatch_compactSVE_interleavedND(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    // printf("In dim: %d %d %d\n", input->dim.depth, input->dim.height, input->dim.width);
    // printf("Out dim: %d %d %d\n", kernel_ch, out_h, out_w);

    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };

    // printf("Patch dim: %d %d %d\n", patch_dims.depth, patch_dims.height, patch_dims.width);
    
    uint32_t n_patches = out_h * (out_w / interl_factor);
    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * n_patches));
    
    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * n_patches) * sizeof(float));


    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract
    // Loop output rows
    for(int r=0; r<out_h; r++){

        // Loop output columns
        for(int c=0; c<(out_w / interl_factor); c++){

            dim3D_t patch_idx = {
                .depth=0,
                .height=(r*conv_layer.stride),
                .width=(c*conv_layer.stride) * interl_factor
            };

            // printf("\nPatch index: %d %d %d\n", patch_idx.depth, patch_idx.height, patch_idx.width);
            // printf("Patch dim: %d %d %d\n", patch_dims.depth, patch_dims.height, patch_dims.width);
            // printf("Index patch matrix: %d\n", patch_cnt * (conv_layer.n_elems_k_channel) * interl_factor);
            get_3Dpatch(input->tensor, 
                        &input->dim,
                        &patch_idx,
                        &patch_dims,
                        conv_layer.n_elems_in_channel * interl_factor,
                        &flat_patches_matrix[patch_cnt * (conv_layer.n_elems_k_channel) * interl_factor]);

            patch_cnt++;

            // for(int i=0; i<(patch_dims.depth * patch_dims.height * patch_dims.width); i++){
            //     printf("[%d] %f\n", i, flat_patches_matrix[(patch_cnt * (conv_layer.n_elems_k_channel) * INTERL_4D_FACTOR) + i]);
            // }

            // print_volume(&flat_patches_matrix[patch_cnt * (conv_layer.n_elems_k_channel)], &patch_dims);
        }
    }

    // for(int i=0; i<(patch_dims.depth * (patch_dims.height * patch_dims.width) * n_patches); i++){
    //     printf("[%d] %f\n", i, flat_patches_matrix[i]);
    // }

    // exit(0);

    if(interl_factor == 2){
        // printf("INTERL 2D\n");
        for(int ch=0; ch<kernel_ch; ch++){
            // printf("\nOUT CH: %d  |  OUT IDX: %d\n\n", ch, (ch*out_h*out_w));
            sve_vect_mul_compact_interleaved2D(&kernel[(ch * conv_layer.n_idxs_word_channel)], 
                                                conv_layer.n_idxs_word_channel, 
                                                conv_layer.n_elems_k_channel, 
                                                flat_patches_matrix, 
                                                n_patches, 
                                                codebook_interl, 
                                                out, 
                                                (ch*out_h*out_w));
        
        }

    } else if(interl_factor == 4){
        // printf("INTERL 4D\n");
        for(int ch=0; ch<kernel_ch; ch++){
            // printf("\nOUT CH: %d  |  OUT IDX: %d\n\n", ch, (ch*out_h*out_w));
            sve_vect_mul_compact_interleaved4D(&kernel[(ch * conv_layer.n_idxs_word_channel)], 
                                                conv_layer.n_idxs_word_channel, 
                                                conv_layer.n_elems_k_channel, 
                                                flat_patches_matrix, 
                                                n_patches, 
                                                codebook_interl, 
                                                out, 
                                                (ch*out_h*out_w));
        
        }

    }


    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias[(ch*interl_factor) + 1];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias[(ch*interl_factor) + 2];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias[(ch*interl_factor) + 3];
            }
        }
    }
    
    free(flat_patches_matrix);
}





void conv3D_staticPatch_compactSVE_interleavedND_tiled(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, uint8_t interl_factor, uint8_t tile_out, tensor3D_t *out){

    // printf("IN: %d %d %d\n", input->dim.depth, input->dim.height, input->dim.width);

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in_h = tile_out * conv_layer.stride + in_tiles_overlap;
    int tile_in_step_h = tile_in_h - in_tiles_overlap;

    int tile_in_w = tile_in_h * interl_factor;
    int tile_in_step_w = tile_in_step_h * interl_factor;


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // printf("\n=======================\n");
    // printf("Out tile: %d\n", tile_out);
    // printf("Overlap: %d\n", in_tiles_overlap);
    // printf("In tile H: %d\n", tile_in_h);
    // printf("Tile step H: %d\n", tile_in_step_h);
    // printf("In tile W: %d\n", tile_in_w);
    // printf("Tile step W: %d\n", tile_in_step_w);
    // printf("=======================\n");


    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };


    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)) * sizeof(float));
    
    // printf("AFTER\n");
    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
    // int out_idx_so_far = 0;


    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<input->dim.height; th+=tile_in_step_h){

        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
            // system("m5 resetstats");
            // printf("\nTILES: %d %d\n", th, tw);
            next_free_idx = 0;

            // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
            uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("H elems: %d\n", tile_h_elems);
            // printf("W elems: %d\n", tile_w_elems);
            // printf("H elems OUT: %d\n", out_tile_h_elems);
            // printf("W elems OUT: %d\n", out_tile_w_elems);
            
            int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride) * interl_factor
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &patch_dims,
                                conv_layer.n_elems_in_channel * interl_factor,
                                &flat_patches_matrix[next_free_idx]);
                    
                    
                    next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
                    patch_cnt++;
                }
            }

            // system("m5 dumpresetstats");

            if(interl_factor == 2){
                printf("ERROR: tiled interl 2D not implemented yet!\n");
            } else if(interl_factor == 4){

                for(int ch=0; ch<kernel_ch; ch++){
                    // printf("CH: %d  --> ", ch);
                    // system("m5 resetstats");
                    ///////////////////////////////////////
                    //// OUTPUT STORED IN TILED MANNER ////
                    ///////////////////////////////////////
                    // out_idx_tiled = out_idx_so_far + (ch * out_h * out_w);
                    // printf("Out %d\n", out_idx_tiled);
    
                    // printf("Idx: %d\n", out_idx);
                    // sve_vect_mul_compact_interleaved4D( &kernel[(ch * conv_layer.n_idxs_word_channel)], 
                    //                                     conv_layer.n_idxs_word_channel, 
                    //                                     conv_layer.n_elems_k_channel, 
                    //                                     flat_patches_matrix, 
                    //                                     (out_tile_h_elems * (out_tile_w_elems / interl_factor)), 
                    //                                     codebook_interl, 
                    //                                     out, 
                    //                                     out_idx_tiled);
                    // out_idx_tiled += (out_tile_h_elems * out_tile_w_elems);

                    ///////////////////////////////////////////
                    //// OUTPUT STORED IN NON-TILED MANNER ////
                    ///////////////////////////////////////////
                    out_idx = (ch * out_h * out_w) + (out_tile_h_idx * out_w) + out_tile_w_idx;
                    // printf("Out idx: %d\n", out_idx);

                    // printf("\nOUT CH: %d  |  OUT IDX: %d\n\n", ch, (ch*out_h*out_w));
                    sve_vect_mul_compact_interleaved4D_non_tiled_out(&kernel[(ch * conv_layer.n_idxs_word_channel)], 
                                                        conv_layer.n_idxs_word_channel, 
                                                        conv_layer.n_elems_k_channel, 
                                                        flat_patches_matrix, 
                                                        out_tile_h_elems, 
                                                        out_tile_w_elems / interl_factor, 
                                                        out_w, 
                                                        codebook_interl, 
                                                        out, 
                                                        (ch*out_h*out_w), 
                                                        out_idx);
                    
                }
            }
            // system("m5 dumpresetstats");

            // out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            out_tile_w_idx += out_tile_w_elems;
            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }

    
    free(flat_patches_matrix);
}





void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in_h = tile_l2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step_h = tile_in_h - in_tiles_overlap;

    int tile_in_w = tile_in_h * interl_factor;
    int tile_in_step_w = tile_in_step_h * interl_factor;


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // printf("\n=======================\n");
    // printf("Out tile: %d\n", tile_l2);
    // printf("Overlap: %d\n", in_tiles_overlap);
    // printf("In tile H: %d\n", tile_in_h);
    // printf("Tile step H: %d\n", tile_in_step_h);
    // printf("In tile W: %d\n", tile_in_w);
    // printf("Tile step W: %d\n", tile_in_step_w);
    // printf("=======================\n");


    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };


    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float));
    float *flat_patches_matrix_tiled_l1 = (float*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float));
    
    // printf("FPM TL2: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)));
    // printf("FPM TL1: %d\n", tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor);

    // printf("AFTER\n");
    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
    int out_idx_so_far = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<input->dim.height; th+=tile_in_step_h){

        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);  // Here Padding is set to 0 since it's the output of a tile

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
            // system("m5 resetstats");
            next_free_idx = 0;

            // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
            uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\nTILES: %d %d\n", th, tw);
            // printf("H elems: %d\n", tile_h_elems);
            // printf("W elems: %d\n", tile_w_elems);
            // printf("H elems OUT: %d\n", out_tile_h_elems);
            // printf("W elems OUT: %d\n", out_tile_w_elems);
            
            int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride) * interl_factor
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &patch_dims,
                                conv_layer.n_elems_in_channel * interl_factor,
                                &flat_patches_matrix[next_free_idx]);
                    
                    // printf("Next idx: %d --> %d\n", next_free_idx, (patch_dims.depth * patch_dims.height * patch_dims.width));

                    // printf("===\n");
                    // for(int k=0; k<(kernel_d * kernel_h * kernel_w * interl_factor); k++){
                    //     printf("%f, ", flat_patches_matrix[next_free_idx + k]);
                    // }
                    // printf("\n");
                    
                    next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
                    patch_cnt++;
                }
            }


            int n_proc_idxs = 0;    // Number of processed indexes per each output channel

            int start_idx_out_tile = (((th / tile_in_step_h) * tile_l2) * out_w) + ((tw / tile_in_step_w) * tile_l2 * interl_factor);

            for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_l1){

                out_tile_w_idx=0;
                out_tile_h_idx=0;

                int tile_l1_w_elems = (twl1 + tile_l1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_l1;
                int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel  ? (conv_layer.n_elems_k_channel  - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);

                
                for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(tile_l1)){
                    
                    int tile_l1_Tin_w_elems = (twl1_Tin + tile_l1) >= (out_tile_h_elems * out_tile_w_elems / interl_factor) ? ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin) : tile_l1;

                    out_tile_w_idx = (twl1_Tin % (out_tile_w_elems/interl_factor)) * interl_factor;
                    out_tile_h_idx = twl1_Tin / (out_tile_w_elems/interl_factor);

                    // Define dimensions and extract the L1 tile from the matrix of flat patches
                    dim3D_t fpm_dim = {.depth=1, .height=(tile_l2 * tile_l2), .width=(kernel_d * (kernel_h * kernel_w)*interl_factor)};
                    dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD*interl_factor};
                    dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile*interl_factor};
                    

                    get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);

                    for(int thl1=0; thl1<kernel_ch; thl1+=tile_l1){

                        int tile_l1_h_elems = (thl1 + tile_l1) >= kernel_ch ? (kernel_ch - thl1) : tile_l1;

                        // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                        // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        // printf("N elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                        // printf("N indexes in tile: %d\n", n_indexes_in_tile);

                        
                        // Loop over the output channels (the height of the tile)
                        for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){ // smaller L1 tile, filter + weights
                            // printf("\n---CH: %d================================================================\n", ch);

                            // out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                            out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                            // printf("Out idx: %d\n", out_idx);
                            // printf("Out Idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d\n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                            // printf("BAse res idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d \n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                            // printf("Kernel idx: (%d * %d) + (%d * %d) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, k_index);
                            // printf("Size: %d %d\n", CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), n_indexes_in_tile);
                            // printf("Cols: %d %d\n", 1, tile_l1_Tin_w_elems);

                            # if N_LEARNERS == 2
                            sve_vect_mul_compact_interleaved2D_non_tiled_out_l1l2(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                                    CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                    n_indexes_in_tile, 
                                                    flat_patches_matrix_tiled_l1, 
                                                    1, 
                                                    tile_l1_Tin_w_elems, 
                                                    out_tile_w_elems,
                                                    out_w, 
                                                    codebook_interl, 
                                                    out, 
                                                    (ch*out_h*out_w), 
                                                    out_idx);

                            #elif (N_LEARNERS == 4) || (N_LEARNERS == 8)
                            sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                                    CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                    n_indexes_in_tile, 
                                                    flat_patches_matrix_tiled_l1, 
                                                    1, 
                                                    tile_l1_Tin_w_elems, 
                                                    out_tile_w_elems,
                                                    out_w, 
                                                    codebook_interl, 
                                                    out, 
                                                    (ch*out_h*out_w), 
                                                    out_idx);
                            #endif
                        }
                    }
                }
                n_proc_idxs += n_indexes_in_tile;
            }
            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }

    #ifdef USE_BIAS

    // Add the bias per ouput channel
    #if N_LEARNERS == 2

    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias[(ch*interl_factor) + 1];
            }
        }
    }

    #elif (N_LEARNERS == 4) || (N_LEARNERS == 8)

    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias[(ch*interl_factor) + 1];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias[(ch*interl_factor) + 2];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias[(ch*interl_factor) + 3];
            }
        }
    }
    #endif

    #endif

    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}

















void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_no_lanes_loop(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in_h = tile_l2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step_h = tile_in_h - in_tiles_overlap;

    int tile_in_w = tile_in_h * interl_factor;
    int tile_in_step_w = tile_in_step_h * interl_factor;


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // printf("\n=======================\n");
    // printf("Out tile: %d\n", tile_l2);
    // printf("Overlap: %d\n", in_tiles_overlap);
    // printf("In tile H: %d\n", tile_in_h);
    // printf("Tile step H: %d\n", tile_in_step_h);
    // printf("In tile W: %d\n", tile_in_w);
    // printf("Tile step W: %d\n", tile_in_step_w);
    // printf("=======================\n");


    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };


    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float));
    float *flat_patches_matrix_tiled_l1 = (float*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float));
    
    // printf("FPM TL2: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)));
    // printf("FPM TL1: %d\n", tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor);

    // printf("AFTER\n");
    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
    int out_idx_so_far = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<input->dim.height; th+=tile_in_step_h){

        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);  // Here Padding is set to 0 since it's the output of a tile

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
            // system("m5 resetstats");
            next_free_idx = 0;

            // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
            uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\nTILES: %d %d\n", th, tw);
            // printf("H elems: %d\n", tile_h_elems);
            // printf("W elems: %d\n", tile_w_elems);
            // printf("H elems OUT: %d\n", out_tile_h_elems);
            // printf("W elems OUT: %d\n", out_tile_w_elems);
            
            int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride) * interl_factor
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &patch_dims,
                                conv_layer.n_elems_in_channel * interl_factor,
                                &flat_patches_matrix[next_free_idx]);
                    
                    // printf("Next idx: %d --> %d\n", next_free_idx, (patch_dims.depth * patch_dims.height * patch_dims.width));

                    // printf("===\n");
                    // for(int k=0; k<(kernel_d * kernel_h * kernel_w * interl_factor); k++){
                    //     printf("%f, ", flat_patches_matrix[next_free_idx + k]);
                    // }
                    // printf("\n");
                    
                    next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
                    patch_cnt++;
                }
            }

            // system("m5 dumpresetstats");

            if(interl_factor == 2){
                printf("ERROR: tiled interl 2D not implemented yet!\n");
            } else if(interl_factor == 4){

                // printf("OUt TILE: %d %d\n", out_tile_h_elems, out_tile_w_elems);

                // uint8_t shamt = 0;
                // uint8_t index = 0;

                // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
                // int n_cnt_rows_tile = 0;
                // int n_cnt_cols_tile = 0;
                // int tile_row_cnt = 0;
                // int tile_col_cnt = 0;

                int n_proc_idxs = 0;    // Number of processed indexes per each output channel

                // int start_idx_out_tile = ((th / tile_in_step_h) * out_w) + (tw / tile_in_step_w);
                // printf("START IDX: (%d * %d) + %d = %d\n", th, out_w, tw, start_idx_out_tile);
                // int start_idx_out_tile = (th * out_w) + tw;
                int start_idx_out_tile = (((th / tile_in_step_h) * tile_l2) * out_w) + ((tw / tile_in_step_w) * tile_l2 * interl_factor);

                for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_l1){

                    out_tile_w_idx=0;
                    out_tile_h_idx=0;

                    // n_cnt_rows_tile = 0;
                    // n_cnt_cols_tile = 0;

                    int tile_l1_w_elems = (twl1 + tile_l1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_l1;
                    int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel  ? (conv_layer.n_elems_k_channel  - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);

                    // for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(TILE_L1)){
                    for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(tile_l1)){
                        // printf("TWL1TIN: %d\n", twl1_Tin);
                        int tile_l1_Tin_w_elems = (twl1_Tin + tile_l1) >= (out_tile_h_elems * out_tile_w_elems / interl_factor) ? ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin) : tile_l1;
                        // printf("(%d + %d) >= %d ? %d : %d --> %d\n", twl1_Tin, TILE_L1, (out_tile_h_elems * out_tile_w_elems / interl_factor), ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin), TILE_L1, tile_l1_Tin_w_elems);

                        out_tile_w_idx = (twl1_Tin % (out_tile_w_elems/interl_factor)) * interl_factor;
                        out_tile_h_idx = twl1_Tin / (out_tile_w_elems/interl_factor);

                        // Define dimensions and extract the L1 tile from the matrix of flat patches
                        dim3D_t fpm_dim = {.depth=1, .height=(tile_l2 * tile_l2), .width=(kernel_d * (kernel_h * kernel_w)*interl_factor)};
                        dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD*interl_factor};
                        dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile*interl_factor};
                        
                        // printf("FPM dim: %d %d %d\n", fpm_dim.depth, fpm_dim.height, fpm_dim.width);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
                        // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

                        // tensor3D_t dbg = {.tensor=flat_patches_matrix_tiled_l1, .dim=p_dim};
                        // print_interleaved_out(&dbg, 4);

                        for(int thl1=0; thl1<kernel_ch; thl1+=tile_l1){

                            int tile_l1_h_elems = (thl1 + tile_l1) >= kernel_ch ? (kernel_ch - thl1) : tile_l1;
                            // int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);


                            // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                            // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("N elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                            // printf("N indexes in tile: %d\n", n_indexes_in_tile);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
                            // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                            
                            // Loop over the output channels (the height of the tile)
                            for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
                                // printf("\n---CH: %d================================================================\n", ch);

                                // out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                // printf("Out idx: %d\n", out_idx);
                                // printf("Out Idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d\n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("BAse res idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d \n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("Kernel idx: (%d * %d) + (%d * %d) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, k_index);
                                // printf("Size: %d %d\n", CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), n_indexes_in_tile);
                                // printf("Cols: %d %d\n", 1, tile_l1_Tin_w_elems);
                                sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_no_lanes_loop(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                                        CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                        n_indexes_in_tile, 
                                                        flat_patches_matrix_tiled_l1, 
                                                        1, 
                                                        tile_l1_Tin_w_elems, 
                                                        out_tile_w_elems,
                                                        out_w, 
                                                        codebook_interl, 
                                                        out, 
                                                        (ch*out_h*out_w), 
                                                        out_idx);
                            }
                        }

                        // // out_tile_w_idx+=interl_factor;
                        // out_tile_w_idx += twl1_Tin * interl_factor;
                        // printf("OUT TILE W IDX: %d\n", out_tile_w_idx);
                        // if(out_tile_w_idx >= out_tile_w_elems){
                        //     out_tile_w_idx=0;
                        //     out_tile_h_idx++;
                        // }

                        // n_cnt_cols_tile = tile_col_cnt;
                        // n_cnt_rows_tile = tile_row_cnt;
                    }
                    n_proc_idxs += n_indexes_in_tile;
                }
            }
            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            // out_tile_w_idx += out_tile_w_elems;
            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }
    

    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias[(ch*interl_factor) + 1];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias[(ch*interl_factor) + 2];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias[(ch*interl_factor) + 3];
            }
        }
    }

    // system("m5 dumpresetstats");
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}

















void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel_interl, const float *codebook_interl, const float *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in_h = tile_l2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step_h = tile_in_h - in_tiles_overlap;

    int tile_in_w = tile_in_h * interl_factor;
    int tile_in_step_w = tile_in_step_h * interl_factor;


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // printf("\n=======================\n");
    // printf("Out tile: %d\n", tile_l2);
    // printf("Overlap: %d\n", in_tiles_overlap);
    // printf("In tile H: %d\n", tile_in_h);
    // printf("Tile step H: %d\n", tile_in_step_h);
    // printf("In tile W: %d\n", tile_in_w);
    // printf("Tile step W: %d\n", tile_in_step_w);
    // printf("=======================\n");


    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };


    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float));
    float *flat_patches_matrix_tiled_l1 = (float*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float));
    
    // printf("FPM TL2: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)));
    // printf("FPM TL1: %d\n", tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor);

    // printf("AFTER\n");
    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
    int out_idx_so_far = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<input->dim.height; th+=tile_in_step_h){

        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);  // Here Padding is set to 0 since it's the output of a tile

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
            // system("m5 resetstats");
            next_free_idx = 0;

            // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
            uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\nTILES: %d %d\n", th, tw);
            // printf("H elems: %d\n", tile_h_elems);
            // printf("W elems: %d\n", tile_w_elems);
            // printf("H elems OUT: %d\n", out_tile_h_elems);
            // printf("W elems OUT: %d\n", out_tile_w_elems);
            
            int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride) * interl_factor
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &patch_dims,
                                conv_layer.n_elems_in_channel * interl_factor,
                                &flat_patches_matrix[next_free_idx]);
                    
                    // printf("Next idx: %d --> %d\n", next_free_idx, (patch_dims.depth * patch_dims.height * patch_dims.width));

                    // printf("===\n");
                    // for(int k=0; k<(kernel_d * kernel_h * kernel_w * interl_factor); k++){
                    //     printf("%f, ", flat_patches_matrix[next_free_idx + k]);
                    // }
                    // printf("\n");
                    
                    next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
                    patch_cnt++;
                }
            }

            // system("m5 dumpresetstats");

            if(interl_factor == 2){
                printf("ERROR: tiled interl 2D not implemented yet!\n");
            } else if(interl_factor == 4){

                // printf("OUt TILE: %d %d\n", out_tile_h_elems, out_tile_w_elems);

                // uint8_t shamt = 0;
                // uint8_t index = 0;

                // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
                // int n_cnt_rows_tile = 0;
                // int n_cnt_cols_tile = 0;
                // int tile_row_cnt = 0;
                // int tile_col_cnt = 0;

                int n_proc_idxs = 0;    // Number of processed indexes per each output channel

                // int start_idx_out_tile = ((th / tile_in_step_h) * out_w) + (tw / tile_in_step_w);
                // printf("START IDX: (%d * %d) + %d = %d\n", th, out_w, tw, start_idx_out_tile);
                // int start_idx_out_tile = (th * out_w) + tw;
                int start_idx_out_tile = (((th / tile_in_step_h) * tile_l2) * out_w) + ((tw / tile_in_step_w) * tile_l2 * interl_factor);

                for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_l1){

                    out_tile_w_idx=0;
                    out_tile_h_idx=0;

                    // n_cnt_rows_tile = 0;
                    // n_cnt_cols_tile = 0;

                    int tile_l1_w_elems = (twl1 + tile_l1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_l1;
                    int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel  ? (conv_layer.n_elems_k_channel  - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);

                    // for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(TILE_L1)){
                    for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(tile_l1)){
                        // printf("TWL1TIN: %d\n", twl1_Tin);
                        int tile_l1_Tin_w_elems = (twl1_Tin + tile_l1) >= (out_tile_h_elems * out_tile_w_elems / interl_factor) ? ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin) : tile_l1;
                        // printf("(%d + %d) >= %d ? %d : %d --> %d\n", twl1_Tin, TILE_L1, (out_tile_h_elems * out_tile_w_elems / interl_factor), ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin), TILE_L1, tile_l1_Tin_w_elems);

                        out_tile_w_idx = (twl1_Tin % (out_tile_w_elems/interl_factor)) * interl_factor;
                        out_tile_h_idx = twl1_Tin / (out_tile_w_elems/interl_factor);

                        // Define dimensions and extract the L1 tile from the matrix of flat patches
                        dim3D_t fpm_dim = {.depth=1, .height=(tile_l2 * tile_l2), .width=(kernel_d * (kernel_h * kernel_w)*interl_factor)};
                        dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD*interl_factor};
                        dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile*interl_factor};
                        
                        // printf("FPM dim: %d %d %d\n", fpm_dim.depth, fpm_dim.height, fpm_dim.width);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
                        // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

                        // tensor3D_t dbg = {.tensor=flat_patches_matrix_tiled_l1, .dim=p_dim};
                        // print_interleaved_out(&dbg, 4);

                        for(int thl1=0; thl1<kernel_ch; thl1+=tile_l1){

                            int tile_l1_h_elems = (thl1 + tile_l1) >= kernel_ch ? (kernel_ch - thl1) : tile_l1;
                            // int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);


                            // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                            // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("N elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                            // printf("N indexes in tile: %d\n", n_indexes_in_tile);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
                            // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                            
                            // Loop over the output channels (the height of the tile)
                            for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
                                // printf("\n---CH: %d================================================================\n", ch);

                                // out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                // printf("Out idx: %d\n", out_idx);
                                // printf("Out Idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d\n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("BAse res idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d \n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("Kernel idx: (%d * %d) + (%d * %d) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, k_index);
                                // printf("Size: %d %d\n", CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), n_indexes_in_tile);
                                // printf("Cols: %d %d\n", 1, tile_l1_Tin_w_elems);
                                // printf("Kernel idx: ((%d * %d) + (%d * %d)*4) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, (twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems));
                                sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq(&kernel_interl[((twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)) * 4], 
                                                        CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                        n_indexes_in_tile, 
                                                        flat_patches_matrix_tiled_l1, 
                                                        1, 
                                                        tile_l1_Tin_w_elems, 
                                                        out_tile_w_elems,
                                                        out_w, 
                                                        codebook_interl, 
                                                        out, 
                                                        (ch*out_h*out_w), 
                                                        out_idx);
                            }
                        }
                        // exit(0);

                        // // out_tile_w_idx+=interl_factor;
                        // out_tile_w_idx += twl1_Tin * interl_factor;
                        // printf("OUT TILE W IDX: %d\n", out_tile_w_idx);
                        // if(out_tile_w_idx >= out_tile_w_elems){
                        //     out_tile_w_idx=0;
                        //     out_tile_h_idx++;
                        // }

                        // n_cnt_cols_tile = tile_col_cnt;
                        // n_cnt_rows_tile = tile_row_cnt;
                    }
                    n_proc_idxs += n_indexes_in_tile;
                }
            }
            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            // out_tile_w_idx += out_tile_w_elems;
            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }


    #ifdef USE_BIAS
    // Add the bias per ouput channel
    for(int ch=0; ch<conv_layer.out_ch; ch++){
        for(int h=0; h<conv_layer.output_dim.height; h++){
            for(int w=0; w<conv_layer.output_dim.width; w++){
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias[(ch*interl_factor) + 1];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias[(ch*interl_factor) + 2];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias[(ch*interl_factor) + 3];
            }
        }
    }
    #endif

    // system("m5 dumpresetstats");
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}




















void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_mem(conv_t conv_layer, tensor3D_t *input, const uint32_t *kernel, const float *codebook_interl, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_t *out){

    uint32_t kernel_ch = conv_layer.out_ch;
    uint32_t kernel_d = conv_layer.kernel_dim.depth;
    uint32_t kernel_h = conv_layer.kernel_dim.height;
    uint32_t kernel_w = conv_layer.kernel_dim.width;

    // Compute output dimensions
    int out_h = out->dim.height;
    int out_w = out->dim.width;
    
    uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

    // Overlap needed between tiles to account for the stride
    int in_tiles_overlap = kernel_h - conv_layer.stride;

    // Dimensions of the tile of the input image (all in channels)
    int tile_in_h = tile_l2 * conv_layer.stride + in_tiles_overlap;
    int tile_in_step_h = tile_in_h - in_tiles_overlap;

    int tile_in_w = tile_in_h * interl_factor;
    int tile_in_step_w = tile_in_step_h * interl_factor;


    // Keep track of the output index for tiling
    int out_tile_w_idx = 0;
    int out_tile_h_idx = 0;

    // printf("\n=======================\n");
    // printf("Out tile: %d\n", tile_l2);
    // printf("Overlap: %d\n", in_tiles_overlap);
    // printf("In tile H: %d\n", tile_in_h);
    // printf("Tile step H: %d\n", tile_in_step_h);
    // printf("In tile W: %d\n", tile_in_w);
    // printf("Tile step W: %d\n", tile_in_step_w);
    // printf("=======================\n");


    dim3D_t patch_dims = {
        .depth = kernel_d,
        .height = kernel_h,
        .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
    };


    // printf("N patches: %d\n", n_patches);
    // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

    // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

    // Matrix holding all the flattened patches extracted statically before the computations
    float *flat_patches_matrix = (float*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float));
    float *flat_patches_matrix_tiled_l1 = (float*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float));
    
    // printf("FPM TL2: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)));
    // printf("FPM TL1: %d\n", tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor);

    // printf("AFTER\n");
    // To address the next writing spot for the matrix of tiled patches
    int next_free_idx = 0;

    int tile_cnt = 0;
    int out_idx = 0;        // Index of the output (stored normally, row-major)
    // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
    int out_idx_so_far = 0;

    // Extract the matrix of patches associated only to an input tile //

    for(int th=0; th<input->dim.height; th+=tile_in_step_h){

        out_tile_w_idx = 0;

        int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
        int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);  // Here Padding is set to 0 since it's the output of a tile

        if(out_tile_h_elems <= 0){
            continue;
        }

        for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
            // system("m5 resetstats");
            next_free_idx = 0;

            // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
            uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
            uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
            if(out_tile_w_elems <= 0){
                continue;
            }

            // printf("\nTILES: %d %d\n", th, tw);
            // printf("H elems: %d\n", tile_h_elems);
            // printf("W elems: %d\n", tile_w_elems);
            // printf("H elems OUT: %d\n", out_tile_h_elems);
            // printf("W elems OUT: %d\n", out_tile_w_elems);
            
            int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
            for(int r=0; r<out_tile_h_elems; r++){
                for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
                    dim3D_t patch_idx = {
                        .depth=0,                       // Always take all the input channels
                        .height=(r*conv_layer.stride),
                        .width=(c*conv_layer.stride) * interl_factor
                    };

                    get_3Dpatch(&input->tensor[in_tensor_idx], 
                                // &tile_dim,   // For tiled input
                                &input->dim,    // For non-tiled input
                                &patch_idx,
                                &patch_dims,
                                conv_layer.n_elems_in_channel * interl_factor,
                                &flat_patches_matrix[next_free_idx]);
                    
                    // printf("Next idx: %d --> %d\n", next_free_idx, (patch_dims.depth * patch_dims.height * patch_dims.width));

                    // printf("===\n");
                    // for(int k=0; k<(kernel_d * kernel_h * kernel_w * interl_factor); k++){
                    //     printf("%f, ", flat_patches_matrix[next_free_idx + k]);
                    // }
                    // printf("\n");
                    
                    next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
                    patch_cnt++;
                }
            }

            // system("m5 dumpresetstats");

            if(interl_factor == 2){
                printf("ERROR: tiled interl 2D not implemented yet!\n");
            } else if(interl_factor == 4){

                // printf("OUt TILE: %d %d\n", out_tile_h_elems, out_tile_w_elems);

                // uint8_t shamt = 0;
                // uint8_t index = 0;

                // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
                // int n_cnt_rows_tile = 0;
                // int n_cnt_cols_tile = 0;
                // int tile_row_cnt = 0;
                // int tile_col_cnt = 0;

                int n_proc_idxs = 0;    // Number of processed indexes per each output channel

                // int start_idx_out_tile = ((th / tile_in_step_h) * out_w) + (tw / tile_in_step_w);
                // printf("START IDX: (%d * %d) + %d = %d\n", th, out_w, tw, start_idx_out_tile);
                // int start_idx_out_tile = (th * out_w) + tw;
                int start_idx_out_tile = (((th / tile_in_step_h) * tile_l2) * out_w) + ((tw / tile_in_step_w) * tile_l2 * interl_factor);

                for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_l1){

                    out_tile_w_idx=0;
                    out_tile_h_idx=0;

                    // n_cnt_rows_tile = 0;
                    // n_cnt_cols_tile = 0;

                    int tile_l1_w_elems = (twl1 + tile_l1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_l1;
                    int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel  ? (conv_layer.n_elems_k_channel  - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);

                    // for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(TILE_L1)){
                    for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(tile_l1)){
                        // printf("TWL1TIN: %d\n", twl1_Tin);
                        int tile_l1_Tin_w_elems = (twl1_Tin + tile_l1) >= (out_tile_h_elems * out_tile_w_elems / interl_factor) ? ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin) : tile_l1;
                        // printf("(%d + %d) >= %d ? %d : %d --> %d\n", twl1_Tin, TILE_L1, (out_tile_h_elems * out_tile_w_elems / interl_factor), ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin), TILE_L1, tile_l1_Tin_w_elems);

                        out_tile_w_idx = (twl1_Tin % (out_tile_w_elems/interl_factor)) * interl_factor;
                        out_tile_h_idx = twl1_Tin / (out_tile_w_elems/interl_factor);

                        // Define dimensions and extract the L1 tile from the matrix of flat patches
                        dim3D_t fpm_dim = {.depth=1, .height=(tile_l2 * tile_l2), .width=(kernel_d * (kernel_h * kernel_w)*interl_factor)};
                        dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD*interl_factor};
                        dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile*interl_factor};
                        
                        // printf("FPM dim: %d %d %d\n", fpm_dim.depth, fpm_dim.height, fpm_dim.width);
                        // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                        // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                        get_3Dpatch(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
                        // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

                        // tensor3D_t dbg = {.tensor=flat_patches_matrix_tiled_l1, .dim=p_dim};
                        // print_interleaved_out(&dbg, 4);

                        for(int thl1=0; thl1<kernel_ch; thl1+=tile_l1){

                            int tile_l1_h_elems = (thl1 + tile_l1) >= kernel_ch ? (kernel_ch - thl1) : tile_l1;
                            // int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);


                            // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
                            // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("N elemes tile T in: %d\n", tile_l1_Tin_w_elems);
                            // printf("N indexes in tile: %d\n", n_indexes_in_tile);
                            // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
                            // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
                            // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
                            // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                            
                            // Loop over the output channels (the height of the tile)
                            for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
                                // printf("\n---CH: %d================================================================\n", ch);

                                // out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
                                // printf("Out idx: %d\n", out_idx);
                                // printf("Out Idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d\n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("BAse res idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d \n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
                                // printf("Kernel idx: (%d * %d) + (%d * %d) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, k_index);
                                // printf("Size: %d %d\n", CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), n_indexes_in_tile);
                                // printf("Cols: %d %d\n", 1, tile_l1_Tin_w_elems);
                                sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                                        CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
                                                        n_indexes_in_tile, 
                                                        flat_patches_matrix_tiled_l1, 
                                                        1, 
                                                        tile_l1_Tin_w_elems, 
                                                        out_tile_w_elems,
                                                        out_w, 
                                                        codebook_interl, 
                                                        out, 
                                                        (ch*out_h*out_w), 
                                                        out_idx);
                            }
                        }

                        // // out_tile_w_idx+=interl_factor;
                        // out_tile_w_idx += twl1_Tin * interl_factor;
                        // printf("OUT TILE W IDX: %d\n", out_tile_w_idx);
                        // if(out_tile_w_idx >= out_tile_w_elems){
                        //     out_tile_w_idx=0;
                        //     out_tile_h_idx++;
                        // }

                        // n_cnt_cols_tile = tile_col_cnt;
                        // n_cnt_rows_tile = tile_row_cnt;
                    }
                    n_proc_idxs += n_indexes_in_tile;
                }
            }
            out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

            // out_tile_w_idx += out_tile_w_elems;
            tile_cnt++;
        }
        out_tile_h_idx += out_tile_h_elems;
    }

    // system("m5 dumpresetstats");
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}














// void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint16_t *kernel, const float16_t *codebook_interl, const float16_t *bias_interl, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_f16_t *out){

//     uint32_t kernel_ch = conv_layer.out_ch;
//     uint32_t kernel_d = conv_layer.kernel_dim.depth;
//     uint32_t kernel_h = conv_layer.kernel_dim.height;
//     uint32_t kernel_w = conv_layer.kernel_dim.width;

//     // Compute output dimensions
//     int out_h = out->dim.height;
//     int out_w = out->dim.width;
    
//     uint32_t patch_cnt = 0; // Counter to the index of the patch to extract

//     // Overlap needed between tiles to account for the stride
//     int in_tiles_overlap = kernel_h - conv_layer.stride;

//     // Dimensions of the tile of the input image (all in channels)
//     int tile_in_h = tile_l2 * conv_layer.stride + in_tiles_overlap;
//     int tile_in_step_h = tile_in_h - in_tiles_overlap;

//     int tile_in_w = tile_in_h * interl_factor;
//     int tile_in_step_w = tile_in_step_h * interl_factor;


//     // Keep track of the output index for tiling
//     int out_tile_w_idx = 0;
//     int out_tile_h_idx = 0;

//     // printf("\n=======================\n");
//     // printf("Out tile: %d\n", tile_l2);
//     // printf("Overlap: %d\n", in_tiles_overlap);
//     // printf("In tile H: %d\n", tile_in_h);
//     // printf("Tile step H: %d\n", tile_in_step_h);
//     // printf("In tile W: %d\n", tile_in_w);
//     // printf("Tile step W: %d\n", tile_in_step_w);
//     // printf("=======================\n");


//     dim3D_t patch_dims = {
//         .depth = kernel_d,
//         .height = kernel_h,
//         .width = kernel_w * interl_factor   // 4 since I am using an interleaved 4D
//     };


//     // printf("N patches: %d\n", n_patches);
//     // printf("N patches = %d * (%d / %d) = %d\n", out_h, out_w, interl_factor, n_patches);

//     // printf("N elems patch matrix: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_out * tile_out)));

//     // Matrix holding all the flattened patches extracted statically before the computations
//     float16_t *flat_patches_matrix = (float16_t*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float16_t));
//     float16_t *flat_patches_matrix_tiled_l1 = (float16_t*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float16_t));
    
//     // printf("FPM TL2: %d\n", (patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)));
//     // printf("FPM TL1: %d\n", tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor);

//     // printf("AFTER\n");
//     // To address the next writing spot for the matrix of tiled patches
//     int next_free_idx = 0;

//     int tile_cnt = 0;
//     int out_idx = 0;        // Index of the output (stored normally, row-major)
//     // int out_idx_tiled = 0;  // Index of the output (stored in a tiled way)
//     int out_idx_so_far = 0;

//     // Extract the matrix of patches associated only to an input tile //



//     for(int th=0; th<input->dim.height; th+=tile_in_step_h){

//         out_tile_w_idx = 0;

//         int tile_h_elems = (th + tile_in_h) >= input->dim.height ? (input->dim.height - th) : tile_in_h;
//         int out_tile_h_elems = CONV_OUT_DIM(tile_h_elems, kernel_h, conv_layer.stride, 0);  // Here Padding is set to 0 since it's the output of a tile

//         if(out_tile_h_elems <= 0){
//             continue;
//         }

//         for(int tw=0; tw<input->dim.width; tw+=tile_in_step_w){
        
//             // system("m5 resetstats");
//             next_free_idx = 0;

//             // printf("%d >= %d? %d : %d\n", (tw + tile_in_w), input->dim.width, (input->dim.width - tw), tile_in_w);
//             uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
//             // uint32_t tile_w_elems = (tw + tile_in_w) >= input->dim.width ? (input->dim.width - tw) : tile_in_w;
//             uint32_t out_tile_w_elems = CONV_OUT_DIM((tile_w_elems / interl_factor), kernel_w, conv_layer.stride, 0) * interl_factor;
            
//             if(out_tile_w_elems <= 0){
//                 continue;
//             }


//             // printf("\nTILES: %d %d\n", th, tw);
//             // printf("H elems: %d\n", tile_h_elems);
//             // printf("W elems: %d\n", tile_w_elems);
//             // printf("H elems OUT: %d\n", out_tile_h_elems);
//             // printf("W elems OUT: %d\n", out_tile_w_elems);
            
//             int in_tensor_idx = (th * input->dim.width) + tw;  // This is for the non-tiled input
            
//             for(int r=0; r<out_tile_h_elems; r++){
//                 for(int c=0; c<(out_tile_w_elems / interl_factor); c++){
                    
//                     dim3D_t patch_idx = {
//                         .depth=0,                       // Always take all the input channels
//                         .height=(r*conv_layer.stride),
//                         .width=(c*conv_layer.stride) * interl_factor
//                     };

//                     get_3Dpatch_f16(&input->tensor[in_tensor_idx], 
//                                 // &tile_dim,   // For tiled input
//                                 &input->dim,    // For non-tiled input
//                                 &patch_idx,
//                                 &patch_dims,
//                                 conv_layer.n_elems_in_channel * interl_factor,
//                                 &flat_patches_matrix[next_free_idx]);
                    
//                     // printf("Next idx: %d --> %d\n", next_free_idx, (patch_dims.depth * patch_dims.height * patch_dims.width));

//                     // printf("===\n");
//                     // for(int k=0; k<(kernel_d * kernel_h * kernel_w * interl_factor); k++){
//                     //     printf("%f, ", flat_patches_matrix[next_free_idx + k]);
//                     // }
//                     // printf("\n");
                    
//                     next_free_idx += (patch_dims.depth * patch_dims.height * patch_dims.width);
//                     patch_cnt++;
//                 }
//             }

//             // system("m5 dumpresetstats");

//             if(interl_factor == 2){
//                 printf("ERROR: tiled interl 2D not implemented yet!\n");
//             } else if(interl_factor == 4){

//                 // printf("OUt TILE: %d %d\n", out_tile_h_elems, out_tile_w_elems);

//                 // uint8_t shamt = 0;
//                 // uint8_t index = 0;

//                 // These counters serve to index the correct row and column of the output L2 tile when processing the L1 tile
//                 // int n_cnt_rows_tile = 0;
//                 // int n_cnt_cols_tile = 0;
//                 // int tile_row_cnt = 0;
//                 // int tile_col_cnt = 0;

//                 int n_proc_idxs = 0;    // Number of processed indexes per each output channel

//                 // int start_idx_out_tile = ((th / tile_in_step_h) * out_w) + (tw / tile_in_step_w);
//                 // printf("START IDX: (%d * %d) + %d = %d\n", th, out_w, tw, start_idx_out_tile);
//                 // int start_idx_out_tile = (th * out_w) + tw;
//                 int start_idx_out_tile = (((th / tile_in_step_h) * tile_l2) * out_w) + ((tw / tile_in_step_w) * tile_l2 * interl_factor);

//                 for(int twl1=0; twl1<conv_layer.n_idxs_word_channel; twl1+=tile_l1){

//                     out_tile_w_idx=0;
//                     out_tile_h_idx=0;

//                     // n_cnt_rows_tile = 0;
//                     // n_cnt_cols_tile = 0;

//                     int tile_l1_w_elems = (twl1 + tile_l1) >= conv_layer.n_idxs_word_channel ? (conv_layer.n_idxs_word_channel - twl1) : tile_l1;
//                     int n_indexes_in_tile = (n_proc_idxs + (tile_l1_w_elems * IDXS_PER_WORD)) >= conv_layer.n_elems_k_channel  ? (conv_layer.n_elems_k_channel  - n_proc_idxs) : (tile_l1_w_elems * IDXS_PER_WORD);
                    
//                     // for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(TILE_L1)){
//                     for(int twl1_Tin=0; twl1_Tin<(out_tile_h_elems*(out_tile_w_elems / interl_factor)); twl1_Tin+=(tile_l1)){
//                         // printf("TWL1TIN: %d\n", twl1_Tin);
//                         int tile_l1_Tin_w_elems = (twl1_Tin + tile_l1) >= (out_tile_h_elems * out_tile_w_elems / interl_factor) ? ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin) : tile_l1;
//                         // printf("(%d + %d) >= %d ? %d : %d --> %d\n", twl1_Tin, TILE_L1, (out_tile_h_elems * out_tile_w_elems / interl_factor), ((out_tile_h_elems * out_tile_w_elems / interl_factor) - twl1_Tin), TILE_L1, tile_l1_Tin_w_elems);

//                         out_tile_w_idx = (twl1_Tin % (out_tile_w_elems/interl_factor)) * interl_factor;
//                         out_tile_h_idx = twl1_Tin / (out_tile_w_elems/interl_factor);

//                         // Define dimensions and extract the L1 tile from the matrix of flat patches
//                         dim3D_t fpm_dim = {.depth=1, .height=(tile_l2 * tile_l2), .width=(kernel_d * (kernel_h * kernel_w)*interl_factor)};
//                         dim3D_t p_idx = {.depth=0, .height=twl1_Tin, .width=twl1*IDXS_PER_WORD*interl_factor};
//                         dim3D_t p_dim = {.depth=1, .height=tile_l1_Tin_w_elems, .width=n_indexes_in_tile*interl_factor};
                        
//                         // printf("FPM dim: %d %d %d\n", fpm_dim.depth, fpm_dim.height, fpm_dim.width);
//                         // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
//                         // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
//                         get_3Dpatch_f16(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
//                         // print_volume(flat_patches_matrix_tiled_l1, &p_dim);

//                         // tensor3D_t dbg = {.tensor=flat_patches_matrix_tiled_l1, .dim=p_dim};
//                         // print_interleaved_out(&dbg, 4);

//                         for(int thl1=0; thl1<kernel_ch; thl1+=tile_l1){

//                             int tile_l1_h_elems = (thl1 + tile_l1) >= kernel_ch ? (kernel_ch - thl1) : tile_l1;
//                             // int k_index = (twl1 * conv_layer.out_ch) + (thl1 * tile_l1_w_elems);


//                             // printf("\nTILES L1: (kernel) (input) : (%d %d)  (%d %d)\n", thl1, twl1, twl1, twl1_Tin);
//                             // printf("N tile elements (kernel): [%d %d]\n", tile_l1_h_elems, tile_l1_w_elems);
//                             // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
//                             // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
//                             // printf("N elemes tile T in: %d\n", tile_l1_Tin_w_elems);
//                             // printf("N indexes in tile: %d\n", n_indexes_in_tile);
//                             // printf("P idx: %d %d %d\n", p_idx.depth, p_idx.height, p_idx.width);
//                             // printf("P dim: %d %d %d\n", p_dim.depth, p_dim.height, p_dim.width);
//                             // printf("n_cnt_rows_tile: %d\n", n_cnt_rows_tile);
//                             // printf("n_cnt_cols_tile: %d\n", n_cnt_cols_tile);

                            
//                             // Loop over the output channels (the height of the tile)
//                             for(int ch=thl1; ch<thl1+tile_l1_h_elems; ch++){
//                                 // printf("\n---CH: %d================================================================\n", ch);

//                                 // out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
//                                 out_idx = (ch * out_h * out_w) + start_idx_out_tile + (out_tile_h_idx * out_w) + out_tile_w_idx;
//                                 // printf("Out idx: %d\n", out_idx);
//                                 // printf("Out Idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d\n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
//                                 // printf("BAse res idx: (%d * %d * %d) + %d + (%d * %d) + %d = %d \n", ch, out_h, out_w, start_idx_out_tile, out_tile_h_idx, out_w, out_tile_w_idx, out_idx);
//                                 // printf("Kernel idx: (%d * %d) + (%d * %d) = %d\n", twl1, conv_layer.out_ch, ch, tile_l1_w_elems, k_index);
//                                 // printf("Size: %d %d\n", CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), n_indexes_in_tile);
//                                 // printf("Cols: %d %d\n", 1, tile_l1_Tin_w_elems);
//                                 sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
//                                                         CEIL_DIV(n_indexes_in_tile, IDXS_PER_WORD), 
//                                                         n_indexes_in_tile, 
//                                                         flat_patches_matrix_tiled_l1, 
//                                                         1, 
//                                                         tile_l1_Tin_w_elems, 
//                                                         out_tile_w_elems,
//                                                         out_w, 
//                                                         codebook_interl, 
//                                                         out, 
//                                                         (ch*out_h*out_w), 
//                                                         out_idx);
//                             }
//                         }

//                         // // out_tile_w_idx+=interl_factor;
//                         // out_tile_w_idx += twl1_Tin * interl_factor;
//                         // printf("OUT TILE W IDX: %d\n", out_tile_w_idx);
//                         // if(out_tile_w_idx >= out_tile_w_elems){
//                         //     out_tile_w_idx=0;
//                         //     out_tile_h_idx++;
//                         // }

//                         // n_cnt_cols_tile = tile_col_cnt;
//                         // n_cnt_rows_tile = tile_row_cnt;
//                     }
//                     n_proc_idxs += n_indexes_in_tile;
//                 }
//             }
//             out_idx_so_far += (out_tile_h_elems * out_tile_w_elems);

//             // out_tile_w_idx += out_tile_w_elems;
//             tile_cnt++;
//         }
//         out_tile_h_idx += out_tile_h_elems;
//     }
    

//     // Add the bias per ouput channel
//     for(int ch=0; ch<conv_layer.out_ch; ch++){
//         for(int h=0; h<conv_layer.output_dim.height; h++){
//             for(int w=0; w<conv_layer.output_dim.width; w++){


//                 out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias_interl[(ch*interl_factor) + 0];
//                 out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias_interl[(ch*interl_factor) + 1];
//                 out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias_interl[(ch*interl_factor) + 2];
//                 out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias_interl[(ch*interl_factor) + 3];


//                 // volatile float out_f0 = (float)out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0];
//                 // volatile float out_f1 = (float)out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1];
//                 // volatile float out_f2 = (float)out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2];
//                 // volatile float out_f3 = (float)out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3];

//                 // volatile float bias_f0 = bias_interl[(ch*interl_factor) + 0];
//                 // volatile float bias_f1 = bias_interl[(ch*interl_factor) + 1];
//                 // volatile float bias_f2 = bias_interl[(ch*interl_factor) + 2];
//                 // volatile float bias_f3 = bias_interl[(ch*interl_factor) + 3];

//                 // out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] = (float16_t)(out_f0 + bias_f0);
//                 // out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] = (float16_t)(out_f1 + bias_f1);
//                 // out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] = (float16_t)(out_f2 + bias_f2);
//                 // out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] = (float16_t)(out_f3 + bias_f3);

//             }
//         }
//     }



//     // system("m5 dumpresetstats");
    
//     free(flat_patches_matrix);
//     free(flat_patches_matrix_tiled_l1);
// }







void sve_vect_mul_compact_interleaved2D(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t mat_cols, const float *codebook_interl, tensor3D_t *res, uint32_t out_index){

    uint8_t interl_factor_2D = 2;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t masks;           // Holds the masks to unpack the indexes

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x2_t in_vals;        // Holds the input values


    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget2_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget2_f32(codebooks_loaded, 1);
    // print_vect_f32(cb0);
    // print_vect_f32(cb1);

    // Loop thorugh the columns of weights
    for(int r=0; r<mat_cols; r++){
        // printf("\n\n>> Column: %d\n", r);
        row_res_vect0 = svdup_n_f32(0.0f);
        row_res_vect1 = svdup_n_f32(0.0f);
    
        input_idx = 0;

        // Loop thorugh the elements of the weights indexes (compact)
        // It indexes the column words (the words that contains the indexes per each column)
        for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
            // printf("\n---- CW %d ----\n", cw);

            load_pg = svwhilelt_b32(cw, vect_size);

            // Load a 32-bits word with IDXS_PER_WORD packed indexes
            packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
            // print_vect_ui32(packed_idxs);

            // Counts how many lanes have been loaded
            n_loaded_lanes = svcntp_b32(load_pg, load_pg);
            // printf("N loaded lanes: %d\n", n_loaded_lanes);


            // Loop through the 32-bits lanes of the vector register
            for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                // printf("\n\n---- LANE %ld ----\n", lane);

                missing_lane = 0;

                // Duplicates a single word of packed indexes in all the lanes
                dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                // print_vect_ui32(dup_idxs_pakd);

                for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                    // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                    
                    missing_lane = (IDXS_PER_WORD - idx_ptr);
                    missing_total = (n_vect_elems - input_idx);
                    // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                    
                    // Stop in case there are no more missing indexes to process
                    if (missing_total<=0){
                        // printf("Exiting...\n");
                        break;
                    }

                    if(missing_lane <= missing_total){
                        bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                    }else{
                        bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                    }
                    // print_predicate_w(bits_mask_pg);

                    // Create the shift amounts to address the correct portion of indexes within the word
                    shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                    // print_vect_ui32(shamts);
                    masks = svlsl_u32_z(bits_mask_pg, svdup_u32(IDX_MASK), shamts);  // Shift amounts for the indexes
                    // print_vect_ui32(masks);

                    // Perform the MASK+SHIFT for the considered word of packed indexes  
                    unpkd_idxs = svand_u32_z(bits_mask_pg, masks, dup_idxs_pakd);
                    // print_vect_ui32(unpkd_idxs);
                    unpkd_idxs = svlsr_u32_z(bits_mask_pg, unpkd_idxs, shamts);
                    // print_vect_ui32(unpkd_idxs);

                    // Load the weights based on the unpacked indexes
                    // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                    svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                    svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                    // print_vect_f32(weights_0);
                    // print_vect_f32(weights_1);
                    // exit(0);

                    // Load the input values based on the unpacked indexes
                    // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                    // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                    // print_vect_f32(in_vals);
                    in_vals = svld2_f32(bits_mask_pg, &mat[((r*n_vect_elems) + input_idx) * interl_factor_2D]);
                    svfloat32_t in_0 = svget2_f32(in_vals, 0);
                    svfloat32_t in_1 = svget2_f32(in_vals, 1);
                    // print_vect_f32(in_0);
                    // print_vect_f32(in_1);
                    // print_vect_f32(in_2);
                    // print_vect_f32(in_3);
                    // exit(0);
                    // printf("-------\n\n");

                    input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                    // MAC
                    // printf("Before:\n");
                    // print_vect_f32(row_res_vect0);
                    // printf("In:\n");
                    // print_vect_f32(in_0);
                    // printf("Weights:\n");
                    // print_vect_f32(weights_0);
                    row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                }
            }
        }

        uint32_t base_index = out_index + (r * interl_factor_2D);


        res->tensor[base_index + 0] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
        res->tensor[base_index + 1] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);

    }
}







void sve_vect_mul_compact_interleaved4D(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t mat_cols, const float *codebook_interl, tensor3D_t *res, uint32_t out_index){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t masks;           // Holds the masks to unpack the indexes

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x4_t in_vals;        // Holds the input values


    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    // Loop thorugh the columns of weights
    for(int r=0; r<mat_cols; r++){
        // printf("\n\n>> Column: %d\n", r);
        row_res_vect0 = svdup_n_f32(0.0f);
        row_res_vect1 = svdup_n_f32(0.0f);
        row_res_vect2 = svdup_n_f32(0.0f);
        row_res_vect3 = svdup_n_f32(0.0f);
    
        input_idx = 0;

        // Loop thorugh the elements of the weights indexes (compact)
        // It indexes the column words (the words that contains the indexes per each column)
        for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
            // printf("\n---- CW %d ----\n", cw);

            load_pg = svwhilelt_b32(cw, vect_size);

            // Load a 32-bits word with IDXS_PER_WORD packed indexes
            packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
            // print_vect_ui32(packed_idxs);

            // Counts how many lanes have been loaded
            n_loaded_lanes = svcntp_b32(load_pg, load_pg);
            // printf("N loaded lanes: %d\n", n_loaded_lanes);


            // Loop through the 32-bits lanes of the vector register
            for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                // printf("\n\n---- LANE %ld ----\n", lane);

                missing_lane = 0;

                // Duplicates a single word of packed indexes in all the lanes
                dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                // print_vect_ui32(dup_idxs_pakd);

                for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                    // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                    
                    missing_lane = (IDXS_PER_WORD - idx_ptr);
                    missing_total = (n_vect_elems - input_idx);
                    // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                    
                    // Stop in case there are no more missing indexes to process
                    if (missing_total<=0){
                        // printf("Exiting...\n");
                        break;
                    }

                    if(missing_lane <= missing_total){
                        bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                    }else{
                        bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                    }
                    // print_predicate_w(bits_mask_pg);

                    // Create the shift amounts to address the correct portion of indexes within the word
                    shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                    // print_vect_ui32(shamts);
                    masks = svlsl_u32_z(bits_mask_pg, svdup_u32(IDX_MASK), shamts);  // Shift amounts for the indexes
                    // print_vect_ui32(masks);

                    // Perform the MASK+SHIFT for the considered word of packed indexes  
                    unpkd_idxs = svand_u32_z(bits_mask_pg, masks, dup_idxs_pakd);
                    // print_vect_ui32(unpkd_idxs);
                    unpkd_idxs = svlsr_u32_z(bits_mask_pg, unpkd_idxs, shamts);
                    // print_vect_ui32(unpkd_idxs);

                    // Load the weights based on the unpacked indexes
                    svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                    svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                    svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                    svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);

                    // Load the input values based on the unpacked indexes
                    in_vals = svld4_f32(bits_mask_pg, &mat[((r*n_vect_elems) + input_idx) * interl_factor_4D]);
                    svfloat32_t in_0 = svget4_f32(in_vals, 0);
                    svfloat32_t in_1 = svget4_f32(in_vals, 1);
                    svfloat32_t in_2 = svget4_f32(in_vals, 2);
                    svfloat32_t in_3 = svget4_f32(in_vals, 3);

                    input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                    // MAC
                    row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f32_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f32_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                }
            }
        }

        uint32_t base_index = out_index + (r * interl_factor_4D);


        res->tensor[base_index + 0] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
        res->tensor[base_index + 1] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
        res->tensor[base_index + 2] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
        res->tensor[base_index + 3] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
    }
}








void sve_vect_mul_compact_interleaved4D_non_tiled_out(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_h, uint32_t tile_w, int full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x4_t in_vals;        // Holds the input values


    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    // printf("tiles %d %d = %d\n", tile_h, tile_w, (tile_h * tile_w));

    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_h; rt++){
        for(int ct=0; ct<tile_w; ct++){
                
            // printf("\n\n>> Column: %d\n", r);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
            row_res_vect2 = svdup_n_f32(0.0f);
            row_res_vect3 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("\n---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("\n\n---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                    // print_vect_ui32(dup_idxs_pakd);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs);

                        // Load the weights based on the unpacked indexes
                        // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                        svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);
                        // print_vect_f32(weights_0);
                        // exit(0);

                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        in_vals = svld4_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat32_t in_0 = svget4_f32(in_vals, 0);
                        svfloat32_t in_1 = svget4_f32(in_vals, 1);
                        svfloat32_t in_2 = svget4_f32(in_vals, 2);
                        svfloat32_t in_3 = svget4_f32(in_vals, 3);
                        // print_vect_f32(in_0);
                        // print_vect_f32(in_1);
                        // print_vect_f32(in_2);
                        // print_vect_f32(in_3);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        // printf("Before:\n");
                        // print_vect_f32(row_res_vect0);
                        // printf("In:\n");
                        // print_vect_f32(in_0);
                        // printf("Weights:\n");
                        // print_vect_f32(weights_0);
                        row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        // printf("After:\n");
                        // print_vect_f32(row_res_vect0);
                        row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f32_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f32_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);
            uint32_t base_index = base_res_idx + (rt * full_out_w) + (ct * interl_factor_4D);
            // printf("BAse idx: %d + %d + ( %d * %d ) + %d = %d\n", out_index, base_res_idx, rt, full_out_w, ct, base_index);

            // print_vect_f32(row_res_vect0);
            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));

            res->tensor[base_index + 0] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            res->tensor[base_index + 2] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            res->tensor[base_index + 3] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);

            // res[0].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            // res[1].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            // res[2].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            // res[3].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            // break;
            col_cnt++;
        }
    }


    // printf("--> %f\n", res->tensor[out_index + 0 + 0]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 1]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 2]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 3]);



    // exit(0);

}









void sve_vect_mul_compact_non_tiled_out_l1l2(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32_t in_vals;        // Holds the input values


    svfloat32_t weights = svdup_n_f32(0.0);

    #if defined(N_SVE_REG_CB_1)
    svfloat32_t cb = svld1_f32(svwhilelt_b32(0, CB_SIZE), codebook);
    #elif defined(N_SVE_REG_CB_2)
    svfloat32_t cb0 = svld1_f32(svwhilelt_b32(0, CB_SIZE), codebook);
    svfloat32_t cb1 = svld1_f32(svwhilelt_b32(0, CB_SIZE), &codebook[4]);
    svfloat32x2_t cb = svcreate2_f32(cb0, cb1);
    #elif defined(N_SVE_REG_CB_4)
    svfloat32_t cb0 = svld1_f32(svwhilelt_b32(0, CB_SIZE), codebook);
    svfloat32_t cb1 = svld1_f32(svwhilelt_b32(0, CB_SIZE), &codebook[4]);
    svfloat32_t cb2 = svld1_f32(svwhilelt_b32(0, CB_SIZE), &codebook[8]);
    svfloat32_t cb3 = svld1_f32(svwhilelt_b32(0, CB_SIZE), &codebook[12]);
    svfloat32x4_t cb = svcreate4_f32(cb0, cb1, cb2, cb3);
    #endif

    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect = svdup_n_f32(0.0f);
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                    // print_vect_ui32(dup_idxs_pakd);
                    // exit(0);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd, shamts); // unpkd_idxs[i] = dup_idxs_pakd[i] >> shamts[i];
                        unpkd_idxs = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs); // unpkd_idxs[i] = base_masks[i] & unpkd_idxs[i];

                        // unpkd_idxs are the real indexes to access the codebook and the input values
                        // Load the weights based on the unpacked indexes
                        #ifdef N_SVE_REG_CB_1
                        weights = svtbl_f32(cb, unpkd_idxs); // out[i] = cb[ unpkd_idxs[i] ]; // table lookup / gather-from-vector
                        #elif defined(N_SVE_REG_CB_2)
                        weights = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb);
                        #elif defined(N_SVE_REG_CB_4)
                        weights = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb);
                        #endif
                        
                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
                        //                                                 input_idx, 
                        //                                                 interl_factor_4D, 
                        //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
                        in_vals = svld1_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx)]); // input values
                        // print_vect_f32(in_0);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect = svmad_f32_x(bits_mask_pg, in_vals, weights, row_res_vect);
                    }
                }
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w)));   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect0);
            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect);
            
            // res[0].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            // res[1].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            // res[2].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            // res[3].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            // break;
            col_cnt++;
        }
    }

    // printf("--> %f\n", res->tensor[out_index + 0 + 0]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 1]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 2]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 3]);

    // exit(0);

}












void sve_vect_mul_compact_interleaved2D_non_tiled_out_l1l2(
    const uint32_t *vect_idxs, // packed indexes for the current tile (already filtered for the tile dimensions)
    uint32_t vect_size, // packed indexes vector size (number of 32-bits words with packed indexes)
    uint32_t n_vect_elems, // input vector length (K dimension)
    float *mat, // activation matrix
    uint32_t tile_l1_h, 
    uint32_t tile_l1_w, 
    uint32_t tile_l2_w, 
    uint32_t full_out_w, // full output width (to compute the correct output indexes based on the tile position)
    const float *codebook_interl, // interleaved codebooks (already filtered for the tile dimensions)
    tensor3D_t *res, // output tensor
    uint32_t out_index, // not used
    int base_res_idx // the position of the first element of the output tile within the output tensor (used to compute the correct output indexes based on the tile position)
){

    uint8_t interl_factor_2D = 2;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f); // Duplicate all the lanes with 0.0f
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); // Duplicate the index mask for all lanes (#define IDX_MASK		a0b111)

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x2_t in_vals;        // Holds the input values


    svfloat32_t weights_0 = svdup_n_f32(0.0);
    svfloat32_t weights_1 = svdup_n_f32(0.0);


    #if defined(N_SVE_REG_CB_1)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl); // Load the interleaved codebooks
    svfloat32_t cb0 = svget2_f32(codebooks_loaded, 0); // Get two codebooks
    svfloat32_t cb1 = svget2_f32(codebooks_loaded, 1);

    #elif defined(N_SVE_REG_CB_2)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0.0)); // Get the first part of the first codebook
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0.0)); // Get the first part of the second codebook
    // cb0_2regs = { cb0_part0, cb0_part1 };
    // cb1_2regs = { cb1_part0, cb1_part1 };

    codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[(N_SVE_LANES*4)]); // N_SVE_LANES is the number of float32 lanes
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget2_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget2_f32(codebooks_loaded, 1));

    #elif defined(N_SVE_REG_CB_4)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[2*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0.0));

    codebooks_loaded = svld2_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[3*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget2_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget2_f32(codebooks_loaded, 1));
    #endif
    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]); // load into N_SVE_LANES lanes the packed indexes
                // uint32 word: [index7][index6][index5][index4][index3][index2][index1][index0]
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                    // print_vect_ui32(dup_idxs_pakd);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs);

                        // Load the weights based on the unpacked indexes

                        #ifdef N_SVE_REG_CB_1
                        weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        #endif

                        #ifdef N_SVE_REG_CB_2
                        weights_0 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb0_2regs); // use the same indexes
                        weights_1 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb1_2regs);
                        #endif

                        #ifdef N_SVE_REG_CB_4
                        weights_0 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb0_4regs);
                        weights_1 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb1_4regs);
                        #endif
                        

                        // Load the input values based on the unpacked indexes
                        in_vals = svld2_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_2D]);

                        // printf("Load from index: ((%d*%d) + %d) * %d = %d\n", col_cnt, n_vect_elems, input_idx, interl_factor_2D, ((col_cnt*n_vect_elems) + input_idx) * interl_factor_2D);

                        svfloat32_t in_0 = svget2_f32(in_vals, 0);
                        svfloat32_t in_1 = svget2_f32(in_vals, 1);
                        // print_vect_f32(in_0);

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                    }
                }
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_2D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_2D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_2D)) * interl_factor_2D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile


            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_2D, full_out_w, ct, interl_factor_2D, base_index);
            // printf("IDXS: %d %d\n", base_index, base_index+1);


            // Compute the final output value by adding all the lanes
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);

            // if(base_index == 2){
            //     print_vect_f32(row_res_vect0);
            //     // printf("--> %f + %f\n", res->tensor[base_index + 0], svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            //     printf("--> %f\n", res->tensor[base_index + 0]);
            //     printf("--\n");
            // }

            col_cnt++;
        }
    }
    // exit(0);
}

















void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x4_t in_vals;        // Holds the input values


    svfloat32_t weights_0 = svdup_n_f32(0.0);
    svfloat32_t weights_1 = svdup_n_f32(0.0);
    svfloat32_t weights_2 = svdup_n_f32(0.0);
    svfloat32_t weights_3 = svdup_n_f32(0.0);

    // svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    // svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    // svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    // svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    // svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

    #elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[2*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[3*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
    #endif
    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("========\n");
            // printf("vect_size | N_SVE_LANES =  %d | %d\n", vect_size, N_SVE_LANES);
            // printf("IDXS_PER_WORD | N_SVE_LANES =  %d | %d\n", IDXS_PER_WORD, N_SVE_LANES);

            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
            row_res_vect2 = svdup_n_f32(0.0f);
            row_res_vect3 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                    // print_vect_ui32(dup_idxs_pakd);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs);
                        // print_vect_ui32(unpkd_idxs);

                        // Load the weights based on the unpacked indexes
                        // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                        // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);

                        #ifdef N_SVE_REG_CB_1
                        weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        weights_3 = svtbl_f32(cb3, unpkd_idxs);
                        #endif

                        #ifdef N_SVE_REG_CB_2
                        weights_0 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb0_2regs);
                        weights_1 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb1_2regs);
                        weights_2 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb2_2regs);
                        weights_3 = extract_weightsx2(bits_mask_pg, unpkd_idxs, cb3_2regs);
                        // print_vect_f32(weights_0);
                        // print_vect_f32(weights_1);
                        // print_vect_f32(weights_2);
                        // print_vect_f32(weights_3);
                        #endif

                        #ifdef N_SVE_REG_CB_4
                        weights_0 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb0_4regs);
                        weights_1 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb1_4regs);
                        weights_2 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb2_4regs);
                        weights_3 = extract_weightsx4(bits_mask_pg, unpkd_idxs, cb3_4regs);
                        // print_vect_f32(weights_0);
                        #endif
                        
                        // print_vect_f32(weights_0);

                        // unpkd_idxs = svlsl_u32_z(bits_mask_pg, unpkd_idxs, svdup_u32(BITS_PER_CB));
                        // svfloat32_t weights_0 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[0], unpkd_idxs);
                        // svfloat32_t weights_1 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[1], unpkd_idxs);
                        // svfloat32_t weights_2 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[2], unpkd_idxs);
                        // svfloat32_t weights_3 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[3], unpkd_idxs);
                        // exit(0);

                        

                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
                        //                                                 input_idx, 
                        //                                                 interl_factor_4D, 
                        //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
                        in_vals = svld4_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat32_t in_0 = svget4_f32(in_vals, 0);
                        svfloat32_t in_1 = svget4_f32(in_vals, 1);
                        svfloat32_t in_2 = svget4_f32(in_vals, 2);
                        svfloat32_t in_3 = svget4_f32(in_vals, 3);
                        // print_vect_f32(in_0);
                        // print_vect_f32(in_1);
                        // print_vect_f32(in_2);
                        // print_vect_f32(in_3);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f32_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f32_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
                // exit(0);
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect0);
            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);

            // res[0].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            // res[1].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            // res[2].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            // res[3].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            // break;
            col_cnt++;
        }
    }

    // exit(0);


    // printf("--> %f\n", res->tensor[out_index + 0 + 0]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 1]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 2]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 3]);

}
















void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_no_lanes_loop(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Compute the number of lanes fully occupied by packed indexes
    int n_full_lanes = n_vect_elems / IDXS_PER_WORD;
    double int_part;

    // If there is a non full lane, how many indexes are in there
    int n_idxs_in_non_full_lanes = modf(((float)n_vect_elems / IDXS_PER_WORD), &int_part) * IDXS_PER_WORD;

    // Number of lanes needed for the full tile
    uint32_t n_lanes_needed = n_full_lanes + (n_idxs_in_non_full_lanes>0 ? 1 : 0);

    // Number of indexes contained in each neede lane
    uint32_t vals[n_lanes_needed];
    for (size_t i = 0; i < n_full_lanes; ++i){
        vals[i] = IDXS_PER_WORD;
    }
    vals[n_full_lanes] = n_idxs_in_non_full_lanes;

    // Offset per each lane
    svuint32_t idx_offsets = svindex_u32(0, IDXS_PER_WORD*interl_factor_4D*sizeof(float));

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    // uint32_t missing_total = 0;       // How many indexes are missing to be processed
    // uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    // uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    // svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    // svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    // svfloat32x4_t in_vals;        // Holds the input values


    svfloat32_t weights_0 = svdup_n_f32(0.0);
    svfloat32_t weights_1 = svdup_n_f32(0.0);
    svfloat32_t weights_2 = svdup_n_f32(0.0);
    svfloat32_t weights_3 = svdup_n_f32(0.0);

    // svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    // svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    // svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    // svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    // svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

    #elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[2*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[3*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
    #endif
    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
            row_res_vect2 = svdup_n_f32(0.0f);
            row_res_vect3 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                input_idx = cw * IDXS_PER_WORD;

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many indexes have been processed per lane
                svuint32_t n_idx_per_lane = svdup_u32(0);

                shamts = svdup_u32(0);

                svuint32_t indexes_in_lane = svld1_u32(svwhilelt_b32(cw, n_lanes_needed), &vals[cw]);

                for(int16_t idx_ptr=0; idx_ptr<svmaxv_u32(svwhilelt_b32(cw, n_lanes_needed), indexes_in_lane); idx_ptr++){

                    svbool_t proc_pg = svcmplt_u32(load_pg, n_idx_per_lane, indexes_in_lane);
                    // print_predicate_w(proc_pg);

                    // print_vect_ui32(shamts);

                    // Perform the MASK+SHIFT for the considered word of packed indexes  
                    unpkd_idxs = svlsr_u32_z(proc_pg, packed_idxs, shamts);
                    unpkd_idxs = svand_u32_z(proc_pg, base_masks, unpkd_idxs);
                    // print_vect_ui32(unpkd_idxs);

                    #ifdef N_SVE_REG_CB_1
                    weights_0 = svtbl_f32(cb0, unpkd_idxs);
                    weights_1 = svtbl_f32(cb1, unpkd_idxs);
                    weights_2 = svtbl_f32(cb2, unpkd_idxs);
                    weights_3 = svtbl_f32(cb3, unpkd_idxs);
                    #endif

                    #ifdef N_SVE_REG_CB_2
                    weights_0 = extract_weightsx2(proc_pg, unpkd_idxs, cb0_2regs);
                    weights_1 = extract_weightsx2(proc_pg, unpkd_idxs, cb1_2regs);
                    weights_2 = extract_weightsx2(proc_pg, unpkd_idxs, cb2_2regs);
                    weights_3 = extract_weightsx2(proc_pg, unpkd_idxs, cb3_2regs);
                    // print_vect_f32(weights_0);
                    // print_vect_f32(weights_1);
                    // print_vect_f32(weights_2);
                    // print_vect_f32(weights_3);
                    #endif

                    #ifdef N_SVE_REG_CB_4
                    weights_0 = extract_weightsx4(proc_pg, unpkd_idxs, cb0_4regs);
                    weights_1 = extract_weightsx4(proc_pg, unpkd_idxs, cb1_4regs);
                    weights_2 = extract_weightsx4(proc_pg, unpkd_idxs, cb2_4regs);
                    weights_3 = extract_weightsx4(proc_pg, unpkd_idxs, cb3_4regs);
                    // print_vect_f32(weights_0);
                    #endif

                    // printf("Start index: %d\n", (((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D));
                    svfloat32_t in_0 = svld1_gather_u32offset_f32(proc_pg, &mat[(((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D)], idx_offsets);
                    svfloat32_t in_1 = svld1_gather_u32offset_f32(proc_pg, &mat[((((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D)) + 1], idx_offsets);
                    svfloat32_t in_2 = svld1_gather_u32offset_f32(proc_pg, &mat[((((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D)) + 2], idx_offsets);
                    svfloat32_t in_3 = svld1_gather_u32offset_f32(proc_pg, &mat[((((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D)) + 3], idx_offsets);

                    input_idx++;


                    row_res_vect0 = svmad_f32_x(proc_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f32_x(proc_pg, in_1, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f32_x(proc_pg, in_2, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f32_x(proc_pg, in_3, weights_3, row_res_vect3);


                    n_idx_per_lane = svadd_u32_m(load_pg, n_idx_per_lane, svdup_u32(1));
                    shamts = svadd_u32_z(load_pg, shamts, svdup_u32(BITS_PER_CB));
                }
            }

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            
            col_cnt++;
        }
    }
}




















void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq(const uint32_t *vect_idxs_interl, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs_0;     // Holds the words with the packed indexes
    svuint32_t packed_idxs_1;
    svuint32_t packed_idxs_2;
    svuint32_t packed_idxs_3;
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    svuint32_t dup_idxs_pakd_0;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t dup_idxs_pakd_1;
    svuint32_t dup_idxs_pakd_2;
    svuint32_t dup_idxs_pakd_3;
    svuint32_t unpkd_idxs_0;      // Holds the unpacked indexes (one per lane)
    svuint32_t unpkd_idxs_1;
    svuint32_t unpkd_idxs_2;
    svuint32_t unpkd_idxs_3;
    svfloat32x4_t in_vals;        // Holds the input values


    svfloat32_t weights_0 = svdup_n_f32(0.0);
    svfloat32_t weights_1 = svdup_n_f32(0.0);
    svfloat32_t weights_2 = svdup_n_f32(0.0);
    svfloat32_t weights_3 = svdup_n_f32(0.0);

    // svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    // svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    // svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    // svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    // svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), codebook_interl);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

    #elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*4)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[2*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0.0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0.0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0.0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0.0));

    codebooks_loaded = svld4_f32(svwhilelt_b32(0, CB_SIZE), &codebook_interl[3*(N_SVE_LANES*4)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));

    #endif
    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
            row_res_vect2 = svdup_n_f32(0.0f);
            row_res_vect3 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                // packed_idxs = svld4_f32(load_pg, &vect_idxs[cw])
                svuint32x4_t packed_idxs = svld4_u32(load_pg, &vect_idxs_interl[cw * 4]);
                packed_idxs_0 = svget4_u32(packed_idxs, 0);
                packed_idxs_1 = svget4_u32(packed_idxs, 1);
                packed_idxs_2 = svget4_u32(packed_idxs, 2);
                packed_idxs_3 = svget4_u32(packed_idxs, 3);

                // print_vect_ui32(packed_idxs_0);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd_0 = svdup_lane_u32(packed_idxs_0, lane);
                    dup_idxs_pakd_1 = svdup_lane_u32(packed_idxs_1, lane);
                    dup_idxs_pakd_2 = svdup_lane_u32(packed_idxs_2, lane);
                    dup_idxs_pakd_3 = svdup_lane_u32(packed_idxs_3, lane);
                    // print_vect_ui32(dup_idxs_pakd);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs_0 = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd_0, shamts);
                        unpkd_idxs_1 = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd_1, shamts);
                        unpkd_idxs_2 = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd_2, shamts);
                        unpkd_idxs_3 = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd_3, shamts);

                        unpkd_idxs_0 = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs_0);
                        unpkd_idxs_1 = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs_1);
                        unpkd_idxs_2 = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs_2);
                        unpkd_idxs_3 = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs_3);

                        // Load the weights based on the unpacked indexes
                        // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                        // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);

                        #ifdef N_SVE_REG_CB_1
                        weights_0 = svtbl_f32(cb0, unpkd_idxs_0);
                        weights_1 = svtbl_f32(cb1, unpkd_idxs_1);
                        weights_2 = svtbl_f32(cb2, unpkd_idxs_2);
                        weights_3 = svtbl_f32(cb3, unpkd_idxs_3);
                        #endif

                        #ifdef N_SVE_REG_CB_2
                        weights_0 = extract_weightsx2(bits_mask_pg, unpkd_idxs_0, cb0_2regs);
                        weights_1 = extract_weightsx2(bits_mask_pg, unpkd_idxs_1, cb1_2regs);
                        weights_2 = extract_weightsx2(bits_mask_pg, unpkd_idxs_2, cb2_2regs);
                        weights_3 = extract_weightsx2(bits_mask_pg, unpkd_idxs_3, cb3_2regs);
                        // print_vect_f32(weights_0);
                        // print_vect_f32(weights_1);
                        // print_vect_f32(weights_2);
                        // print_vect_f32(weights_3);
                        #endif

                        #ifdef N_SVE_REG_CB_4
                        weights_0 = extract_weightsx4(bits_mask_pg, unpkd_idxs_0, cb0_4regs);
                        weights_1 = extract_weightsx4(bits_mask_pg, unpkd_idxs_1, cb1_4regs);
                        weights_2 = extract_weightsx4(bits_mask_pg, unpkd_idxs_2, cb2_4regs);
                        weights_3 = extract_weightsx4(bits_mask_pg, unpkd_idxs_3, cb3_4regs);
                        // print_vect_f32(weights_0);
                        #endif

                        // unpkd_idxs = svlsl_u32_z(bits_mask_pg, unpkd_idxs, svdup_u32(BITS_PER_CB));
                        // svfloat32_t weights_0 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[0], unpkd_idxs);
                        // svfloat32_t weights_1 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[1], unpkd_idxs);
                        // svfloat32_t weights_2 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[2], unpkd_idxs);
                        // svfloat32_t weights_3 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[3], unpkd_idxs);
                        // exit(0);

                        

                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
                        //                                                 input_idx, 
                        //                                                 interl_factor_4D, 
                        //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
                        in_vals = svld4_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat32_t in_0 = svget4_f32(in_vals, 0);
                        svfloat32_t in_1 = svget4_f32(in_vals, 1);
                        svfloat32_t in_2 = svget4_f32(in_vals, 2);
                        svfloat32_t in_3 = svget4_f32(in_vals, 3);
                        // print_vect_f32(in_0);
                        // print_vect_f32(in_1);
                        // print_vect_f32(in_2);
                        // print_vect_f32(in_3);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f32_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f32_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect0);
            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            
            // res[0].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            // res[1].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            // res[2].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            // res[3].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            // exit(0);
            // break;
            col_cnt++;
        }
    }


    // printf("--> %f\n", res->tensor[out_index + 0 + 0]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 1]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 2]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 3]);

    // exit(0);

}










void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_mem(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float *codebook_interl, tensor3D_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat32_t row_res_vect0 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect1 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect2 = svdup_n_f32(0.0f);
    svfloat32_t row_res_vect3 = svdup_n_f32(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint32_t shamts;          // Shift amounts for the masks
    svuint32_t base_masks = svdup_u32(IDX_MASK); 

    svuint32_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint32_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat32x4_t in_vals;        // Holds the input values


    svfloat32_t weights_0 = svdup_n_f32(0.0);
    svfloat32_t weights_1 = svdup_n_f32(0.0);
    svfloat32_t weights_2 = svdup_n_f32(0.0);
    svfloat32_t weights_3 = svdup_n_f32(0.0);


    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f32(0.0f);
            row_res_vect1 = svdup_n_f32(0.0f);
            row_res_vect2 = svdup_n_f32(0.0f);
            row_res_vect3 = svdup_n_f32(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u32(packed_idxs, lane);
                    // print_vect_ui32(dup_idxs_pakd);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
                        missing_lane = (IDXS_PER_WORD - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b32((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u32((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u32_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u32_z(bits_mask_pg, base_masks, unpkd_idxs);

                        // Load the weights based on the unpacked indexes
                        // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                        // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);
                        

                        weights_0 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[0], svmul_n_u32_m(bits_mask_pg, unpkd_idxs, 4));
                        weights_1 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[1], svmul_n_u32_m(bits_mask_pg, unpkd_idxs, 4));
                        weights_2 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[2], svmul_n_u32_m(bits_mask_pg, unpkd_idxs, 4));
                        weights_3 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[3], svmul_n_u32_m(bits_mask_pg, unpkd_idxs, 4));


                        // unpkd_idxs = svlsl_u32_z(bits_mask_pg, unpkd_idxs, svdup_u32(BITS_PER_CB));
                        // svfloat32_t weights_0 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[0], unpkd_idxs);
                        // svfloat32_t weights_1 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[1], unpkd_idxs);
                        // svfloat32_t weights_2 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[2], unpkd_idxs);
                        // svfloat32_t weights_3 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[3], unpkd_idxs);
                        // exit(0);

                        

                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
                        //                                                 input_idx, 
                        //                                                 interl_factor_4D, 
                        //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
                        in_vals = svld4_f32(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat32_t in_0 = svget4_f32(in_vals, 0);
                        svfloat32_t in_1 = svget4_f32(in_vals, 1);
                        svfloat32_t in_2 = svget4_f32(in_vals, 2);
                        svfloat32_t in_3 = svget4_f32(in_vals, 3);
                        // print_vect_f32(in_0);
                        // print_vect_f32(in_1);
                        // print_vect_f32(in_2);
                        // print_vect_f32(in_3);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b32(bits_mask_pg, bits_mask_pg);

                        // MAC
                        // printf("Before:\n");
                        // print_vect_f32(row_res_vect0);
                        // printf("In:\n");
                        // print_vect_f32(in_0);
                        // printf("Weights:\n");
                        // print_vect_f32(weights_0);
                        row_res_vect0 = svmad_f32_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        // printf("After:\n");
                        // print_vect_f32(row_res_vect0);
                        row_res_vect1 = svmad_f32_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f32_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f32_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
            }

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect0);
            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);
            res->tensor[base_index + 0] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            
            // res[0].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0);
            // res[1].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1);
            // res[2].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2);
            // res[3].tensor[out_index + r] = svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3);
            // break;
            col_cnt++;
        }
    }


    // printf("--> %f\n", res->tensor[out_index + 0 + 0]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 1]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 2]);
    // printf("--> %f\n", res->tensor[out_index + 0 + 3]);

    // exit(0);

}











// void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(const uint16_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx){

//     uint8_t interl_factor_4D = 4;

//     // Vect register to store the SIMD intermediate results of a row
//     svfloat16_t row_res_vect0 = svdup_n_f16(0.0f);
//     svfloat16_t row_res_vect1 = svdup_n_f16(0.0f);
//     svfloat16_t row_res_vect2 = svdup_n_f16(0.0f);
//     svfloat16_t row_res_vect3 = svdup_n_f16(0.0f);

    
//     uint32_t missing_total = 0;       // How many indexes are missing to be processed
//     uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
//     uint32_t input_idx = 0;     // Index to the next input element to be loaded

//     // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
//     svbool_t load_pg;           // Predicate for loading the indexes
//     svuint16_t packed_idxs;     // Holds the words with the packed indexes
//     uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

//     svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
//     svuint16_t shamts;          // Shift amounts for the masks
//     svuint16_t base_masks = svdup_u16(IDX_MASK); 

//     svuint16_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
//     svuint16_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
//     svfloat16x4_t in_vals;        // Holds the input values


//     svfloat16_t weights_0 = svdup_n_f16(0.0);
//     svfloat16_t weights_1 = svdup_n_f16(0.0);
//     svfloat16_t weights_2 = svdup_n_f16(0.0);
//     svfloat16_t weights_3 = svdup_n_f16(0.0);

//     #if defined(N_SVE_REG_CB_F16_1)
//     svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), codebook_interl);
//     svfloat16_t cb0 = svget4_f16(codebooks_loaded, 0);
//     svfloat16_t cb1 = svget4_f16(codebooks_loaded, 1);
//     svfloat16_t cb2 = svget4_f16(codebooks_loaded, 2);
//     svfloat16_t cb3 = svget4_f16(codebooks_loaded, 3);

//     #elif defined(N_SVE_REG_CB_F16_2)
//     svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*8)]);
//     svfloat16x2_t cb0_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 0), svdup_n_f16((float16_t)0.0));
//     svfloat16x2_t cb1_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 1), svdup_n_f16((float16_t)0.0));
//     svfloat16x2_t cb2_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 2), svdup_n_f16((float16_t)0.0));
//     svfloat16x2_t cb3_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 3), svdup_n_f16((float16_t)0.0));

    
//     codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*8)]);
//     cb0_2regs = svcreate2_f16(svget2_f16(cb0_2regs, 0), svget4_f16(codebooks_loaded, 0));
//     cb1_2regs = svcreate2_f16(svget2_f16(cb1_2regs, 0), svget4_f16(codebooks_loaded, 1));
//     cb2_2regs = svcreate2_f16(svget2_f16(cb2_2regs, 0), svget4_f16(codebooks_loaded, 2));
//     cb3_2regs = svcreate2_f16(svget2_f16(cb3_2regs, 0), svget4_f16(codebooks_loaded, 3));
//     #endif

//     int col_cnt = 0;

//     // printf("Vect size: %d\n", vect_size);

//     // Loop thorugh the columns of weights
//     // for(int r=0; r<mat_cols; r++){
//     for(int rt=0; rt<tile_l1_h; rt++){
//         for(int ct=0; ct<tile_l1_w; ct++){
                
//             // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
//             row_res_vect0 = svdup_n_f16(0.0f);
//             row_res_vect1 = svdup_n_f16(0.0f);
//             row_res_vect2 = svdup_n_f16(0.0f);
//             row_res_vect3 = svdup_n_f16(0.0f);
        
//             input_idx = 0;

//             // Loop thorugh the elements of the weights indexes (compact)
//             // It indexes the column words (the words that contains the indexes per each column)
//             for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
//                 // printf("---- CW %d ----\n", cw);

//                 load_pg = svwhilelt_b16(cw, vect_size);

//                 // Load a 32-bits word with IDXS_PER_WORD packed indexes
//                 packed_idxs = svld1_u16(load_pg, &vect_idxs[cw]);
//                 // print_vect_ui16(packed_idxs);

//                 // Counts how many lanes have been loaded
//                 n_loaded_lanes = svcntp_b16(load_pg, load_pg);
//                 // printf("N loaded lanes: %d\n", n_loaded_lanes);


//                 // Loop through the 32-bits lanes of the vector register
//                 for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
//                     // printf("---- LANE %ld ----\n", lane);

//                     missing_lane = 0;

//                     // Duplicates a single word of packed indexes in all the lanes
//                     dup_idxs_pakd = svdup_lane_u16(packed_idxs, lane);
//                     // print_vect_ui16(dup_idxs_pakd);
//                     // exit(0);

//                     for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
//                         // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                        
//                         missing_lane = (IDXS_PER_WORD - idx_ptr);
//                         missing_total = (n_vect_elems - input_idx);
//                         // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
//                         // Stop in case there are no more missing indexes to process
//                         if (missing_total<=0){
//                             // printf("Exiting...\n");
//                             break;
//                         }

//                         if(missing_lane <= missing_total){
//                             bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_lane);
//                         }else{
//                             bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_total);
//                         }
//                         // print_predicate_w(bits_mask_pg);

//                         // Create the shift amounts to address the correct portion of indexes within the word
//                         shamts = svindex_u16((idx_ptr*BITS_PER_CB), BITS_PER_CB);
//                         // print_vect_ui32(shamts);

//                         // Perform the MASK+SHIFT for the considered word of packed indexes  
//                         unpkd_idxs = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd, shamts);
//                         unpkd_idxs = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs);
//                         // print_vect_ui16(unpkd_idxs);

//                         // Load the weights based on the unpacked indexes
//                         // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
//                         // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
//                         // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
//                         // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
//                         // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);
                        
//                         #if defined(N_SVE_REG_CB_F16_1)
//                         weights_0 = svtbl_f16(cb0, unpkd_idxs);
//                         weights_1 = svtbl_f16(cb1, unpkd_idxs);
//                         weights_2 = svtbl_f16(cb2, unpkd_idxs);
//                         weights_3 = svtbl_f16(cb3, unpkd_idxs);
//                         // weights_0 = get_weights_f32(cb0, unpkd_idxs);
//                         // weights_1 = get_weights_f32(cb1, unpkd_idxs);
//                         // weights_2 = get_weights_f32(cb2, unpkd_idxs);
//                         // weights_3 = get_weights_f32(cb3, unpkd_idxs);
                        
//                         #elif defined(N_SVE_REG_CB_F16_2)
//                         // weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb0_2regs);
//                         weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb0_2regs);
//                         weights_1 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb1_2regs);
//                         weights_2 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb2_2regs);
//                         weights_3 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb3_2regs);
//                         // print_vect_f16(weights_0);
//                         // exit(0);
//                         #endif

//                         // unpkd_idxs = svlsl_u32_z(bits_mask_pg, unpkd_idxs, svdup_u32(BITS_PER_CB));
//                         // svfloat32_t weights_0 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[0], unpkd_idxs);
//                         // svfloat32_t weights_1 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[1], unpkd_idxs);
//                         // svfloat32_t weights_2 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[2], unpkd_idxs);
//                         // svfloat32_t weights_3 = svld1_gather_u32index_f32(bits_mask_pg, &codebook_interl[3], unpkd_idxs);
//                         // exit(0);

                        

//                         // Load the input values based on the unpacked indexes
//                         // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
//                         // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
//                         // print_vect_f32(in_vals);
//                         // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
//                         //                                                 input_idx, 
//                         //                                                 interl_factor_4D, 
//                         //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
//                         in_vals = svld4_f16(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
//                         svfloat16_t in_0 = svget4_f16(in_vals, 0);
//                         svfloat16_t in_1 = svget4_f16(in_vals, 1);
//                         svfloat16_t in_2 = svget4_f16(in_vals, 2);
//                         svfloat16_t in_3 = svget4_f16(in_vals, 3);
//                         // print_vect_f32(in_0);
//                         // print_vect_f32(in_1);
//                         // print_vect_f32(in_2);
//                         // print_vect_f32(in_3);
//                         // exit(0);
//                         // printf("\n\n");

//                         input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

//                         // MAC
//                         // printf("Before:\n");
//                         // print_vect_f16(row_res_vect0);
//                         // printf("In:\n");
//                         // print_vect_f16(in_0);
//                         // printf("Weights:\n");
//                         // print_vect_f16(weights_0);
//                         row_res_vect0 = svmad_f16_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
//                         // printf("After:\n");
//                         // print_vect_f16(row_res_vect0);
//                         // exit(0);
//                         row_res_vect1 = svmad_f16_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
//                         row_res_vect2 = svmad_f16_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
//                         row_res_vect3 = svmad_f16_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
//                     }
//                 }
//             }

//             // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

//             uint32_t base_index =   base_res_idx + 
//                                     (rt * full_out_w) + 
//                                     ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
//                                     ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
//             // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
//             // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

//             // print_vect_f32(row_res_vect1);
//             // Compute the final output value by adding all the lanes
//             // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
//             // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
//             // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
//             // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
//             // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);

//             res->tensor[base_index + 0] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
//             res->tensor[base_index + 1] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
//             res->tensor[base_index + 2] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
//             res->tensor[base_index + 3] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3); 

//             // volatile float tmp0 = (float)svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
//             // volatile float tmp1 = (float)svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
//             // volatile float tmp2 = (float)svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
//             // volatile float tmp3 = (float)svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3);

//             // volatile float res_f0 = (float)res->tensor[base_index + 0];
//             // volatile float res_f1 = (float)res->tensor[base_index + 1];
//             // volatile float res_f2 = (float)res->tensor[base_index + 2];
//             // volatile float res_f3 = (float)res->tensor[base_index + 3];

//             // res->tensor[base_index + 0] = (float16_t)(res_f0 + tmp0);
//             // res->tensor[base_index + 1] = (float16_t)(res_f1 + tmp1);
//             // res->tensor[base_index + 2] = (float16_t)(res_f2 + tmp2);
//             // res->tensor[base_index + 3] = (float16_t)(res_f3 + tmp3);
     
//             col_cnt++;
//         }
//     }
// }