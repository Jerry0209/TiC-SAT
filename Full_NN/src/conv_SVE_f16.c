
#include <SVE_implementations.h>



void sve_vect_mul_compact_non_tiled_out_l1l2_f16(const uint16_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx);

void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx);

void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16_optim(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx);

void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq_f16(const uint16_t *vect_idxs_interl, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx);




void conv3D_staticPatch_compact_tiled_L1L2_SVE_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint16_t *kernel, const float16_t *codebook, const float16_t *bias, uint32_t tile_L2, uint32_t tile_L1, tensor3D_f16_t *out){

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
    float16_t *flat_patches_matrix = (float16_t*)malloc((kernel_d * (kernel_h * kernel_w) * (tile_L2 * tile_L2)) * sizeof(float16_t));
    float16_t *flat_patches_matrix_tiled_l1 = (float16_t*)malloc((tile_L1 * tile_L1 * IDXS_PER_WORD) * sizeof(float16_t));   // Rearranged for L1 tiling

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

                    get_3Dpatch_f16(&input->tensor[in_tensor_idx], 
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
                    

                    get_3Dpatch_f16(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
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

                            sve_vect_mul_compact_non_tiled_out_l1l2_f16(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
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






void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_diff_seq_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint16_t *kernel_interl, const float16_t *codebook_interl, const float16_t *bias, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_f16_t *out){

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
    float16_t *flat_patches_matrix = (float16_t*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float16_t));
    float16_t *flat_patches_matrix_tiled_l1 = (float16_t*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float16_t));
    
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

                    get_3Dpatch_f16(&input->tensor[in_tensor_idx], 
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
                        get_3Dpatch_f16(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
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
                                sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq_f16(&kernel_interl[((twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)) * 4], 
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
    // exit(0);

    // system("m5 dumpresetstats");
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}





void conv3D_staticPatch_compactSVE_interleavedND_tiled_L1L2_f16(conv_t conv_layer, tensor3D_f16_t *input, const uint32_t *kernel, const float16_t *codebook_interl, const float16_t *bias_interl, uint8_t interl_factor, uint32_t tile_l2, uint32_t tile_l1, tensor3D_f16_t *out){

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
    float16_t *flat_patches_matrix = (float16_t*)malloc((patch_dims.depth * (patch_dims.height * patch_dims.width) * (tile_l2 * tile_l2)) * sizeof(float16_t));
    float16_t *flat_patches_matrix_tiled_l1 = (float16_t*)malloc(tile_l1 * tile_l1 * IDXS_PER_WORD * interl_factor * sizeof(float16_t));
    
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

                    get_3Dpatch_f16(&input->tensor[in_tensor_idx], 
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
                        get_3Dpatch_f16(flat_patches_matrix, &fpm_dim, &p_idx, &p_dim, 0, flat_patches_matrix_tiled_l1);
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

                                // sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
                                sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16_optim(&kernel[(twl1 * conv_layer.out_ch) + (ch * tile_l1_w_elems)], 
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

                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 0] += bias_interl[(ch*interl_factor) + 0];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 1] += bias_interl[(ch*interl_factor) + 1];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 2] += bias_interl[(ch*interl_factor) + 2];
                out->tensor[(((ch*conv_layer.output_dim.height*conv_layer.output_dim.width) + (h*conv_layer.output_dim.width) + w) * interl_factor) + 3] += bias_interl[(ch*interl_factor) + 3];

            }
        }
    }

    // printf("==> %f\n", out->tensor[0]);
    // exit(0);


    // system("m5 dumpresetstats");
    
    free(flat_patches_matrix);
    free(flat_patches_matrix_tiled_l1);
}























void sve_vect_mul_compact_non_tiled_out_l1l2_f16(const uint16_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx){

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect = svdup_n_f16(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint16_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint16_t shamts;          // Shift amounts for the masks
    svuint16_t base_masks = svdup_u16(IDX_MASK); 

    svuint16_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat16_t in_vals;        // Holds the input values


    svfloat16_t weights = svdup_n_f16(0.0);

    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16_t cb = svld1_f16(svwhilelt_b16(0, CB_SIZE), codebook);
    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16_t cb0 = svld1_f16(svwhilelt_b16(0, CB_SIZE), codebook);
    svfloat16_t cb1 = svld1_f16(svwhilelt_b16(0, CB_SIZE), &codebook[8]);
    svfloat16x2_t cb = svcreate2_f16(cb0, cb1);
    #endif

    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect = svdup_n_f16(0.0f);
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b16(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u16(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b16(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u16(packed_idxs, lane);
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
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u16((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs);

                        // Load the weights based on the unpacked indexes
                        #ifdef N_SVE_REG_CB_F16_1
                        weights = svtbl_f16(cb, unpkd_idxs);
                        #elif defined(N_SVE_REG_CB_F16_2)
                        weights = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb);
                        #endif
                        
                        // Load the input values based on the unpacked indexes
                        // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                        // in_vals = svld1_f32(bits_mask_pg, &mat[(r*n_vect_elems) + input_idx]);
                        // print_vect_f32(in_vals);
                        // printf("Mat idx: ((%d*%d) + %d) * %d = %d\n",   col_cnt, n_vect_elems, 
                        //                                                 input_idx, 
                        //                                                 interl_factor_4D, 
                        //                                                 ((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D);
                        in_vals = svld1_f16(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx)]);
                        // print_vect_f32(in_0);
                        // exit(0);
                        // printf("\n\n");


                        input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect = svmad_f16_x(bits_mask_pg, in_vals, weights, row_res_vect);
                    
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
            
            res->tensor[base_index] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect);
            
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








void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_diff_seq_f16(const uint16_t *vect_idxs_interl, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect0 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect1 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect2 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect3 = svdup_n_f16(0.0f);
    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint16_t packed_idxs_0;     // Holds the words with the packed indexes
    svuint16_t packed_idxs_1;
    svuint16_t packed_idxs_2;
    svuint16_t packed_idxs_3;
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint16_t shamts;          // Shift amounts for the masks
    svuint16_t base_masks = svdup_u16(IDX_MASK); 

    svuint16_t dup_idxs_pakd_0;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t dup_idxs_pakd_1;
    svuint16_t dup_idxs_pakd_2;
    svuint16_t dup_idxs_pakd_3;
    svuint16_t unpkd_idxs_0;      // Holds the unpacked indexes (one per lane)
    svuint16_t unpkd_idxs_1;
    svuint16_t unpkd_idxs_2;
    svuint16_t unpkd_idxs_3;
    svfloat16x4_t in_vals;        // Holds the input values


    svfloat16_t weights_0 = svdup_n_f16(0.0);
    svfloat16_t weights_1 = svdup_n_f16(0.0);
    svfloat16_t weights_2 = svdup_n_f16(0.0);
    svfloat16_t weights_3 = svdup_n_f16(0.0);


    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), codebook_interl);
    svfloat16_t cb0 = svget4_f16(codebooks_loaded, 0);
    svfloat16_t cb1 = svget4_f16(codebooks_loaded, 1);
    svfloat16_t cb2 = svget4_f16(codebooks_loaded, 2);
    svfloat16_t cb3 = svget4_f16(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*8)]);
    svfloat16x2_t cb0_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 0), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb1_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 1), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb2_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 2), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb3_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 3), svdup_n_f16((float16_t)0.0));

    codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*8)]);
    cb0_2regs = svcreate2_f16(svget2_f16(cb0_2regs, 0), svget4_f16(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f16(svget2_f16(cb1_2regs, 0), svget4_f16(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f16(svget2_f16(cb2_2regs, 0), svget4_f16(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f16(svget2_f16(cb3_2regs, 0), svget4_f16(codebooks_loaded, 3));
    #endif

    int col_cnt = 0;

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f16(0.0f);
            row_res_vect1 = svdup_n_f16(0.0f);
            row_res_vect2 = svdup_n_f16(0.0f);
            row_res_vect3 = svdup_n_f16(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b16(cw, vect_size);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                // packed_idxs = svld4_f32(load_pg, &vect_idxs[cw])
                svuint16x4_t packed_idxs = svld4_u16(load_pg, &vect_idxs_interl[cw * interl_factor_4D]);
                packed_idxs_0 = svget4_u16(packed_idxs, 0);
                packed_idxs_1 = svget4_u16(packed_idxs, 1);
                packed_idxs_2 = svget4_u16(packed_idxs, 2);
                packed_idxs_3 = svget4_u16(packed_idxs, 3);

                // print_vect_ui32(packed_idxs_0);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b16(load_pg, load_pg);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);


                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd_0 = svdup_lane_u16(packed_idxs_0, lane);
                    dup_idxs_pakd_1 = svdup_lane_u16(packed_idxs_1, lane);
                    dup_idxs_pakd_2 = svdup_lane_u16(packed_idxs_2, lane);
                    dup_idxs_pakd_3 = svdup_lane_u16(packed_idxs_3, lane);
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
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u16((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs_0 = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd_0, shamts);
                        unpkd_idxs_1 = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd_1, shamts);
                        unpkd_idxs_2 = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd_2, shamts);
                        unpkd_idxs_3 = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd_3, shamts);

                        unpkd_idxs_0 = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs_0);
                        unpkd_idxs_1 = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs_1);
                        unpkd_idxs_2 = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs_2);
                        unpkd_idxs_3 = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs_3);

                        // Load the weights based on the unpacked indexes

                        #ifdef N_SVE_REG_CB_F16_1
                        weights_0 = svtbl_f16(cb0, unpkd_idxs_0);
                        weights_1 = svtbl_f16(cb1, unpkd_idxs_1);
                        weights_2 = svtbl_f16(cb2, unpkd_idxs_2);
                        weights_3 = svtbl_f16(cb3, unpkd_idxs_3);
                        #endif

                        #ifdef N_SVE_REG_CB_F16_2
                        weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_0, cb0_2regs);
                        weights_1 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_1, cb1_2regs);
                        weights_2 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_2, cb2_2regs);
                        weights_3 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_3, cb3_2regs);
                        // print_vect_f32(weights_0);
                        // print_vect_f32(weights_1);
                        // print_vect_f32(weights_2);
                        // print_vect_f32(weights_3);
                        #endif

                        in_vals = svld4_f16(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat16_t in_0 = svget4_f16(in_vals, 0);
                        svfloat16_t in_1 = svget4_f16(in_vals, 1);
                        svfloat16_t in_2 = svget4_f16(in_vals, 2);
                        svfloat16_t in_3 = svget4_f16(in_vals, 3);

                        input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

                        // MAC
                        row_res_vect0 = svmad_f16_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        row_res_vect1 = svmad_f16_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f16_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f16_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
            }

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            res->tensor[base_index + 0] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3);
            col_cnt++;
        }
    }

}









void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx){

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect0 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect1 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect2 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect3 = svdup_n_f16(0.0f);

    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint16_t shamts;          // Shift amounts for the masks
    svuint16_t base_masks = svdup_u16(IDX_MASK); 

    svuint16_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat16x4_t in_vals;        // Holds the input values


    svfloat16_t weights_0 = svdup_n_f16(0.0);
    svfloat16_t weights_1 = svdup_n_f16(0.0);
    svfloat16_t weights_2 = svdup_n_f16(0.0);
    svfloat16_t weights_3 = svdup_n_f16(0.0);

    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), codebook_interl);
    svfloat16_t cb0 = svget4_f16(codebooks_loaded, 0);
    svfloat16_t cb1 = svget4_f16(codebooks_loaded, 1);
    svfloat16_t cb2 = svget4_f16(codebooks_loaded, 2);
    svfloat16_t cb3 = svget4_f16(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*8)]);
    svfloat16x2_t cb0_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 0), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb1_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 1), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb2_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 2), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb3_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 3), svdup_n_f16((float16_t)0.0));

    
    codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*8)]);
    cb0_2regs = svcreate2_f16(svget2_f16(cb0_2regs, 0), svget4_f16(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f16(svget2_f16(cb1_2regs, 0), svget4_f16(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f16(svget2_f16(cb2_2regs, 0), svget4_f16(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f16(svget2_f16(cb3_2regs, 0), svget4_f16(codebooks_loaded, 3));
    #endif

    int col_cnt = 0;

    // printf("Vect size: %d\n", vect_size);

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("========\n");
            // printf("vect_size | N_SVE_LANES =  %d | %d\n", vect_size, N_SVE_LANES);
            // printf("IDXS_PER_WORD_16 | N_SVE_LANES =  %d | %d\n", IDXS_PER_WORD_16, N_SVE_HALF);

            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f16(0.0f);
            row_res_vect1 = svdup_n_f16(0.0f);
            row_res_vect2 = svdup_n_f16(0.0f);
            row_res_vect3 = svdup_n_f16(0.0f);
        
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);
                print_predicate_w(load_pg);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                svuint16_t packed_idxs_u16 = svreinterpret_u16_u32(packed_idxs);
                // print_vect_ui16(packed_idxs_u16);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                n_loaded_lanes *= 2; // lanes of 16-bits
                // printf("Vect Size = %d\n", vect_size);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);

                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);

                    missing_lane = 0;

                    // Duplicates a single word of packed indexes in all the lanes
                    dup_idxs_pakd = svdup_lane_u16(packed_idxs_u16, lane);
                    // print_vect_ui16(dup_idxs_pakd);
                    // exit(0);

                    for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD_16; idx_ptr+=N_SVE_HALF){
                        // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);

                        missing_lane = (IDXS_PER_WORD_16 - idx_ptr);
                        missing_total = (n_vect_elems - input_idx);
                        // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                        
                        // Stop in case there are no more missing indexes to process
                        if (missing_total<=0){
                            // printf("Exiting...\n");
                            break;
                        }

                        if(missing_lane <= missing_total){
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_lane);
                        }else{
                            bits_mask_pg = svwhilelt_b16((uint64_t)0, (uint64_t)missing_total);
                        }
                        // print_predicate_w(bits_mask_pg);

                        // Create the shift amounts to address the correct portion of indexes within the word
                        shamts = svindex_u16((idx_ptr*BITS_PER_CB), BITS_PER_CB);
                        // print_vect_ui32(shamts);

                        // Perform the MASK+SHIFT for the considered word of packed indexes  
                        unpkd_idxs = svlsr_u16_z(bits_mask_pg, dup_idxs_pakd, shamts);
                        unpkd_idxs = svand_u16_z(bits_mask_pg, base_masks, unpkd_idxs);
                        // print_vect_ui16(unpkd_idxs);
                        // print_predicate_h(bits_mask_pg);

                        // Load the weights based on the unpacked indexes
                        // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                        // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                        // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                        // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                        // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);
                        
                        #if defined(N_SVE_REG_CB_F16_1)
                        weights_0 = svtbl_f16(cb0, unpkd_idxs);
                        weights_1 = svtbl_f16(cb1, unpkd_idxs);
                        weights_2 = svtbl_f16(cb2, unpkd_idxs);
                        weights_3 = svtbl_f16(cb3, unpkd_idxs);
                        // weights_0 = get_weights_f32(cb0, unpkd_idxs);
                        // weights_1 = get_weights_f32(cb1, unpkd_idxs);
                        // weights_2 = get_weights_f32(cb2, unpkd_idxs);
                        // weights_3 = get_weights_f32(cb3, unpkd_idxs);
                        
                        #elif defined(N_SVE_REG_CB_F16_2)
                        // weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb0_2regs);
                        weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb0_2regs);
                        weights_1 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb1_2regs);
                        weights_2 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb2_2regs);
                        weights_3 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, cb3_2regs);
                        // print_vect_f16(weights_0);
                        // exit(0);
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
                        in_vals = svld4_f16(bits_mask_pg, &mat[((col_cnt*n_vect_elems) + input_idx) * interl_factor_4D]);
                        svfloat16_t in_0 = svget4_f16(in_vals, 0);
                        svfloat16_t in_1 = svget4_f16(in_vals, 1);
                        svfloat16_t in_2 = svget4_f16(in_vals, 2);
                        svfloat16_t in_3 = svget4_f16(in_vals, 3);
                        // print_vect_f32(in_0);
                        // print_vect_f32(in_1);
                        // print_vect_f32(in_2);
                        // print_vect_f32(in_3);
                        // exit(0);
                        // printf("\n\n");

                        input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

                        // MAC
                        // printf("Before:\n");
                        // print_vect_f16(row_res_vect0);
                        // printf("In:\n");
                        // print_vect_f16(in_0);
                        // printf("Weights:\n");
                        // print_vect_f16(weights_0);
                        row_res_vect0 = svmad_f16_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                        // printf("After:\n");
                        // print_vect_f16(row_res_vect0);
                        // exit(0);
                        row_res_vect1 = svmad_f16_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                        row_res_vect2 = svmad_f16_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                        row_res_vect3 = svmad_f16_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                    }
                }
            }

            // exit(0);

            // uint32_t base_index = out_index + (col_cnt * interl_factor_4D);

            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);

            res->tensor[base_index + 0] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3);
     
            col_cnt++;
        }
    }

    // exit(0);
}








void sve_vect_mul_compact_interleaved4D_non_tiled_out_l1l2_f16_optim(const uint32_t *vect_idxs, uint32_t vect_size, uint32_t n_vect_elems, float16_t *mat, uint32_t tile_l1_h, uint32_t tile_l1_w, uint32_t tile_l2_w, uint32_t full_out_w, const float16_t *codebook_interl, tensor3D_f16_t *res, uint32_t out_index, int base_res_idx){

    // printf("N vect elems = %d\n", n_vect_elems);

    uint8_t interl_factor_4D = 4;

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect0 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect1 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect2 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect3 = svdup_n_f16(0.0f);

    
    uint32_t missing_total = 0;       // How many indexes are missing to be processed
    uint32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint32_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint16_t shamts;          // Shift amounts for the masks
    svuint16_t base_masks = svdup_u16(IDX_MASK); 

    svuint16_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat16x4_t in_vals;        // Holds the input values


    svfloat16_t weights_0 = svdup_n_f16(0.0);
    svfloat16_t weights_1 = svdup_n_f16(0.0);
    svfloat16_t weights_2 = svdup_n_f16(0.0);
    svfloat16_t weights_3 = svdup_n_f16(0.0);

    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), codebook_interl);
    svfloat16_t cb0 = svget4_f16(codebooks_loaded, 0);
    svfloat16_t cb1 = svget4_f16(codebooks_loaded, 1);
    svfloat16_t cb2 = svget4_f16(codebooks_loaded, 2);
    svfloat16_t cb3 = svget4_f16(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[0*(N_SVE_LANES*8)]);
    svfloat16x2_t cb0_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 0), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb1_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 1), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb2_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 2), svdup_n_f16((float16_t)0.0));
    svfloat16x2_t cb3_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 3), svdup_n_f16((float16_t)0.0));

    
    codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &codebook_interl[1*(N_SVE_LANES*8)]);
    cb0_2regs = svcreate2_f16(svget2_f16(cb0_2regs, 0), svget4_f16(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f16(svget2_f16(cb1_2regs, 0), svget4_f16(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f16(svget2_f16(cb2_2regs, 0), svget4_f16(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f16(svget2_f16(cb3_2regs, 0), svget4_f16(codebooks_loaded, 3));
    #endif

    int col_cnt = 0;

    int n_idx_processed = 0;

    // printf("Vect size: %d\n", vect_size);

    // Loop thorugh the columns of weights
    // for(int r=0; r<mat_cols; r++){
    for(int rt=0; rt<tile_l1_h; rt++){
        for(int ct=0; ct<tile_l1_w; ct++){
                
            // printf("========\n");
            // printf("vect_size | N_SVE_LANES =  %d | %d\n", vect_size, N_SVE_LANES);
            // printf("IDXS_PER_WORD_16 | N_SVE_LANES =  %d | %d\n", IDXS_PER_WORD_16, N_SVE_HALF);

            // printf("\n>>> Rt, Ct: %d %d\n", rt, ct);
            row_res_vect0 = svdup_n_f16(0.0f);
            row_res_vect1 = svdup_n_f16(0.0f);
            row_res_vect2 = svdup_n_f16(0.0f);
            row_res_vect3 = svdup_n_f16(0.0f);
            
            n_idx_processed = 0;
            input_idx = 0;

            // Loop thorugh the elements of the weights indexes (compact)
            // It indexes the column words (the words that contains the indexes per each column)
            for(uint32_t cw=0; cw<vect_size; cw+=N_SVE_LANES){
                // printf("---- CW %d ----\n", cw);

                load_pg = svwhilelt_b32(cw, vect_size);
                // print_predicate_w(load_pg);

                // Load a 32-bits word with IDXS_PER_WORD packed indexes
                packed_idxs = svld1_u32(load_pg, &vect_idxs[cw]);
                // print_vect_ui32(packed_idxs);

                // svuint16_t packed_idxs_u16 = svreinterpret_u16_u32(packed_idxs);
                // print_vect_ui16(packed_idxs_u16);

                // Counts how many lanes have been loaded
                n_loaded_lanes = svcntp_b32(load_pg, load_pg);
                // n_loaded_lanes *= 2; // lanes of 16-bits
                // printf("Vect Size = %d\n", vect_size);
                // printf("N loaded lanes: %d\n", n_loaded_lanes);

                // Loop through the 32-bits lanes of the vector register
                for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                    // printf("---- LANE %ld ----\n", lane);


                    // Handles the special case in which a full set of 16-bits lanes cannot hold
                    // all the indexes packed in a full 32-bits word
                    #if ((CB_SIZE == 4) || (CB_SIZE == 8)) && (N_SVE_LANES == 4)

                    // missing_lane = 0;

                    svuint32_t dup_idxs_32b = svdup_lane_u32(packed_idxs, lane);
                    dup_idxs_pakd = svreinterpret_u16_u32(dup_idxs_32b);    // reinterpret them as 16 bits words
                    // print_vect_ui16(dup_idxs_pakd);

                    ////// take the first lane //////
                    // printf("FIRST LANE\n");

                    missing_total = n_vect_elems - n_idx_processed;
                    // printf("Missing total = %d - %d = %d\n", n_vect_elems, n_idx_processed, missing_total);

                    
                    svuint16_t pakd_idxs = svdup_lane_u16(dup_idxs_pakd, 0);    
                    // print_vect_ui16(pakd_idxs);

                    int pred_bound = (missing_total < IDXS_PER_WORD_16) ? (missing_total) : (IDXS_PER_WORD_16);
                    svbool_t lane_pg = svwhilelt_b16((uint64_t)0, (uint64_t)pred_bound);
                    // printf("PRED = %d | %d\n", 0, pred_bound);
                    // print_predicate_h(lane_pg);

                    in_vals = svld4_f16(lane_pg, &mat[((col_cnt*n_vect_elems) + n_idx_processed) * interl_factor_4D]);
                    svfloat16_t in_0 = svget4_f16(in_vals, 0);
                    svfloat16_t in_1 = svget4_f16(in_vals, 1);
                    svfloat16_t in_2 = svget4_f16(in_vals, 2);
                    svfloat16_t in_3 = svget4_f16(in_vals, 3);
                    
                    shamts = svindex_u16(0, BITS_PER_CB);

                    unpkd_idxs = svlsr_u16_z(svptrue_b16(), pakd_idxs, shamts);
                    unpkd_idxs = svand_u16_z(svptrue_b16(), base_masks, unpkd_idxs);
                    // print_vect_ui16(unpkd_idxs);

                    #if defined(N_SVE_REG_CB_F16_1)
                    weights_0 = svtbl_f16(cb0, unpkd_idxs);
                    weights_1 = svtbl_f16(cb1, unpkd_idxs);
                    weights_2 = svtbl_f16(cb2, unpkd_idxs);
                    weights_3 = svtbl_f16(cb3, unpkd_idxs);
                    
                    #elif defined(N_SVE_REG_CB_F16_2)
                    weights_0 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb0_2regs);
                    weights_1 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb1_2regs);
                    weights_2 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb2_2regs);
                    weights_3 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb3_2regs);
                    // exit(0);
                    #endif

                    svbool_t mul_pg = lane_pg;

                    // if(missing_total < IDXS_PER_WORD_16){
                    //     // pred_bound*2 to cover the cases in which there are left less indexes than the one a half word can store
                    //     mul_pg = svwhilelt_b16((uint64_t)0, (uint64_t)(pred_bound*2));
                    // } else {
                    //     mul_pg = lane_pg;
                    // }

                    // printf("== MAC ==\n");
                    // print_predicate_h(mul_pg);
                    // print_vect_f16(in_0);
                    // print_vect_ui16(unpkd_idxs);
                    // print_vect_f16(weights_0);

                    // MAC
                    row_res_vect0 = svmad_f16_x(mul_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f16_x(mul_pg, in_1, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f16_x(mul_pg, in_2, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f16_x(mul_pg, in_3, weights_3, row_res_vect3);

                    // print_vect_f16(row_res_vect0);

                    n_idx_processed += svcntp_b16(mul_pg, mul_pg);


                    ////// take the second lane //////
                    // printf("SECOND LANE\n");

                    missing_total = n_vect_elems - n_idx_processed;
                    // printf("Missing total = %d - %d = %d\n", n_vect_elems, n_idx_processed, missing_total);
                    
                    pakd_idxs = svdup_lane_u16(dup_idxs_pakd, 1);    
                    // print_vect_ui16(pakd_idxs);

                    pred_bound = (missing_total < IDXS_PER_WORD_16) ? (missing_total) : (IDXS_PER_WORD_16);
                    lane_pg = svwhilelt_b16((uint64_t)0, (uint64_t)pred_bound);
                    // printf("PRED = %d | %d\n", 0, pred_bound);
                    // print_predicate_h(lane_pg);

                    in_vals = svld4_f16(lane_pg, &mat[((col_cnt*n_vect_elems) + n_idx_processed) * interl_factor_4D]);
                    in_0 = svget4_f16(in_vals, 0);
                    in_1 = svget4_f16(in_vals, 1);
                    in_2 = svget4_f16(in_vals, 2);
                    in_3 = svget4_f16(in_vals, 3);
                    
                    shamts = svindex_u16(0, BITS_PER_CB);

                    unpkd_idxs = svlsr_u16_z(svptrue_b16(), pakd_idxs, shamts);
                    unpkd_idxs = svand_u16_z(svptrue_b16(), base_masks, unpkd_idxs);
                    // print_vect_ui16(unpkd_idxs);

                    #if defined(N_SVE_REG_CB_F16_1)
                    weights_0 = svtbl_f16(cb0, unpkd_idxs);
                    weights_1 = svtbl_f16(cb1, unpkd_idxs);
                    weights_2 = svtbl_f16(cb2, unpkd_idxs);
                    weights_3 = svtbl_f16(cb3, unpkd_idxs);
                    
                    #elif defined(N_SVE_REG_CB_F16_2)
                    weights_0 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb0_2regs);
                    weights_1 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb1_2regs);
                    weights_2 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb2_2regs);
                    weights_3 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb3_2regs);
                    // exit(0);
                    #endif

                    mul_pg = lane_pg;

                    // if(missing_total < IDXS_PER_WORD_16){
                    //     // pred_bound*2 to cover the cases in which there are left less indexes than the one a half word can store
                    //     mul_pg = svwhilelt_b16((uint64_t)0, (uint64_t)(pred_bound*2));
                    // } else {
                    //     mul_pg = lane_pg;
                    // }

                    // printf("== MAC ==\n");
                    // print_predicate_h(mul_pg);
                    // print_vect_f16(in_0);
                    // print_vect_ui16(unpkd_idxs);
                    // print_vect_f16(weights_0);

                    // MAC
                    row_res_vect0 = svmad_f16_x(mul_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f16_x(mul_pg, in_1, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f16_x(mul_pg, in_2, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f16_x(mul_pg, in_3, weights_3, row_res_vect3);

                    // print_vect_f16(row_res_vect0);

                    n_idx_processed += svcntp_b16(mul_pg, mul_pg);


                    // exit(0);

                    #else

                    missing_lane = 0;
                    
                    missing_total = n_vect_elems - n_idx_processed;
                    // printf("Missing total = %d - %d = %d\n", n_vect_elems, n_idx_processed, missing_total);


                    // // Duplicates a single word of packed indexes in all the lanes
                    svuint32_t dup_idxs_32b = svdup_lane_u32(packed_idxs, lane);
                    dup_idxs_pakd = svreinterpret_u16_u32(dup_idxs_32b);    // reinterpret them as 16 bits words
                    // print_vect_ui16(dup_idxs_pakd);


                    int pred_bound = (missing_total < IDXS_PER_WORD) ? (missing_total) : (IDXS_PER_WORD);

                    svbool_t lane_pg = svwhilelt_b16((uint64_t)0, (uint64_t)pred_bound);
                    // printf("PRED = %d | %d\n", 0, pred_bound);
                    // print_predicate_h(lane_pg);

                    in_vals = svld4_f16(lane_pg, &mat[((col_cnt*n_vect_elems) + n_idx_processed) * interl_factor_4D]);
                    svfloat16_t in_0f = svget4_f16(in_vals, 0);
                    svfloat16_t in_1f = svget4_f16(in_vals, 1);
                    svfloat16_t in_2f = svget4_f16(in_vals, 2);
                    svfloat16_t in_3f = svget4_f16(in_vals, 3);

                    svfloat16_t f16_zero = svdup_f16(0.0);
                    
                    // Take only the second half of the vector register
                    svfloat16_t in_0s = svext_f16(in_0f, svdup_f16(0), IDXS_PER_WORD_16);
                    svfloat16_t in_1s = svext_f16(in_1f, svdup_f16(0), IDXS_PER_WORD_16);
                    svfloat16_t in_2s = svext_f16(in_2f, svdup_f16(0), IDXS_PER_WORD_16);
                    svfloat16_t in_3s = svext_f16(in_3f, svdup_f16(0), IDXS_PER_WORD_16);

                    // print_vect_f16(in_0f);
                    // print_vect_f16(in_0s);
                    
                    // rearragne them so they are interleaved
                    svfloat16_t in_0_rearranged = svzip1_f16(in_0f, in_0s);
                    svfloat16_t in_1_rearranged = svzip1_f16(in_1f, in_1s);
                    svfloat16_t in_2_rearranged = svzip1_f16(in_2f, in_2s);
                    svfloat16_t in_3_rearranged = svzip1_f16(in_3f, in_3s);
                    // print_vect_f16(in_0_rearranged);
                    // exit(0);

                    // shamts = svindex_u16((n_idx_processed*BITS_PER_CB), BITS_PER_CB);
                    shamts = svindex_u16(0, BITS_PER_CB);
                    shamts = svzip1_u16(shamts, shamts);
                    // print_vect_ui16(shamts);

                    unpkd_idxs = svlsr_u16_z(svptrue_b16(), dup_idxs_pakd, shamts);
                    // print_vect_ui16(unpkd_idxs);
                    unpkd_idxs = svand_u16_z(svptrue_b16(), base_masks, unpkd_idxs);
                    // print_vect_ui16(unpkd_idxs);


                    #if defined(N_SVE_REG_CB_F16_1)
                    weights_0 = svtbl_f16(cb0, unpkd_idxs);
                    weights_1 = svtbl_f16(cb1, unpkd_idxs);
                    weights_2 = svtbl_f16(cb2, unpkd_idxs);
                    weights_3 = svtbl_f16(cb3, unpkd_idxs);
                    
                    #elif defined(N_SVE_REG_CB_F16_2)
                    weights_0 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb0_2regs);
                    weights_1 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb1_2regs);
                    weights_2 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb2_2regs);
                    weights_3 = extract_weightsx2_f16(lane_pg, unpkd_idxs, cb3_2regs);
                    // exit(0);
                    #endif

                    // print_vect_f16(weights_0);

                    
                    // svbool_t mul_pg = svwhilelt_b16((uint64_t)0, (uint64_t)(pred_bound*2));  // pred_bound*2 to cover the cases in which there are left less indexes than the one a half word can store
                    // svbool_t mul_pg = svwhilelt_b16((uint64_t)0, (uint64_t)(pred_bound));
                    // svbool_t mul_pg = lane_pg;
                    svbool_t mul_pg;

                    if(missing_total < IDXS_PER_WORD_16){
                        // pred_bound*2 to cover the cases in which there are left less indexes than the one a half word can store
                        mul_pg = svwhilelt_b16((uint64_t)0, (uint64_t)(pred_bound*2));
                    } else {
                        mul_pg = lane_pg;
                    }

                    // printf("== MAC ==\n");
                    // print_predicate_h(mul_pg);
                    // print_vect_f16(in_0_rearranged);
                    // print_vect_ui16(unpkd_idxs);
                    // print_vect_f16(weights_0);

                    // MAC
                    row_res_vect0 = svmad_f16_x(mul_pg, in_0_rearranged, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f16_x(mul_pg, in_1_rearranged, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f16_x(mul_pg, in_2_rearranged, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f16_x(mul_pg, in_3_rearranged, weights_3, row_res_vect3);

                    // print_vect_f16(row_res_vect0);

                    n_idx_processed += IDXS_PER_WORD;
                    #endif
                }
            }
            // exit(0);
                    
            uint32_t base_index =   base_res_idx + 
                                    (rt * full_out_w) + 
                                    ((ct / (tile_l2_w/interl_factor_4D)) * full_out_w) +        // the division acts as a floor division to identify the line of the L2 tile 
                                    ((ct % (tile_l2_w/interl_factor_4D)) * interl_factor_4D);   // the modulo serves to have a continuous variable resetting at 0 every new line of the L2 tile
            
            // printf("BASE idx: %d + ( %d * %d ) + (%d / %d * %d) + (%d * %d) = %d\n", base_res_idx, rt, full_out_w, ct, tile_l2_w/interl_factor_4D, full_out_w, ct, interl_factor_4D, base_index);
            // printf("IDXS: %d %d %d %d\n", base_index, base_index+1, base_index+2, base_index+3);

            // print_vect_f32(row_res_vect1);
            // Compute the final output value by adding all the lanes
            // printf("Index: %d --> %f\n", base_index + 0, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect0));
            // printf("Index: %d --> %f\n", base_index + 1, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect1));
            // printf("Index: %d --> %f\n", base_index + 2, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect2));
            // printf("Index: %d --> %f\n", base_index + 3, svaddv_f32(svwhilelt_b32((uint64_t)0, svcntw()), row_res_vect3));
            // printf("\nBEFORE: %f\n", res->tensor[base_index + 0]);

            res->tensor[base_index + 0] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
            res->tensor[base_index + 1] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
            res->tensor[base_index + 2] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
            res->tensor[base_index + 3] += svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3);
            
            // printf("res[%d] = %f\n", base_index+0, res->tensor[base_index + 0]);
            // break;

            col_cnt++;
        }
    }

    // exit(0);
}





