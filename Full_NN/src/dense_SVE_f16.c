
#include <SVE_implementations.h>
#include <dense_SVE.h>



// void exec_sve_compact_interleaved_4D(dense_t dense_layer, const float *in_interl, const float *cb_interl, float *out0, float *out1, float *out2, float *out3){
void exec_sve_compact_SVE_f16(dense_t dense_layer, const float16_t *in, const float16_t *cb, const uint16_t *weight_indexes, const float16_t *bias, float16_t *out){

    // Number of words contained in a vector register
    // const uint64_t n_W_lanes = svcntw();

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect = svdup_n_f16(0.0f);
    
    int32_t missing_total = 0;       // How many indexes are missing to be processed
    int32_t missing_lane = 0;       // How many indexes are missing inside the lane
    uint32_t input_idx = 0;     // Index to the next input element to be loaded

    // svfloat32_t weights_vals = svdup_n_f32(0.0f);   // Keep the values of the weights from the codebooks
    svbool_t load_pg;           // Predicate for loading the indexes
    svuint16_t packed_idxs;     // Holds the words with the packed indexes
    uint8_t n_loaded_lanes;     // Counts the number of packed-indexes lanes that have been loaded 

    svbool_t bits_mask_pg;      // Predicate for producing the shifted masks to unpack the indexes
    svuint16_t shamts;          // Shift amounts for the masks
    svuint16_t masks;           // Holds the masks to unpack the indexes

    svuint16_t dup_idxs_pakd;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t unpkd_idxs;      // Holds the unpacked indexes (one per lane)
    svfloat16_t in_vals;        // Holds the input values

    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16_t codebook = svld1_f16(svwhilelt_b16(0, CB_SIZE), cb);
    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16_t cb0 = svld1_f16(svwhilelt_b16(0, CB_SIZE), cb);
    svfloat16_t cb1 = svld1_f16(svwhilelt_b16(0, CB_SIZE), &cb[4]);
    svfloat16x2_t codebook = svcreate2_f16(cb0, cb1);
    #endif


    // Loop thorugh the rows of weights
    for(int r=0; r<dense_layer.out_size; r++){
        row_res_vect = svdup_n_f16(0.0f);
    
        input_idx = 0;
        int cw = 0;

        // Loop thorugh the columns of weights
        // It indexes the column words (the words that contains the indexes per each column)
        for(cw=0; cw<dense_layer.n_words_row; cw+=N_SVE_LANES){
            // printf("\n\n---- CW %d ----\n", cw);

            load_pg = svwhilelt_b16(cw, dense_layer.n_words_row);

            // Load a 32-bits word with IDXS_PER_WORD packed indexes
            // packed_idxs = svld1_u32(load_pg, &weight_idx_compact[r][cw]);
            packed_idxs = svld1_u16(load_pg, &weight_indexes[(r*dense_layer.n_words_row) + cw]);

            // Counts how many lanes have been loaded
            n_loaded_lanes = svcntp_b16(load_pg, load_pg);


            // Loop through the 32-bits lanes of the vector register
            for (size_t lane = 0; lane < n_loaded_lanes; ++lane) {
                // printf("\n\n---- LANE %ld ----\n", lane);

                missing_lane = 0;

                // Duplicates a single word of packed indexes in all the lanes
                dup_idxs_pakd = svdup_lane_u16(packed_idxs, lane);

                for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                    // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                    
                    missing_lane = (IDXS_PER_WORD - idx_ptr);
                    missing_total = (dense_layer.in_size - input_idx);
                    // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                    
                    // Stop in case there are no more missing indexes to process
                    if (missing_total<=0){
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
                    masks = svlsl_u16_z(bits_mask_pg, svdup_u16(IDX_MASK), shamts);  // Shift amounts for the indexes
                    // print_vect_ui32(masks);

                    // Perform the MASK+SHIFT for the considered word of packed indexes  
                    unpkd_idxs = svand_u16_z(bits_mask_pg, masks, dup_idxs_pakd);
                    // print_vect_ui32(unpkd_idxs);
                    unpkd_idxs = svlsr_u16_z(bits_mask_pg, unpkd_idxs, shamts);
                    // print_vect_ui32(unpkd_idxs);

                    // Load the weights based on the unpacked indexes
                    // weights_vals = svld1_gather_u32index_f32(bits_mask_pg, cb_0, unpkd_idxs);
                    // svfloat32_t weights_0 = svtbl_f32(cb0, unpkd_idxs);
                    // svfloat32_t weights_1 = svtbl_f32(cb1, unpkd_idxs);
                    // svfloat32_t weights_2 = svtbl_f32(cb2, unpkd_idxs);
                    // svfloat32_t weights_3 = svtbl_f32(cb3, unpkd_idxs);

                    svfloat16_t weights;

                    #ifdef N_SVE_REG_CB_F16_1
                    weights = svtbl_f16(codebook, unpkd_idxs);
                    #elif defined(N_SVE_REG_CB_F16_2)
                    weights = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs, codebook);
                    #endif



                    // Load the input values based on the unpacked indexes
                    // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                    in_vals = svld1_f16(bits_mask_pg, &in[input_idx]);
                    // print_vect_f32(in_0);

                    input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

                    // MAC
                    // printf("Before:\n");
                    // print_vect_f32(row_res_vect0);
                    // printf("In:\n");
                    // print_vect_f32(in_0);
                    // printf("Weights:\n");
                    // print_vect_f32(weights_0);
                    row_res_vect = svmad_f16_x(bits_mask_pg, in_vals, weights, row_res_vect);
                }
            }
        }

        // Compute the final output value by adding all the lanes
        out[r] = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect);
        out[r] += bias[r];

        // break;
    }
}








// void exec_sve_compact_interleaved_4D(dense_t dense_layer, const float *in_interl, const float *cb_interl, float *out0, float *out1, float *out2, float *out3){
void exec_sve_compact_interleaved_4D_diff_seq_f16(dense_t dense_layer, const float16_t *in_interl, const float16_t *cb_interl, const uint16_t *weight_indexes_interl, const float16_t *bias, float16_t *out_interl){

    uint8_t interl_4D_factor = 4;

    // Number of words contained in a vector register
    // const uint64_t n_W_lanes = svcntw();

    // Vect register to store the SIMD intermediate results of a row
    svfloat16_t row_res_vect0 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect1 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect2 = svdup_n_f16(0.0f);
    svfloat16_t row_res_vect3 = svdup_n_f16(0.0f);
    
    int32_t missing_total = 0;       // How many indexes are missing to be processed
    int32_t missing_lane = 0;       // How many indexes are missing inside the lane
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
    svuint16_t masks;           // Holds the masks to unpack the indexes

    svuint16_t dup_idxs_pakd_0;   // Holds duplicated instances of a packed-indexes word (to be masked with different masks)
    svuint16_t dup_idxs_pakd_1;
    svuint16_t dup_idxs_pakd_2;
    svuint16_t dup_idxs_pakd_3;
    svuint16_t unpkd_idxs_0;      // Holds the unpacked indexes (one per lane)
    svuint16_t unpkd_idxs_1;
    svuint16_t unpkd_idxs_2;
    svuint16_t unpkd_idxs_3;
    svfloat16x4_t in_vals;        // Holds the input values


    #if defined(N_SVE_REG_CB_F16_1)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), cb_interl);
    svfloat16_t cb0 = svget4_f16(codebooks_loaded, 0);
    svfloat16_t cb1 = svget4_f16(codebooks_loaded, 1);
    svfloat16_t cb2 = svget4_f16(codebooks_loaded, 2);
    svfloat16_t cb3 = svget4_f16(codebooks_loaded, 3);

    #elif defined(N_SVE_REG_CB_F16_2)
    svfloat16x4_t codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &cb_interl[0*(N_SVE_LANES*8)]);
    svfloat16x2_t cb0_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 0), svdup_n_f16(0.0));
    svfloat16x2_t cb1_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 1), svdup_n_f16(0.0));
    svfloat16x2_t cb2_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 2), svdup_n_f16(0.0));
    svfloat16x2_t cb3_2regs = svcreate2_f16(svget4_f16(codebooks_loaded, 3), svdup_n_f16(0.0));

    codebooks_loaded = svld4_f16(svwhilelt_b16(0, CB_SIZE), &cb_interl[1*(N_SVE_LANES*8)]);
    cb0_2regs = svcreate2_f16(svget2_f16(cb0_2regs, 0), svget4_f16(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f16(svget2_f16(cb1_2regs, 0), svget4_f16(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f16(svget2_f16(cb2_2regs, 0), svget4_f16(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f16(svget2_f16(cb3_2regs, 0), svget4_f16(codebooks_loaded, 3));
    #endif



    // Loop thorugh the rows of weights
    for(int r=0; r<dense_layer.out_size; r++){
        row_res_vect0 = svdup_n_f16(0.0f);
        row_res_vect1 = svdup_n_f16(0.0f);
        row_res_vect2 = svdup_n_f16(0.0f);
        row_res_vect3 = svdup_n_f16(0.0f);
    
        input_idx = 0;
        int cw = 0;

        // Loop thorugh the columns of weights
        // It indexes the column words (the words that contains the indexes per each column)
        for(cw=0; cw<dense_layer.n_words_row; cw+=N_SVE_LANES){
            // printf("\n\n---- CW %d ----\n", cw);
            // print_vect_f32(row_res_vect0);

            load_pg = svwhilelt_b16(cw, dense_layer.n_words_row);

            // Load a 32-bits word with IDXS_PER_WORD packed indexes
            // packed_idxs = svld1_u32(load_pg, &weight_idx_compact[r][cw]);
            // packed_idxs = svld1_u32(load_pg, &weight_indexes[(r*dense_layer.n_words_row) + cw]);
            svuint16x4_t packed_idxs = svld4_u16(load_pg, &weight_indexes_interl[((r*dense_layer.n_words_row) + cw) * interl_4D_factor]);
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
                // printf("\n\n---- LANE %ld ----\n", lane);

                missing_lane = 0;

                // Duplicates a single word of packed indexes in all the lanes
                dup_idxs_pakd_0 = svdup_lane_u16(packed_idxs_0, lane);
                dup_idxs_pakd_1 = svdup_lane_u16(packed_idxs_1, lane);
                dup_idxs_pakd_2 = svdup_lane_u16(packed_idxs_2, lane);
                dup_idxs_pakd_3 = svdup_lane_u16(packed_idxs_3, lane);

                for (size_t idx_ptr=0; idx_ptr<IDXS_PER_WORD; idx_ptr+=N_SVE_LANES){
                    // printf("---- IXD P. %ld | IN P. %d ----\n", idx_ptr, input_idx);
                    
                    missing_lane = (IDXS_PER_WORD - idx_ptr);
                    missing_total = (dense_layer.in_size - input_idx);
                    // printf("Missing : %d (tot) - %d (lane)\n", missing_total, missing_lane);
                    
                    // Stop in case there are no more missing indexes to process
                    if (missing_total<=0){
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
                    masks = svlsl_u16_z(bits_mask_pg, svdup_u16(IDX_MASK), shamts);  // Shift amounts for the indexes
                    // print_vect_ui32(masks);

                    // Perform the MASK+SHIFT for the considered word of packed indexes  
                    unpkd_idxs_0 = svand_u16_z(bits_mask_pg, masks, dup_idxs_pakd_0);
                    unpkd_idxs_1 = svand_u16_z(bits_mask_pg, masks, dup_idxs_pakd_1);
                    unpkd_idxs_2 = svand_u16_z(bits_mask_pg, masks, dup_idxs_pakd_2);
                    unpkd_idxs_3 = svand_u16_z(bits_mask_pg, masks, dup_idxs_pakd_3);

                    unpkd_idxs_0 = svlsr_u16_z(bits_mask_pg, unpkd_idxs_0, shamts);
                    unpkd_idxs_1 = svlsr_u16_z(bits_mask_pg, unpkd_idxs_1, shamts);
                    unpkd_idxs_2 = svlsr_u16_z(bits_mask_pg, unpkd_idxs_2, shamts);
                    unpkd_idxs_3 = svlsr_u16_z(bits_mask_pg, unpkd_idxs_3, shamts);
                    // print_vect_ui32(unpkd_idxs_0);

                    svfloat16_t weights_0;
                    svfloat16_t weights_1;
                    svfloat16_t weights_2;
                    svfloat16_t weights_3;

                    
                    #ifdef N_SVE_REG_CB_F16_1
                    weights_0 = svtbl_f16(cb0, unpkd_idxs_0);
                    weights_1 = svtbl_f16(cb1, unpkd_idxs_1);
                    weights_2 = svtbl_f16(cb2, unpkd_idxs_2);
                    weights_3 = svtbl_f16(cb3, unpkd_idxs_3);
                    
                    #elif defined(N_SVE_REG_CB_F16_2)
                    weights_0 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_0, cb0_2regs);
                    weights_1 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_1, cb1_2regs);
                    weights_2 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_2, cb2_2regs);
                    weights_3 = extract_weightsx2_f16(bits_mask_pg, unpkd_idxs_3, cb3_2regs);
                    #else 
                        printf("ERROR: No suitable `N_SVE_REG_CB_x` defined...");
                        exit(1);
                    #endif


                    // Load the input values based on the unpacked indexes
                    // in_vals = svld1_f32(bits_mask_pg, &in[input_idx]);
                    in_vals = svld4_f16(bits_mask_pg, &in_interl[input_idx*interl_4D_factor]);
                    svfloat16_t in_0 = svget4_f16(in_vals, 0);
                    svfloat16_t in_1 = svget4_f16(in_vals, 1);
                    svfloat16_t in_2 = svget4_f16(in_vals, 2);
                    svfloat16_t in_3 = svget4_f16(in_vals, 3);
                    // print_vect_f32(in_0);

                    input_idx += svcntp_b16(bits_mask_pg, bits_mask_pg);

                    // MAC
                    row_res_vect0 = svmad_f16_x(bits_mask_pg, in_0, weights_0, row_res_vect0);
                    row_res_vect1 = svmad_f16_x(bits_mask_pg, in_1, weights_1, row_res_vect1);
                    row_res_vect2 = svmad_f16_x(bits_mask_pg, in_2, weights_2, row_res_vect2);
                    row_res_vect3 = svmad_f16_x(bits_mask_pg, in_3, weights_3, row_res_vect3);
                }
            }
        }

        // Compute the final output value by adding all the lanes
        out_interl[(r*interl_4D_factor) + 0] = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect0);
        out_interl[(r*interl_4D_factor) + 1] = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect1);
        out_interl[(r*interl_4D_factor) + 2] = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect2);
        out_interl[(r*interl_4D_factor) + 3] = svaddv_f16(svwhilelt_b16((uint64_t)0, svcnth()), row_res_vect3);


        out_interl[(r*interl_4D_factor) + 0] += bias[(r*interl_4D_factor) + 0];
        out_interl[(r*interl_4D_factor) + 1] += bias[(r*interl_4D_factor) + 1];
        out_interl[(r*interl_4D_factor) + 2] += bias[(r*interl_4D_factor) + 2];
        out_interl[(r*interl_4D_factor) + 3] += bias[(r*interl_4D_factor) + 3];

    }
}

    