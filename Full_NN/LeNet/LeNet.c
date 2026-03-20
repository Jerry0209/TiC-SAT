#include <stdio.h>
#include <string.h>

#include <main.h>
#include <print_functs.h>

#include <SVE_implementations.h>
#include <dense_SVE.h>

#include <lenet_def.h>


#include <conv_exec.h>
#include <pooling.h>
#include <relu.h>



void exec_LeNet5_learner_by_learner(){

        for(int ens=0; ens<N_LEARNERS; ens++){
        // printf("\n================= LEARNER %d =================\n", ens);
        
        
        // Allocate an array of output tensors
        tensor3D_t out_tensor = {
            .dim = output_dims_0,
            .tensor = (float*)malloc((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float))
        };

        tensor3D_t *padded_input = check_padding(&input_0, conv_0.padding);

        // printf("INP %d %d %d\n", padded_input->dim.depth, padded_input->dim.height, padded_input->dim.width);

        // Conv 0
        conv3D_staticPatch_compact(conv_0, padded_input, kernel_compact_perCH_0, codebook_0[ens], &out_tensor); 

        // printf("CONV 0 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);
        
        free(padded_input->tensor);
        free(padded_input);  

        // print_volume(out_tensor.tensor, &out_tensor.dim);

        relu(out_tensor.tensor, (output_dims_0.depth * output_dims_0.height * output_dims_0.width));



        // Max pooling 1
        maxPool(&out_tensor, (out_tensor.dim.height * out_tensor.dim.width), POOL_SIZE_1, POOL_STRIDE_1, &out_tensor);
        // printf("MaxPool 1 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        


        tensor3D_t out_tensor_2 = conv_layer(conv_2, &out_tensor, kernel_compact_perCH_2, (const float*)&codebook_2[ens]);

        // printf("CONV 2 %d %d %d\n", out_tensor_2.dim.depth, out_tensor_2.dim.height, out_tensor_2.dim.width);

        relu(out_tensor_2.tensor, (out_tensor_2.dim.depth * out_tensor_2.dim.height * out_tensor_2.dim.width));



        // Max pooling 3
        maxPool(&out_tensor_2, (out_tensor_2.dim.height * out_tensor_2.dim.width), POOL_SIZE_3, POOL_STRIDE_3, &out_tensor_2);

        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);



        // Dense 4
        float *out4 = (float*)malloc(dense_4.out_size * sizeof(float));
        exec_compact(dense_4, out_tensor_2.tensor, weight_idx_compact_4, codebooks_4[ens], out4);
        relu(out4, dense_4.out_size);
        free(out_tensor_2.tensor);



        // Dense 5
        float *out5 = (float*)malloc(dense_5.out_size * sizeof(float));
        exec_compact(dense_5, out4, weight_idx_compact_5, codebooks_5[ens], out5);
        relu(out5,   dense_5.out_size);
        free(out4);


        // Dense 6
        float *out6 = (float*)malloc(dense_6.out_size * sizeof(float));
        exec_compact(dense_6, out5, weight_idx_compact_6, codebooks_6[ens], out6);

    #ifdef PRINT_EN
        printf("\nOut 6\n");
        printf("\n---- Learner %d ----\n", ens);
        for(int i=0; i<dense_6.out_size; i++){
            printf("[%d] %f\n", i, out6[i]);
        }
    #endif

        free(out5);
        free(out6);

    }
}





void exec_LeNet5_learner_by_learner_tiled_l2(){

    for(int ens=0; ens<N_LEARNERS; ens++){
        // printf("\n================= LEARNER %d =================\n", ens);
        
        
        // Allocate an array of output tensors
        tensor3D_t out_tensor = {
            .dim = output_dims_0,
            .tensor = (float*)malloc((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float))
        };
        memset(out_tensor.tensor, 0.0, ((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float)));

        tensor3D_t *padded_input = check_padding(&input_0, conv_0.padding);

        // printf("INP %d %d %d\n", padded_input->dim.depth, padded_input->dim.height, padded_input->dim.width);

        // Conv 0
        // conv3D_staticPatch_compact_tiled(conv_0, padded_input, kernel_compact_perCH_0, codebook_0[ens], TILE_L2_SIZE, &out_tensor); 
        conv3D_staticPatch_compact_tiled(conv_0, padded_input, kernel_compact_perCH_0, codebook_0[ens], TILE_L2_SIZE, &out_tensor);

        // printf("CONV 0 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);
        
        free(padded_input->tensor);
        free(padded_input);  

        // print_volume(out_tensor.tensor, &out_tensor.dim);

        relu(out_tensor.tensor, (output_dims_0.depth * output_dims_0.height * output_dims_0.width));



        // Max pooling 1
        maxPool(&out_tensor, (out_tensor.dim.height * out_tensor.dim.width), POOL_SIZE_1, POOL_STRIDE_1, &out_tensor);
        // printf("MaxPool 1 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);

        tensor3D_t out_tensor_2 = conv_layer_tiled_l2(conv_2, &out_tensor, kernel_compact_perCH_2, TILE_L2_SIZE, (const float*)&codebook_2[ens]);

        // printf("CONV 2 %d %d %d\n", out_tensor_2.dim.depth, out_tensor_2.dim.height, out_tensor_2.dim.width);
        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);
        // exit(0);

        relu(out_tensor_2.tensor, (out_tensor_2.dim.depth * out_tensor_2.dim.height * out_tensor_2.dim.width));



        // Max pooling 3
        maxPool(&out_tensor_2, (out_tensor_2.dim.height * out_tensor_2.dim.width), POOL_SIZE_3, POOL_STRIDE_3, &out_tensor_2);

        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);



        // Dense 4
        float *out4 = (float*)malloc(dense_4.out_size * sizeof(float));
        memset(out4, 0.0, dense_4.out_size*sizeof(float));
        exec_compact(dense_4, out_tensor_2.tensor, weight_idx_compact_4, codebooks_4[ens], out4);
        relu(out4, dense_4.out_size);
        free(out_tensor_2.tensor);



        // Dense 5
        float *out5 = (float*)malloc(dense_5.out_size * sizeof(float));
        memset(out5, 0.0, dense_5.out_size*sizeof(float));
        exec_compact(dense_5, out4, weight_idx_compact_5, codebooks_5[ens], out5);
        relu(out5,   dense_5.out_size);
        free(out4);


        // Dense 6
        float *out6 = (float*)malloc(dense_6.out_size * sizeof(float));
        memset(out6, 0.0, dense_6.out_size*sizeof(float));
        exec_compact(dense_6, out5, weight_idx_compact_6, codebooks_6[ens], out6);

    #ifdef PRINT_EN
        printf("\nOut 6\n");
        printf("\n---- Learner %d ----\n", ens);
        for(int i=0; i<dense_6.out_size; i++){
            printf("[%d] %f\n", i, out6[i]);
        }
    #endif

        free(out5);
        free(out6);

    }
}





void exec_LeNet5_learner_by_learner_tiled_l2_diff_seq(){

    for(int ens=0; ens<N_LEARNERS; ens++){
        // printf("\n================= LEARNER %d =================\n", ens);
        
        
        // Allocate an array of output tensors
        tensor3D_t out_tensor = {
            .dim = output_dims_0,
            .tensor = (float*)malloc((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float))
        };
        memset(out_tensor.tensor, 0.0, ((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float)));

        tensor3D_t *padded_input = check_padding(&input_0, conv_0.padding);

        // printf("INP %d %d %d\n", padded_input->dim.depth, padded_input->dim.height, padded_input->dim.width);

        // Conv 0
        // conv3D_staticPatch_compact_tiled(conv_0, padded_input, kernel_compact_perCH_0, codebook_0[ens], TILE_L2_SIZE, &out_tensor); 
        conv3D_staticPatch_compact_tiled(conv_0, padded_input, &kernel_compact_perCH_0[ens], codebook_0[ens], TILE_L2_SIZE, &out_tensor);

        // printf("CONV 0 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);
        
        free(padded_input->tensor);
        free(padded_input);  

        // print_volume(out_tensor.tensor, &out_tensor.dim);

        relu(out_tensor.tensor, (output_dims_0.depth * output_dims_0.height * output_dims_0.width));



        // Max pooling 1
        maxPool(&out_tensor, (out_tensor.dim.height * out_tensor.dim.width), POOL_SIZE_1, POOL_STRIDE_1, &out_tensor);
        // printf("MaxPool 1 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);

        tensor3D_t out_tensor_2 = conv_layer_tiled_l2(conv_2, &out_tensor, &kernel_compact_perCH_2[ens], TILE_L2_SIZE, (const float*)&codebook_2[ens]);

        // printf("CONV 2 %d %d %d\n", out_tensor_2.dim.depth, out_tensor_2.dim.height, out_tensor_2.dim.width);
        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);
        // exit(0);

        relu(out_tensor_2.tensor, (out_tensor_2.dim.depth * out_tensor_2.dim.height * out_tensor_2.dim.width));



        // Max pooling 3
        maxPool(&out_tensor_2, (out_tensor_2.dim.height * out_tensor_2.dim.width), POOL_SIZE_3, POOL_STRIDE_3, &out_tensor_2);

        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);
        // break;


        // Dense 4
        float *out4 = (float*)malloc(dense_4.out_size * sizeof(float));
        memset(out4, 0.0, dense_4.out_size*sizeof(float));
        exec_compact(dense_4, out_tensor_2.tensor, &weight_idx_compact_4[ens], codebooks_4[ens], out4);
        relu(out4, dense_4.out_size);
        free(out_tensor_2.tensor);

        // for(int i=0; i<dense_4.out_size; i++){
        //     printf("%f\n", out4[i]);
        // }
        // break;


        // Dense 5
        float *out5 = (float*)malloc(dense_5.out_size * sizeof(float));
        memset(out5, 0.0, dense_5.out_size*sizeof(float));
        exec_compact(dense_5, out4, &weight_idx_compact_5[ens], codebooks_5[ens], out5);
        relu(out5,   dense_5.out_size);
        free(out4);

        // for(int i=0; i<dense_5.out_size; i++){
        //     printf("%f\n", out5[i]);
        // }
        // break;

        // Dense 6
        float *out6 = (float*)malloc(dense_6.out_size * sizeof(float));
        memset(out6, 0.0, dense_6.out_size*sizeof(float));
        exec_compact(dense_6, out5, &weight_idx_compact_6[ens], codebooks_6[ens], out6);

    #ifdef PRINT_EN
        printf("\nOut 6\n");
        printf("\n---- Learner %d ----\n", ens);
        for(int i=0; i<dense_6.out_size; i++){
            printf("[%d] %f\n", i, out6[i]);
        }
    #endif

        free(out5);
        free(out6);

    }
}





void exec_LeNet5_learner_by_learner_tiled_l2l1(){

    for(int ens=0; ens<N_LEARNERS; ens++){
        // printf("\n================= LEARNER %d =================\n", ens);
        
        
        // Allocate an array of output tensors
        tensor3D_t out_tensor = {
            .dim = output_dims_0,
            .tensor = (float*)malloc((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float))
        };
        memset(out_tensor.tensor, 0.0, ((output_dims_0.depth * output_dims_0.height * output_dims_0.width) * sizeof(float)));

        tensor3D_t *padded_input = check_padding(&input_0, conv_0.padding);

        // printf("INP %d %d %d\n", padded_input->dim.depth, padded_input->dim.height, padded_input->dim.width);

        // Conv 0
        // conv3D_staticPatch_compact_tiled(conv_0, padded_input, kernel_compact_perCH_0, codebook_0[ens], TILE_L2_SIZE, &out_tensor); 
        conv3D_staticPatch_compact_tiled_L1L2(conv_0, padded_input, kernel_compact_tiled_0, codebook_0[ens], TILE_L2_SIZE, TILE_L1_SIZE, &out_tensor);

        // printf("CONV 0 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);
        
        free(padded_input->tensor);
        free(padded_input);  

        // print_volume(out_tensor.tensor, &out_tensor.dim);

        relu(out_tensor.tensor, (output_dims_0.depth * output_dims_0.height * output_dims_0.width));



        // Max pooling 1
        maxPool(&out_tensor, (out_tensor.dim.height * out_tensor.dim.width), POOL_SIZE_1, POOL_STRIDE_1, &out_tensor);
        // printf("MaxPool 1 %d %d %d\n", out_tensor.dim.depth, out_tensor.dim.height, out_tensor.dim.width);
        // print_volume(out_tensor.tensor, &out_tensor.dim);
        // exit(0);

        tensor3D_t out_tensor_2 = conv_layer_tiled_l2l1(conv_2, &out_tensor, kernel_compact_tiled_2, TILE_L2_SIZE, TILE_L1_SIZE, (const float*)&codebook_2[ens]);

        // printf("CONV 2 %d %d %d\n", out_tensor_2.dim.depth, out_tensor_2.dim.height, out_tensor_2.dim.width);

        relu(out_tensor_2.tensor, (out_tensor_2.dim.depth * out_tensor_2.dim.height * out_tensor_2.dim.width));



        // Max pooling 3
        maxPool(&out_tensor_2, (out_tensor_2.dim.height * out_tensor_2.dim.width), POOL_SIZE_3, POOL_STRIDE_3, &out_tensor_2);

        // print_volume(out_tensor_2.tensor, &out_tensor_2.dim);



        // Dense 4
        float *out4 = (float*)malloc(dense_4.out_size * sizeof(float));
        memset(out4, 0.0, dense_4.out_size*sizeof(float));
        exec_compact(dense_4, out_tensor_2.tensor, weight_idx_compact_4, codebooks_4[ens], out4);
        relu(out4, dense_4.out_size);
        free(out_tensor_2.tensor);



        // Dense 5
        float *out5 = (float*)malloc(dense_5.out_size * sizeof(float));
        memset(out5, 0.0, dense_5.out_size*sizeof(float));
        exec_compact(dense_5, out4, weight_idx_compact_5, codebooks_5[ens], out5);
        relu(out5,   dense_5.out_size);
        free(out4);


        // Dense 6
        float *out6 = (float*)malloc(dense_6.out_size * sizeof(float));
        memset(out6, 0.0, dense_6.out_size*sizeof(float));
        exec_compact(dense_6, out5, weight_idx_compact_6, codebooks_6[ens], out6);

    #ifdef PRINT_EN
        printf("\nOut 6\n");
        printf("\n---- Learner %d ----\n", ens);
        for(int i=0; i<dense_6.out_size; i++){
            printf("[%d] %f\n", i, out6[i]);
        }
    #endif

        free(out5);
        free(out6);

    }
}



void exec_LeNet5_interleaved_2D(){

    // Dimension of the interleved inputs
    dim3D_t interl_dims = {
        .depth = input_0.dim.depth,
        .height = input_0.dim.height,
        .width = input_0.dim.width * N_LEARNERS,
    };

    // Allocate space for the interleaved 4D input
    float *interl_in = (float*)malloc(interl_dims.depth * interl_dims.height * interl_dims.width * sizeof(float));


    merge_interleaved_2D(   input_0.tensor, input_0.tensor, 
                            (input_0.dim.depth * input_0.dim.height * input_0.dim.width),
                            interl_in);

    // Instantiate a tensor structure for the interleaved inputs
    tensor3D_t interl_in_tensor = {
        .dim = interl_dims,
        .tensor = interl_in
    };

    // Conv 0 //
    tensor3D_t output_tensor_interl_2D_0 = conv_layer_interl(conv_0, &interl_in_tensor, kernel_compact_perCH_0, codebook_interleaved_0, 2);

    // ReLu //
    relu(output_tensor_interl_2D_0.tensor, (output_tensor_interl_2D_0.dim.depth * output_tensor_interl_2D_0.dim.height * output_tensor_interl_2D_0.dim.width));

    // Max Pool 1 //
    maxPool_interleavedND(&output_tensor_interl_2D_0, POOL_SIZE_1, POOL_STRIDE_1, 2, &output_tensor_interl_2D_0);


    // Conv 2 //
    tensor3D_t output_tensor_interl_2D_2 = conv_layer_interl(conv_2, &output_tensor_interl_2D_0, kernel_compact_perCH_2, codebook_interleaved_2, 2);

    // print_interleaved_out(&output_tensor_interl_2D_2, 2);
    // exit(0);
    // ReLu //
    relu(output_tensor_interl_2D_2.tensor, (output_tensor_interl_2D_2.dim.depth * output_tensor_interl_2D_2.dim.height * output_tensor_interl_2D_2.dim.width));

    // Max Pool 3 //
    maxPool_interleavedND(&output_tensor_interl_2D_2, POOL_SIZE_3, POOL_STRIDE_3, 2, &output_tensor_interl_2D_2);

    // print_interleaved_out(&output_tensor_interl_2D_2, 2);
    // exit(0);

    // Dense 4 //
    float *out4_interl = (float*)malloc(dense_4.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_2D(dense_4, output_tensor_interl_2D_2.tensor, codebook_interleaved_4, weight_idx_compact_4, out4_interl);
    relu(out4_interl, dense_4.out_size * N_LEARNERS);
    free(output_tensor_interl_2D_2.tensor);


    // Dense 5 //
    float *out5_interl = (float*)malloc(dense_5.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_2D(dense_5, out4_interl, codebook_interleaved_5, weight_idx_compact_5, out5_interl);
    relu(out5_interl, dense_5.out_size * N_LEARNERS);
    free(out4_interl);


    // Dense 6 //
    float *out6_interl = (float*)malloc(dense_6.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_2D(dense_6, out5_interl, codebook_interleaved_6, weight_idx_compact_6, out6_interl);
    free(out5_interl);

    #ifdef PRINT_EN
    print_flattened_interleaved(out6_interl, dense_6.out_size, N_LEARNERS);
    #endif

    free(out6_interl);
}






void exec_LeNet5_interleaved_4D(){

    // Dimension of the interleved inputs
    dim3D_t interl_dims = {
        .depth = input_0.dim.depth,
        .height = input_0.dim.height,
        .width = input_0.dim.width * N_LEARNERS,
    };

    // Allocate space for the interleaved padded inputs (flattened and all learners together)
    float *interleaved_padded_input = (float*)malloc(interl_dims.depth * interl_dims.height * interl_dims.width * sizeof(float));
    
    // Merge the padded inputs in interleaved fashion
    merge_interleaved_4D(   input_0.tensor, 
                            input_0.tensor,
                            input_0.tensor, 
                            input_0.tensor,
                            (input_0.dim.depth * input_0.dim.height * input_0.dim.width),
                            interleaved_padded_input);

    // Instantiate a tensor structure for the interleaved inputs
    tensor3D_t interl_in_tensor = {
        .dim = interl_dims,
        .tensor = interleaved_padded_input
    };

    tensor3D_t output_tensors_interl_4D_0 = conv_layer_interl(conv_0, &interl_in_tensor, kernel_compact_perCH_0, codebook_interleaved_0, 4);


    // ReLu //
    relu(output_tensors_interl_4D_0.tensor, (output_tensors_interl_4D_0.dim.depth * output_tensors_interl_4D_0.dim.height * output_tensors_interl_4D_0.dim.width));

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_0, POOL_SIZE_1, POOL_STRIDE_1, 4,  &output_tensors_interl_4D_0);

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);


    // Conv 2 //
    tensor3D_t output_tensors_interl_4D_2 = conv_layer_interl(conv_2, &output_tensors_interl_4D_0, kernel_compact_perCH_2, codebook_interleaved_2, 4);

    // ReLu //
    relu(output_tensors_interl_4D_2.tensor, (output_tensors_interl_4D_2.dim.depth * output_tensors_interl_4D_2.dim.height * output_tensors_interl_4D_2.dim.width));

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_2, POOL_SIZE_3, POOL_STRIDE_3, 4, &output_tensors_interl_4D_2);

    // print_interleaved_out(&output_tensors_interl_4D_2, 4);


    // Dense 4 //
    float *out_4_interl = (float*)malloc(dense_4.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_4, output_tensors_interl_4D_2.tensor, codebook_interleaved_4, weight_idx_compact_4, out_4_interl);
    relu(out_4_interl, dense_4.out_size * N_LEARNERS);
    free(output_tensors_interl_4D_2.tensor);

    // print_flattened_interleaved(out_4_interl, dense_4.out_size, N_LEARNERS);


    // Dense 5 //
    float *out_5_interl = (float*)malloc(dense_5.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_5, out_4_interl, codebook_interleaved_5, weight_idx_compact_5, out_5_interl);
    relu(out_5_interl, dense_5.out_size * N_LEARNERS);
    free(out_4_interl);
    // print_flattened_interleaved(out_5_interl, dense_5.out_size, N_LEARNERS);


    // Dense 6 //
    float *out_6_interl = (float*)malloc(dense_6.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_6, out_5_interl, codebook_interleaved_6, weight_idx_compact_6, out_6_interl);
    free(out_5_interl);


#ifdef PRINT_EN
    print_flattened_interleaved(out_6_interl, dense_6.out_size, N_LEARNERS);
#endif

    free(out_6_interl);

    // conv3D_staticPatch_compactSVE_interleaved4D(conv_2, output_tensors);

    // for(int ens=0; ens<N_LEARNERS; ens++){
    //     free(padded_inputs_arrs[ens]->tensor);
    // }

}




void exec_LeNet5_interleaved_4D_tiled_l2l1(){

    // Dimension of the interleved inputs
    dim3D_t interl_dims = {
        .depth = input_0.dim.depth,
        .height = input_0.dim.height,
        .width = input_0.dim.width * N_LEARNERS,
    };

    // Allocate space for the interleaved padded inputs (flattened and all learners together)
    float *interleaved_padded_input = (float*)malloc(interl_dims.depth * interl_dims.height * interl_dims.width * sizeof(float));
    
    // Merge the padded inputs in interleaved fashion
    merge_interleaved_4D(   input_0.tensor, 
                            input_0.tensor,
                            input_0.tensor, 
                            input_0.tensor,
                            (input_0.dim.depth * input_0.dim.height * input_0.dim.width),
                            interleaved_padded_input);

    // Instantiate a tensor structure for the interleaved inputs
    tensor3D_t interl_in_tensor = {
        .dim = interl_dims,
        .tensor = interleaved_padded_input
    };

    // tensor3D_t output_tensors_interl_4D_0 = conv_layer_interl_tiled(conv_0, &interl_in_tensor, kernel_compact_perCH_0, codebook_interleaved_0, TILE_L2_SIZE, 4);
    tensor3D_t output_tensors_interl_4D_0 = conv_layer_interl_tiled_l2l1(conv_0, &interl_in_tensor, kernel_compact_tiled_0, codebook_interleaved_0, TILE_L2_SIZE, TILE_L1_SIZE, 4);
    
    // print_interleaved_out(&output_tensors_interl_4D_0, 4);
    // exit(0);

    // ReLu //
    relu(output_tensors_interl_4D_0.tensor, (output_tensors_interl_4D_0.dim.depth * output_tensors_interl_4D_0.dim.height * output_tensors_interl_4D_0.dim.width));

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_0, POOL_SIZE_1, POOL_STRIDE_1, 4,  &output_tensors_interl_4D_0);

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);
    // exit(0);


    // Conv 2 //
    // tensor3D_t output_tensors_interl_4D_2 = conv_layer_interl_tiled(conv_2, &output_tensors_interl_4D_0, kernel_compact_perCH_2, codebook_interleaved_2, TILE_L2_SIZE, 4);
    tensor3D_t output_tensors_interl_4D_2 = conv_layer_interl_tiled_l2l1(conv_2, &output_tensors_interl_4D_0, kernel_compact_tiled_2, codebook_interleaved_2, TILE_L2_SIZE, TILE_L1_SIZE, 4);

    // print_interleaved_out(&output_tensors_interl_4D_2, 4);
    // exit(0);

    // ReLu //
    relu(output_tensors_interl_4D_2.tensor, (output_tensors_interl_4D_2.dim.depth * output_tensors_interl_4D_2.dim.height * output_tensors_interl_4D_2.dim.width));

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_2, POOL_SIZE_3, POOL_STRIDE_3, 4, &output_tensors_interl_4D_2);

    // print_interleaved_out(&output_tensors_interl_4D_2, 4);


    // Dense 4 //
    float *out_4_interl = (float*)malloc(dense_4.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_4, output_tensors_interl_4D_2.tensor, codebook_interleaved_4, weight_idx_compact_4, out_4_interl);
    relu(out_4_interl, dense_4.out_size * N_LEARNERS);
    free(output_tensors_interl_4D_2.tensor);

    // print_flattened_interleaved(out_4_interl, dense_4.out_size, N_LEARNERS);


    // Dense 5 //
    float *out_5_interl = (float*)malloc(dense_5.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_5, out_4_interl, codebook_interleaved_5, weight_idx_compact_5, out_5_interl);
    relu(out_5_interl, dense_5.out_size * N_LEARNERS);
    free(out_4_interl);
    // print_flattened_interleaved(out_5_interl, dense_5.out_size, N_LEARNERS);


    // Dense 6 //
    float *out_6_interl = (float*)malloc(dense_6.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D(dense_6, out_5_interl, codebook_interleaved_6, weight_idx_compact_6, out_6_interl);
    free(out_5_interl);


#ifdef PRINT_EN
    print_flattened_interleaved(out_6_interl, dense_6.out_size, N_LEARNERS);
#endif

    free(out_6_interl);

    // conv3D_staticPatch_compactSVE_interleaved4D(conv_2, output_tensors);

    // for(int ens=0; ens<N_LEARNERS; ens++){
    //     free(padded_inputs_arrs[ens]->tensor);
    // }

}





void exec_LeNet5_interleaved_4D_tiled_f16(){

    // Dimension of the interleved inputs
    dim3D_t interl_dims = {
        .depth = input_0.dim.depth,
        .height = input_0.dim.height,
        .width = input_0.dim.width * N_LEARNERS,
    };

    // Allocate space for the interleaved padded inputs (flattened and all learners together)
    float *interleaved_padded_input = (float*)malloc(interl_dims.depth * interl_dims.height * interl_dims.width * sizeof(float));
    
    // Merge the padded inputs in interleaved fashion
    merge_interleaved_4D(   input_0.tensor, 
                            input_0.tensor,
                            input_0.tensor, 
                            input_0.tensor,
                            (input_0.dim.depth * input_0.dim.height * input_0.dim.width),
                            interleaved_padded_input);

    // Instantiate a tensor structure for the interleaved inputs
    tensor3D_t interl_in_tensor = {
        .dim = interl_dims,
        .tensor = interleaved_padded_input
    };

    // tensor3D_t output_tensors_interl_4D_0 = conv_layer_interl_tiled(conv_0, &interl_in_tensor, kernel_compact_perCH_0, codebook_interleaved_0, TILE_L2_SIZE, 4);
    tensor3D_t output_tensors_interl_4D_0 = conv_layer_interl_tiled_f16(conv_0, &interl_in_tensor, kernel_compact_tiled_0, codebook_interleaved_0_f16, TILE_L2_SIZE, TILE_L1_SIZE, 4);
    
    // print_interleaved_out(&output_tensors_interl_4D_0, 4);
    // exit(0);

    // ReLu //
    relu(output_tensors_interl_4D_0.tensor, (output_tensors_interl_4D_0.dim.depth * output_tensors_interl_4D_0.dim.height * output_tensors_interl_4D_0.dim.width));

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_0, POOL_SIZE_1, POOL_STRIDE_1, 4,  &output_tensors_interl_4D_0);

    // print_interleaved_out(&output_tensors_interl_4D_0, 4);
    // exit(0);


    // Conv 2 //
    // tensor3D_t output_tensors_interl_4D_2 = conv_layer_interl_tiled(conv_2, &output_tensors_interl_4D_0, kernel_compact_perCH_2, codebook_interleaved_2, TILE_L2_SIZE, 4);
    tensor3D_t output_tensors_interl_4D_2 = conv_layer_interl_tiled_f16(conv_2, &output_tensors_interl_4D_0, kernel_compact_tiled_2, codebook_interleaved_2_f16, TILE_L2_SIZE, TILE_L1_SIZE, 4);

    // print_interleaved_out(&output_tensors_interl_4D_2, 4);
    // exit(0);

    // ReLu //
    relu(output_tensors_interl_4D_2.tensor, (output_tensors_interl_4D_2.dim.depth * output_tensors_interl_4D_2.dim.height * output_tensors_interl_4D_2.dim.width));

    // Max Pool //
    maxPool_interleavedND(&output_tensors_interl_4D_2, POOL_SIZE_3, POOL_STRIDE_3, 4, &output_tensors_interl_4D_2);

    // print_interleaved_out(&output_tensors_interl_4D_2, 4);


    // Dense 4 //
    float *out_4_interl = (float*)malloc(dense_4.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D_f16(dense_4, output_tensors_interl_4D_2.tensor, codebook_interleaved_4_f16, weight_idx_compact_4, out_4_interl);
    relu(out_4_interl, dense_4.out_size * N_LEARNERS);
    free(output_tensors_interl_4D_2.tensor);

    // print_flattened_interleaved(out_4_interl, dense_4.out_size, N_LEARNERS);


    // Dense 5 //
    float *out_5_interl = (float*)malloc(dense_5.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D_f16(dense_5, out_4_interl, codebook_interleaved_5_f16, weight_idx_compact_5, out_5_interl);
    relu(out_5_interl, dense_5.out_size * N_LEARNERS);
    free(out_4_interl);
    // print_flattened_interleaved(out_5_interl, dense_5.out_size, N_LEARNERS);


    // Dense 6 //
    float *out_6_interl = (float*)malloc(dense_6.out_size * N_LEARNERS * sizeof(float));
    exec_sve_compact_interleaved_4D_f16(dense_6, out_5_interl, codebook_interleaved_6_f16, weight_idx_compact_6, out_6_interl);
    free(out_5_interl);


#ifdef PRINT_EN
    print_flattened_interleaved(out_6_interl, dense_6.out_size, N_LEARNERS);
#endif

    free(out_6_interl);

    // conv3D_staticPatch_compactSVE_interleaved4D(conv_2, output_tensors);

    // for(int ens=0; ens<N_LEARNERS; ens++){
    //     free(padded_inputs_arrs[ens]->tensor);
    // }

}



