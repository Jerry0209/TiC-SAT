
#include <stdio.h>

#include <codebooks_def.h>
#include <dense_exec.h>





void exec_compact_noCB(dense_t dense_layer, const float *in, const float *weights, float *bias, float *out){

    float temp_res = 0.0;

    for(int r=0; r<dense_layer.out_size; r++){
        temp_res = 0.0;
        for(int w=0; w<dense_layer.in_size; w++){
            // printf("r: %d, w: %d | in[%d] * weights[%d]\n", r, w, w, (r*dense_layer.in_size) + w);
            temp_res += in[w] * weights[(r*dense_layer.in_size) + w];
            // printf("done mult\n");
        }
        // printf("done row\n");
        out[r] = temp_res + bias[r];
    }
    // printf("done row\n");
}






void exec_compact(dense_t dense_layer, const float *in, const uint32_t *indexes, const float* codebook, float *bias, float *out){

    uint8_t index = 0; 
    uint8_t shamt = 0;
    uint32_t missing;   // missing indexes to process
    float temp_res = 0.0;

    for(int r=0; r<dense_layer.out_size; r++){
        missing = dense_layer.in_size;
        temp_res = 0.0;
        for(int w=0; w<dense_layer.n_words_row; w++){
            for(int c=0; c<IDXS_PER_WORD; c++){
                shamt = c*BITS_PER_CB;
                index = ((indexes[(r * dense_layer.n_words_row) + w] >> shamt) & IDX_MASK);
                // out[r] += in[(w*IDXS_PER_WORD)+c] * codebook[index];
                temp_res += in[(w*IDXS_PER_WORD)+c] * codebook[index];

                // printf("R: %d, C: %d, out = %f (%f) ([%d] %f, [%d] %f)\n", r, c, out[r], in[(w*IDXS_PER_WORD)+c] * codebook[index], (w*IDXS_PER_WORD)+c, in[(w*IDXS_PER_WORD)+c], index, codebook[index]);

                missing--;
                if (missing <= 0){
                    break;
                }

            }
        }
        out[r] = temp_res + bias[r];
    }
}