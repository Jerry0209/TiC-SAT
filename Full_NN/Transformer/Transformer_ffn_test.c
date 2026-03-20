#include <math.h>
#include <stdio.h>
#include <string.h>

#include <feedforward.h>
#include "transformer_def.h"

static void print_matrix(const char *label,
                         const float *data,
                         uint32_t rows,
                         uint32_t cols) {
    printf("%s\n", label);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t col = 0; col < cols; col++) {
            printf("%8.4f ", data[(row * cols) + col]);
        }
        printf("\n");
    }
}

int main(void) {
    float hidden_ref[TRANSFORMER_SEQ_LEN * TRANSFORMER_D_FF];
    float out_ref[TRANSFORMER_SEQ_LEN * TRANSFORMER_D_MODEL];
    float hidden_compact[TRANSFORMER_SEQ_LEN * TRANSFORMER_D_FF];
    float out_compact[TRANSFORMER_SEQ_LEN * TRANSFORMER_D_MODEL];
    int ok = 1;

    memset(hidden_ref, 0, sizeof(hidden_ref));
    memset(out_ref, 0, sizeof(out_ref));
    memset(hidden_compact, 0, sizeof(hidden_compact));
    memset(out_compact, 0, sizeof(out_compact));

    feedforward_exec_noCB(transformer_ffn,
                          transformer_input,
                          transformer_fc0_weights,
                          transformer_fc0_bias,
                          transformer_fc1_weights,
                          transformer_fc1_bias,
                          hidden_ref,
                          out_ref);

    feedforward_exec_compact(transformer_ffn,
                             transformer_input,
                             transformer_fc0_idx,
                             transformer_codebook_0,
                             transformer_fc0_bias,
                             transformer_fc1_idx,
                             transformer_codebook_1,
                             transformer_fc1_bias,
                             hidden_compact,
                             out_compact,
                             TRANSFORMER_BITS_PER_CB);

    for (uint32_t idx = 0; idx < TRANSFORMER_SEQ_LEN * TRANSFORMER_D_MODEL; idx++) {
        float diff = fabsf(out_ref[idx] - out_compact[idx]);
        if (diff > 1e-5f) {
            ok = 0;
            printf("Mismatch at %u: got=%f expected=%f diff=%f\n",
                   idx, out_compact[idx], out_ref[idx], diff);
        }
    }

    print_matrix("Reference output:", out_ref,
                 TRANSFORMER_SEQ_LEN, TRANSFORMER_D_MODEL);
    print_matrix("Compact output:", out_compact,
                 TRANSFORMER_SEQ_LEN, TRANSFORMER_D_MODEL);

    if (ok) {
        printf("PASS\n");
        return 0;
    }

    printf("FAIL\n");
    return 1;
}

/*
 * Example build command:
 * gcc -O2 -std=c11 \
 *   Transformer/Transformer_ffn_test.c src/gemm_exec.c src/feedforward.c src/relu.c \
 *   -Iinc -ITransformer -march=armv8.2-a+sve -o Transformer_ffn_test
 */
