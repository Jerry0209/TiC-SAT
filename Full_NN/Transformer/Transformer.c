#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <main.h>

#include <Transformer.h>
#include <transformer_def.h>

static uint32_t transformer_get_packed_index(const uint32_t *packed_row,
                                             uint32_t elem_idx) {
    uint32_t word_idx = elem_idx / IDXS_PER_WORD;
    uint32_t offset = (elem_idx % IDXS_PER_WORD) * TRANSFORMER_BITS_PER_CB;

    return (packed_row[word_idx] >> offset) & ((1u << TRANSFORMER_BITS_PER_CB) - 1u);
}

static void transformer_unpack_gemm_0_weights(const float *codebook, float *weights) {
    for (uint32_t out_idx = 0; out_idx < transformer_gemm_0.output_size; out_idx++) {
        const uint32_t *packed_row =
            &weight_idx_compact_0[out_idx * transformer_gemm_0.n_words_row];

        for (uint32_t in_idx = 0; in_idx < transformer_gemm_0.input_size; in_idx++) {
            uint32_t cb_idx = transformer_get_packed_index(packed_row, in_idx);
            weights[(out_idx * transformer_gemm_0.input_size) + in_idx] = codebook[cb_idx];
        }
    }
}

static int transformer_check_outputs(const float *reference_out,
                                     const float *compact_out,
                                     uint32_t n_elems) {
    for (uint32_t idx = 0; idx < n_elems; idx++) {
        float diff = compact_out[idx] - reference_out[idx];

        if (diff < 0.0f) {
            diff = -diff;
        }

        if (diff > 1e-5f) {
            printf("Mismatch at %u: compact=%f reference=%f diff=%f\n",
                   idx, compact_out[idx], reference_out[idx], diff);
            return 0;
        }
    }

    return 1;
}

static void transformer_print_output(const char *label, const float *out) {
    printf("%s\n", label);

    for (uint32_t seq = 0; seq < transformer_gemm_0.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < transformer_gemm_0.output_size; out_idx++) {
            printf("%10.6f ", out[(seq * transformer_gemm_0.output_size) + out_idx]);
        }
        printf("\n");
    }
}

static void exec_transformer_gemm_impl(void) {
    uint32_t n_weights = transformer_gemm_0.output_size * transformer_gemm_0.input_size;
    uint32_t n_outputs = transformer_gemm_0.seq_len * transformer_gemm_0.output_size;

    for (uint32_t ens = 0; ens < N_LEARNERS; ens++) {
        const float *codebook = codebooks_0[ens];
        const float *bias = bias_0[ens];
        float *decoded_weights = (float *)malloc(n_weights * sizeof(float));
        float *reference_out = (float *)malloc(n_outputs * sizeof(float));
        float *compact_out = (float *)malloc(n_outputs * sizeof(float));

        if (decoded_weights == NULL || reference_out == NULL || compact_out == NULL) {
            printf("Transformer GEMM allocation failed for learner %u\n", ens);
            free(decoded_weights);
            free(reference_out);
            free(compact_out);
            return;
        }

        memset(reference_out, 0, n_outputs * sizeof(float));
        memset(compact_out, 0, n_outputs * sizeof(float));

        transformer_unpack_gemm_0_weights(codebook, decoded_weights);

        gemm_exec_noCB(transformer_gemm_0,
                       input_matrix,
                       decoded_weights,
                       bias,
                       reference_out);

        gemm_exec_compact(transformer_gemm_0,
                          input_matrix,
                          weight_idx_compact_0,
                          codebook,
                          bias,
                          compact_out,
                          TRANSFORMER_BITS_PER_CB);

        if (!transformer_check_outputs(reference_out, compact_out, n_outputs)) {
            printf("Transformer GEMM check failed for learner %u\n", ens);
        }

#ifdef PRINT_EN
        printf("\n---- Transformer learner %u ----\n", ens);
        transformer_print_output("GEMM output:", compact_out);
#endif

        free(decoded_weights);
        free(reference_out);
        free(compact_out);
    }
}

void exec_Transformer_gemm() {
    exec_transformer_gemm_impl();
}

void exec_Transformer_learner_by_learner() {
    exec_transformer_gemm_impl();
}

void exec_Transformer_learner_by_learner_tiled_l2l1_diff_seq() {
    exec_transformer_gemm_impl();
}
