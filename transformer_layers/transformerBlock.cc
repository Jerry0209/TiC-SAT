//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>


#ifdef USE_CODEBOOK
#include "codebookDense.h" // Further improvement: find a more flexible ways to integrate more layers in the future instead of hardcoded the names
#define codebooks_ codebooks_0 // The same name of the attributes in weights header files
#define codebook_interleaved_ codebook_interleaved_0
#include "../Full_NN/gemm_definitions/gemm_header_0.h"
#undef codebook_interleaved_
#undef codebooks_
#if __has_include("../Full_NN/gemm_definitions/gemm_header_1.h")
#define codebooks_ codebooks_1 // ?
#define codebook_interleaved_ codebook_interleaved_1
#include "../Full_NN/gemm_definitions/gemm_header_1.h"
#undef codebook_interleaved_
#undef codebooks_
#define TIC_SAT_HAS_CODEBOOK_FF1 1
#else
#define TIC_SAT_HAS_CODEBOOK_FF1 0
#endif
#endif

namespace {
constexpr const char *kFfn0DebugPath = "/home/thu/TiC-SAT/weights/ffn0_output.bin";
constexpr const char *kFfn1DebugPath = "/home/thu/TiC-SAT/weights/ffn1_output_pre_addnorm.bin";
constexpr const char *kCondenseAddNormDebugPath = "/home/thu/TiC-SAT/weights/condense_out_after_addnorm.txt";



// Run statistics if the program is run on gem 5
void runM5IfAvailable(const char *command) {
    if (std::system("command -v m5 >/dev/null 2>&1") == 0) {
        std::system(command);
    }
}

#ifdef USE_CODEBOOK
// Class for preloading the codebooked weights from layer header files
CodebookDenseConfig makeCodebookDenseConfig(std::size_t expected_input_size,
                                            std::size_t expected_output_size,
                                            std::size_t input_size,
                                            std::size_t output_size,
                                            std::size_t n_words_row,
                                            const uint32_t *weight_idx,
                                            const int8_t *codebook_int8,
                                            const float *bias) {
    if (expected_input_size != input_size || expected_output_size != output_size) {
        throw std::invalid_argument("CodebookDense config shape does not match Transformer FFN dimensions");
    }

    (void)bias;

    CodebookDenseConfig config{};
    config.input_size = input_size;
    config.output_size = output_size;
    config.n_words_row = n_words_row;
    config.bits_per_cb = BITS_PER_CB;
    config.weight_idx = weight_idx;
    config.codebook_int8 = codebook_int8;
    config.bias = nullptr;
    config.input_dequant_scale = 1.0f;
    config.output_quant_scale = 1.0f;
    config.reverse_input_groups_of_4 = true;
    return config;
}
#endif
}

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t ** weightVector,
                                   std::size_t kernelDim, std::size_t maxCol) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3,
                                              kernelDim, maxCol);
    }

    condense = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3]);

    multihead_out = new uint32_t[pre_seq_len * num_heads * head_hidden_size >> 2]();
    condense_out = new uint32_t[pre_seq_len * input_dim >> 2]();
    intermediateFF = new uint32_t[pre_seq_len * ff_size >> 2]();
#ifdef USE_CODEBOOK
    referenceFF0 = new uint32_t[pre_seq_len * ff_size >> 2]();
    referenceFF1 = new uint32_t[pre_seq_len * input_dim >> 2]();
#endif

#ifndef BWMA
    multihead_out_reshape = new uint32_t[pre_seq_len * num_heads * head_hidden_size >> 2]();
#endif

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);
#ifdef USE_CODEBOOK
    // Use codebooked feedforward layer
    feedForward0 = new CodebookDense(makeCodebookDenseConfig(
            input_dim, ff_size, INPUT_SIZE_0, OUTPUT_SIZE_0, N_WORDS_ROW_0,
            weight_idx_compact_0, codebooks_0[0], bias_0[0])); // Get codebooks, indexes and bias from gemm_header
#if TIC_SAT_HAS_CODEBOOK_FF1
    // Test for two codebooked feedforward layers
    feedForward1 = new CodebookDense(makeCodebookDenseConfig(
            ff_size, input_dim, INPUT_SIZE_1, OUTPUT_SIZE_1, N_WORDS_ROW_1,
            weight_idx_compact_1, codebooks_1[0], bias_1[0]));
#else
    std::cout << "USE_CODEBOOK enabled, but gemm_header_1.h was not found; feedForward1 falls back to Dense." << std::endl;
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
#endif
#else
    feedForward0 = new Dense(input_dim, ff_size, weightVector[num_heads * 3+ 1]);
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
#endif
#ifdef USE_CODEBOOK
    feedForward0Reference = new Dense(input_dim, ff_size, weightVector[num_heads * 3 + 1]);
    feedForward1Reference = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
#endif
}

TransformerBlock::~TransformerBlock() = default;


void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    runM5IfAvailable("m5 resetstats");
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ >> 2));
    }

#ifndef BWMA
    Transpose::multihead_transpose(multihead_out, multihead_out_reshape,
                                   seq_len, head_hidden_size_ >> 2, num_heads_);
    multihead_out = multihead_out_reshape;
#endif

    std::cout << "Condense"  << std::endl;
    condense->compute(seq_len, multihead_out, condense_out);
    printPackedPreview("condense_out", condense_out, (seq_len * input_dim_) >> 2);


    std::cout << "Add Norm"  << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(input, condense_out);
#else
    addNorm->compute(input, condense_out);
#endif

    printPackedMatrix("condense_out_after_addnorm", condense_out, seq_len, input_dim_);
    savePackedMatrixText(kCondenseAddNormDebugPath, condense_out, seq_len, input_dim_);

    
    runM5IfAvailable("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);
    printPackedPreview("ffn0", intermediateFF, seq_len * ff_size_ >> 2); // For debugging
    savePackedBuffer(kFfn0DebugPath, intermediateFF, seq_len * ff_size_ >> 2);
#ifdef USE_CODEBOOK
    std::fill(referenceFF0, referenceFF0 + (seq_len * ff_size_ >> 2), 0u);
    feedForward0Reference->compute(seq_len, condense_out, referenceFF0);
    comparePackedBuffers("ffn0", referenceFF0, intermediateFF, seq_len * ff_size_ >> 2);
#endif

    std::cout << "Feed Forward 1"  << std::endl;
    feedForward1->compute(seq_len, intermediateFF, output);
    printPackedPreview("ffn1_pre_addnorm", output, seq_len * input_dim_ >> 2);
    savePackedBuffer(kFfn1DebugPath, output, seq_len * input_dim_ >> 2);
#ifdef USE_CODEBOOK 
    std::fill(referenceFF1, referenceFF1 + (seq_len * input_dim_ >> 2), 0u);
    feedForward1Reference->compute(seq_len, referenceFF0, referenceFF1);
    comparePackedBuffers("ffn1_pre_addnorm", referenceFF1, output, seq_len * input_dim_ >> 2);
#endif

    std::cout << "Add Norm"  << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(condense_out, output);
#else
    addNorm->compute(condense_out, output);
#endif
    runM5IfAvailable("m5 dumpresetstats");

}
