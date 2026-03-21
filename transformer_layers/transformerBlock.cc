//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>


#ifdef USE_CODEBOOK
#include "codebookDense.h"
#include "../Full_NN/gemm_definitions/gemm_header_0.h"
#if __has_include("../Full_NN/gemm_definitions/gemm_header_1.h")
#include "../Full_NN/gemm_definitions/gemm_header_1.h"
#define TIC_SAT_HAS_CODEBOOK_FF1 1
#else
#define TIC_SAT_HAS_CODEBOOK_FF1 0
#endif
#endif

namespace {
constexpr const char *kFfn0DebugPath = "/home/thu/TiC-SAT/weights/ffn0_output.bin";
constexpr const char *kFfn1DebugPath = "/home/thu/TiC-SAT/weights/ffn1_output_pre_addnorm.bin";
constexpr float kNotebookWeightQuantScale = 32.0f;

void runM5IfAvailable(const char *command) {
    if (std::system("command -v m5 >/dev/null 2>&1") == 0) {
        std::system(command);
    }
}

void printPackedPreview(const char *label, const uint32_t *buffer, std::size_t packed_size) {
    std::size_t preview = std::min<std::size_t>(packed_size, 8);
    std::cout << label << " preview (first " << preview << " packed words):" << std::endl;
    for (std::size_t i = 0; i < preview; i++) {
        std::cout << label << "[" << i << "] = " << buffer[i] << " -> [";
        for (int j = 0; j < 4; j++) {
            int8_t value = static_cast<int8_t>((buffer[i] >> (8 * j)) & 0xFF);
            std::cout << static_cast<int>(value);
            if (j != 3) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

void savePackedBuffer(const char *filename, const uint32_t *buffer, std::size_t packed_size) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cout << filename << " Not saved" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < packed_size; i++) {
        fout << buffer[i] << " ";
    }
    fout.close();
}

int8_t unpackPackedValue(const uint32_t *buffer, std::size_t elem_idx) {
    std::size_t word_idx = elem_idx / 4;
    std::size_t byte_idx = elem_idx % 4;
    return static_cast<int8_t>((buffer[word_idx] >> (byte_idx * 8)) & 0xFF);
}

void comparePackedBuffers(const char *label,
                          const uint32_t *dense_reference,
                          const uint32_t *candidate,
                          std::size_t packed_size) { 
    std::size_t total_values = packed_size * 4;
    int max_abs_diff = 0;
    std::size_t mismatch_count = 0;
    std::size_t first_mismatch = total_values;
    int first_dense_value = 0;
    int first_candidate_value = 0;

    for (std::size_t idx = 0; idx < total_values; idx++) {
        int dense_value = static_cast<int>(unpackPackedValue(dense_reference, idx));
        int candidate_value = static_cast<int>(unpackPackedValue(candidate, idx));
        int abs_diff = dense_value >= candidate_value ? (dense_value - candidate_value)
                                                     : (candidate_value - dense_value);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        if (abs_diff != 0) {
            if (first_mismatch == total_values) {
                first_mismatch = idx;
                first_dense_value = dense_value;
                first_candidate_value = candidate_value;
            }
            mismatch_count++;
        }
    }

    std::cout << label << " diff vs Dense reference: max_abs_diff=" << max_abs_diff
              << ", mismatches=" << mismatch_count << "/" << total_values << std::endl;
    if (mismatch_count != 0) {
        std::cout << label << " first mismatch at value[" << first_mismatch << "]: dense="
                  << first_dense_value << ", candidate=" << first_candidate_value << std::endl;
    }
}

#ifdef USE_CODEBOOK
CodebookDenseConfig makeCodebookDenseConfig(std::size_t expected_input_size,
                                            std::size_t expected_output_size,
                                            std::size_t input_size,
                                            std::size_t output_size,
                                            std::size_t n_words_row,
                                            const uint32_t *weight_idx,
                                            const float *codebook,
                                            const float *bias) {
    if (expected_input_size != input_size || expected_output_size != output_size) {
        throw std::invalid_argument("CodebookDense config shape does not match Transformer FFN dimensions");
    }

    return CodebookDenseConfig{
        input_size,
        output_size,
        n_words_row,
        BITS_PER_CB,
        weight_idx,
        codebook,
        nullptr,
        1.0f,
        kNotebookWeightQuantScale,
    };
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
    feedForward0 = new CodebookDense(makeCodebookDenseConfig(
            input_dim, ff_size, INPUT_SIZE_0, OUTPUT_SIZE_0, N_WORDS_ROW_0,
            weight_idx_compact_0, codebooks_0[0], bias_0[0]));
#if TIC_SAT_HAS_CODEBOOK_FF1
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


    std::cout << "Add Norm"  << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(input, condense_out);
#else
    addNorm->compute(input, condense_out);
#endif

    runM5IfAvailable("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);
    printPackedPreview("ffn0", intermediateFF, seq_len * ff_size_ >> 2);
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
