#include "transformerBlockInterleavedHelpers.h"

#if CFG_FULL_INTERLEAVED_PIPELINE

#include "codebookDense.h"
#include "debuggerFunctions.h"
#include "interleavedPipeline.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

CodebookDense* requireInterleavedCodebookDense2(const char* label,
                                                LinearLayer* const layers[2]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved2DSameSeq()) {
        throw std::runtime_error(std::string(label) + " does not support the 2D interleaved pipeline");
    }

    for (std::size_t learner = 1; learner < 2u; learner++) {
        auto* layer = dynamic_cast<CodebookDense*>(layers[learner]);
        if (layer == nullptr || !layer->supportsInterleaved2DSameSeq()) {
            throw std::runtime_error(std::string(label) + " learner layer does not support the 2D interleaved pipeline");
        }
    }

    return primary;
}

CodebookDense* requireInterleavedCodebookDense4(const char* label,
                                                LinearLayer* const layers[4]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved4DDiffSeq()) {
        throw std::runtime_error(std::string(label) + " does not support the 4D interleaved pipeline");
    }

    for (std::size_t learner = 1; learner < 4u; learner++) {
        auto* layer = dynamic_cast<CodebookDense*>(layers[learner]);
        if (layer == nullptr || !layer->supportsInterleaved4DDiffSeq()) {
            throw std::runtime_error(std::string(label) + " learner layer does not support the 4D interleaved pipeline");
        }
    }

    return primary;
}

} // namespace

void computeCodebookDenseInterleaved2D(const char* label,
                                       LinearLayer* const layers[2],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense2(label, layers);
    primary->computeInterleaved2DToInt8(seq_len, input_interleaved, output_interleaved);
}

void computeCodebookDenseInterleaved4D(const char* label,
                                       LinearLayer* const layers[4],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense4(label, layers);
    primary->computeInterleaved4DToInt8(seq_len, input_interleaved, output_interleaved);
}

#if CFG_ENABLE_DEBUG_PRINT
void printInterleavedPackedPreview2D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner) {
    std::vector<uint32_t> packed((rows * cols) >> 2, 0u);
    packInterleavedLearner2(rows, cols, input_interleaved, learner, packed.data());
    printPackedPreview(label, packed.data(), packed.size());
}

void printInterleavedPackedPreview4D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner) {
    std::vector<uint32_t> packed((rows * cols) >> 2, 0u);
    packInterleavedLearner4(rows, cols, input_interleaved, learner, packed.data());
    printPackedPreview(label, packed.data(), packed.size());
}
#endif

#if CFG_USE_CODEBOOK_REFERENCE
void compareInterleavedDenseReference2D(const char* label,
                                        LinearLayer* const references[2],
                                        uint32_t* const reference_outputs[2],
                                        const std::size_t learner_ids[2],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_input((seq_len * input_cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * output_cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 2u; learner++) {
        std::fill(reference_outputs[learner],
                  reference_outputs[learner] + ((seq_len * output_cols) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner2(
            seq_len,
            input_cols,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner2(
            seq_len,
            output_cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        references[learner]->compute(
            seq_len,
            packed_input.data(),
            reference_outputs[learner]);

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            reference_outputs[learner],
            packed_candidate.data(),
            (seq_len * output_cols) >> 2);
    }
}

void compareInterleavedDenseReference4D(const char* label,
                                        LinearLayer* const references[4],
                                        uint32_t* const reference_outputs[4],
                                        const std::size_t learner_ids[4],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_input((seq_len * input_cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * output_cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 4u; learner++) {
        std::fill(reference_outputs[learner],
                  reference_outputs[learner] + ((seq_len * output_cols) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner4(
            seq_len,
            input_cols,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner4(
            seq_len,
            output_cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        references[learner]->compute(
            seq_len,
            packed_input.data(),
            reference_outputs[learner]);

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            reference_outputs[learner],
            packed_candidate.data(),
            (seq_len * output_cols) >> 2);
    }
}

void compareInterleavedAddNormReference2D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[2],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_residual((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_reference((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 2u; learner++) {
        std::fill(packed_residual.begin(), packed_residual.end(), 0u);
        std::fill(packed_reference.begin(), packed_reference.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner2(
            seq_len,
            cols,
            residual_interleaved,
            learner,
            packed_residual.data());
        packInterleavedLearner2(
            seq_len,
            cols,
            pre_addnorm_interleaved,
            learner,
            packed_reference.data());
        packInterleavedLearner2(
            seq_len,
            cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        add_norm->compute(packed_residual.data(), packed_reference.data());

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            packed_reference.data(),
            packed_candidate.data(),
            (seq_len * cols) >> 2);
    }
}

void compareInterleavedAddNormReference4D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[4],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_residual((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_reference((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 4u; learner++) {
        std::fill(packed_residual.begin(), packed_residual.end(), 0u);
        std::fill(packed_reference.begin(), packed_reference.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner4(
            seq_len,
            cols,
            residual_interleaved,
            learner,
            packed_residual.data());
        packInterleavedLearner4(
            seq_len,
            cols,
            pre_addnorm_interleaved,
            learner,
            packed_reference.data());
        packInterleavedLearner4(
            seq_len,
            cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        add_norm->compute(packed_residual.data(), packed_reference.data());

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            packed_reference.data(),
            packed_candidate.data(),
            (seq_len * cols) >> 2);
    }
}
#endif

#endif
