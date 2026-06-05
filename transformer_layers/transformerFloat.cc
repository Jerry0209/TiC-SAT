#include "transformerFloat.h"

#include "run_mode_config.h"

#if CFG_USE_FP32_TRANSFORMER

#include "floatCommon.h"
#include "floatDump.h"
#include "floatTransformerBlock.h"

#include "../Full_NN/gemm_definitions/input_matrix.h"
#include "../transformer.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Load the generated FP32 input matrix used as the starting tensor for the
// transformer block. The static checks keep the generated header dimensions in
// sync with the transformer configuration before the data is copied into the
// Matrix container used by the floating-point implementation.
TransformerFloat::Matrix loadInputMatrix() {
    static_assert(GEMM_M == D_SEQ, "input_matrix.h GEMM_M must match D_SEQ");
    static_assert(GEMM_K == D_MODEL, "input_matrix.h GEMM_K must match D_MODEL");
    return TransformerFloat::Matrix(input_matrix, input_matrix + (D_SEQ * D_MODEL));
}

// Build the optional per-learner dump directory. An empty output root disables
// dumping, while a non-empty root gives each learner its own directory so the
// intermediate tensors can be compared independently.
std::string learnerDumpDir(const std::string& c_output_root,
                           std::size_t learner) {
    if (c_output_root.empty()) {
        return std::string();
    }
    return c_output_root + "/learner" + std::to_string(learner);
}

} // namespace

namespace TransformerFloat {

// Top-level entry point for the FP32 transformer reference path.
//
// This function prepares the shared input, constructs one transformer block per
// learner, chooses the appropriate single/grouped execution path, and finally
// writes optional output dumps. It is used when CFG_USE_FP32_TRANSFORMER selects
// the floating-point transformer implementation instead of the integer path.
void run(std::size_t learner_count, const std::string& c_output_root) {
    // Step 1: Treat a zero learner request as the default one-learner run.
    if (learner_count == 0u) {
        learner_count = 1u;
    }

    std::cout << "CFG_USE_FP32_TRANSFORMER = 1" << std::endl;
    std::cout << "FP32 Transformer uses generated codebook registry; Dense reference is disabled."
              << std::endl;

    // Step 2: Load the generated input matrix and allocate one output matrix,
    // dump path, and transformer block holder for each requested learner.
    Matrix input = loadInputMatrix();
    std::vector<std::string> dump_dirs(learner_count);
    std::vector<Matrix> outputs(learner_count, Matrix(D_SEQ * D_MODEL, 0.0f));
    std::vector<std::unique_ptr<FloatTransformerBlock>> blocks;
    blocks.reserve(learner_count);

    // Step 3: Create the per-learner transformer blocks. Each block owns the
    // codebook dense layers, attention heads, feed-forward layers, and dump
    // location for that learner.
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        if (learner_count > 1u) {
            std::cout << "\n=============== LEARNER " << learner
                      << " ===============\n" << std::endl;
        }

        dump_dirs[learner] = learnerDumpDir(c_output_root, learner);
        if (!dump_dirs[learner].empty()) {
            std::filesystem::create_directories(dump_dirs[learner]);
        }
        dumpFloatMatrixIfEnabled(dump_dirs[learner], "input_matrix.txt",
                                 input.data(), D_SEQ, D_MODEL);

        blocks.push_back(std::make_unique<FloatTransformerBlock>(
            D_SEQ,
            D_MODEL,
            D_Q,
            NUM_HEAD,
            D_FF,
            learner,
            dump_dirs[learner]));
    }

    // Step 4: Run the transformer. Two- and four-learner requests use grouped
    // helpers so shared pipeline stages can execute together; other counts fall
    // back to independent block execution.
    if (learner_count == 2u) {
        FloatTransformerBlock* grouped_blocks[2] = {
            blocks[0].get(),
            blocks[1].get(),
        };
        const float* grouped_inputs[2] = {
            input.data(),
            input.data(),
        };
        float* grouped_outputs[2] = {
            outputs[0].data(),
            outputs[1].data(),
        };
        FloatTransformerBlock::computeGroup2(
            D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);
    } else if (learner_count == 4u) {
        FloatTransformerBlock* grouped_blocks[4] = {
            blocks[0].get(),
            blocks[1].get(),
            blocks[2].get(),
            blocks[3].get(),
        };
        const float* grouped_inputs[4] = {
            input.data(),
            input.data(),
            input.data(),
            input.data(),
        };
        float* grouped_outputs[4] = {
            outputs[0].data(),
            outputs[1].data(),
            outputs[2].data(),
            outputs[3].data(),
        };
        FloatTransformerBlock::computeGroup4(
            D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);
    } else {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            blocks[learner]->compute(D_SEQ, input.data(), outputs[learner].data());
        }
    }

    // Step 5: Dump the final tensor for each learner when dumping is enabled.
    for (std::size_t learner = 0; learner < outputs.size(); learner++) {
        dumpFloatMatrixIfEnabled(dump_dirs[learner], "output.txt",
                                 outputs[learner].data(), D_SEQ, D_MODEL);
    }
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
