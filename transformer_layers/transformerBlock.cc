//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "layerFactory.h"
#include "run_mode_config.h"
#include "codebookDense.h"
#include "interleavedPipeline.h"


namespace {

// Run m5 command only if gem5 helper is available.
void runM5IfAvailable(const char* command) {
    if (std::system("command -v m5 >/dev/null 2>&1") == 0) {
        std::system(command);
    }
}

#if CFG_FULL_INTERLEAVED_PIPELINE
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

void computeCodebookDenseInterleaved4D(const char* label,
                                       LinearLayer* const layers[4],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense4(label, layers);
    primary->computeInterleaved4DToInt8(seq_len, input_interleaved, output_interleaved);
}

#if CFG_ENABLE_DEBUG_PRINT
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

} // namespace

TransformerBlock::TransformerBlock(std::size_t pre_seq_len,
                                   std::size_t input_dim,
                                   std::size_t head_hidden_size,
                                   std::size_t num_heads,
                                   std::size_t ff_size,
                                   uint32_t** weightVector,
                                   std::size_t kernelDim,
                                   std::size_t maxCol,
                                   std::size_t learner_idx,
                                   std::string dump_dir) {
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;
    learner_idx_ = learner_idx;
    dump_dir_ = dump_dir;

    // Create one self-attention module per head.
    selfatten_.reserve(num_heads_);
    for (std::size_t n = 0; n < num_heads_; ++n) {
        selfatten_.push_back(
            new SingleHeadSelfAttn(
                n,
                pre_seq_len,
                input_dim,
                head_hidden_size,
                weightVector + n * 3,
                kernelDim,
                maxCol,
                learner_idx_,
                dump_dir_));
    }

    // Allocate intermediate buffers.
    multihead_out = new uint32_t[(pre_seq_len * num_heads * head_hidden_size) >> 2]();
    condense_out = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    intermediateFF = new uint32_t[(pre_seq_len * ff_size) >> 2]();

#ifndef BWMA
    multihead_out_reshape = new uint32_t[(pre_seq_len * num_heads * head_hidden_size) >> 2]();
#endif

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);

    // Create post-attention and FFN layers through the factory.
    // The factory automatically uses CodebookDense if the layer exists in registry,
    // otherwise it falls back to Dense.
    auto condense_bundle = LayerFactory::create(
        "condense",
        num_heads * head_hidden_size,
        input_dim,
        weightVector[num_heads * 3],
        learner_idx_);

    condense = condense_bundle.main;

    auto ff0_bundle = LayerFactory::create(
        "ff0",
        input_dim,
        ff_size,
        weightVector[num_heads * 3 + 1],
        learner_idx_);

    feedForward0 = ff0_bundle.main;

    auto ff1_bundle = LayerFactory::create(
        "ff1",
        ff_size,
        input_dim,
        weightVector[num_heads * 3 + 2],
        learner_idx_);

    feedForward1 = ff1_bundle.main;

#if CFG_USE_CODEBOOK_REFERENCE
    // Optional Dense reference path for validation.
    condenseReference = condense_bundle.reference;
    feedForward0Reference = ff0_bundle.reference;
    feedForward1Reference = ff1_bundle.reference;

    referenceCondense = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceCondenseAfterAddNorm = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceFF0 = new uint32_t[(pre_seq_len * ff_size) >> 2]();
    referenceFF1 = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceFinalOutput = new uint32_t[(pre_seq_len * input_dim) >> 2]();
#endif
}

TransformerBlock::~TransformerBlock() {
    for (auto* h : selfatten_) {
        delete h;
    }

    delete[] multihead_out;
    delete[] condense_out;
    delete[] intermediateFF;

#ifndef BWMA
    delete[] multihead_out_reshape;
#endif

    delete addNorm;

    delete condense;
    delete feedForward0;
    delete feedForward1;

#if CFG_USE_CODEBOOK_REFERENCE
    delete condenseReference;
    delete feedForward0Reference;
    delete feedForward1Reference;

    delete[] referenceCondense;
    delete[] referenceCondenseAfterAddNorm;
    delete[] referenceFF0;
    delete[] referenceFF1;
    delete[] referenceFinalOutput;
    
#endif
}

void TransformerBlock::compute(std::size_t seq_len, uint32_t* input, uint32_t* output) {
    runM5IfAvailable("m5 resetstats");

    // Compute each attention head output independently.
    for (std::size_t n = 0; n < num_heads_; ++n) {
        std::cout << "Head : " << n << std::endl;
        selfatten_[n]->compute(
            seq_len,
            input,
            multihead_out + n * ((seq_len * head_hidden_size_) >> 2));
    }

    // Do not overwrite the member pointer multihead_out.
    // Use a local pointer to select the correct tensor fed into condense.
    uint32_t* multihead_for_condense = multihead_out;

#ifndef BWMA
    Transpose::multihead_transpose(
        multihead_out,
        multihead_out_reshape,
        seq_len,
        head_hidden_size_ >> 2,
        num_heads_);

    multihead_for_condense = multihead_out_reshape;
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "multihead_out.txt",
        multihead_for_condense,
        seq_len,
        num_heads_ * head_hidden_size_);

    std::cout << "Condense" << std::endl;
    condense->compute(seq_len, multihead_for_condense, condense_out);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "condense_out.txt",
        condense_out,
        seq_len,
        input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceCondense,
              referenceCondense + ((seq_len * input_dim_) >> 2),
              0u);

    condenseReference->compute(seq_len, multihead_for_condense, referenceCondense);

    comparePackedBuffers(
        "condense_out",
        referenceCondense,
        condense_out,
        (seq_len * input_dim_) >> 2);

    std::copy(referenceCondense,
              referenceCondense + ((seq_len * input_dim_) >> 2),
              referenceCondenseAfterAddNorm); // To keep both the pre-addnorm and post-addnorm reference values for later comparison.
#endif

    std::cout << "Add Norm" << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(input, condense_out);
#else
    addNorm->compute(input, condense_out); // Directly modify condense_out itself to be the output of addNorm
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "after_attn_addnorm.txt",
        condense_out,
        seq_len,
        input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
    addNorm->computeRearranged(input, referenceCondenseAfterAddNorm);
#else
    addNorm->compute(input, referenceCondenseAfterAddNorm);
#endif

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "condense_out_after_addnorm",
        referenceCondenseAfterAddNorm,
        condense_out,
        (seq_len * input_dim_) >> 2);
#endif

#endif

    runM5IfAvailable("m5 dumpresetstats");

    std::cout << "Feed Forward 0" << std::endl;
    feedForward0->compute(seq_len, condense_out, intermediateFF);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "ff0_out.txt",
        intermediateFF,
        seq_len,
        ff_size_);

#if CFG_ENABLE_DEBUG_PRINT
    printPackedPreview("ffn0", intermediateFF, (seq_len * ff_size_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceFF0,
              referenceFF0 + ((seq_len * ff_size_) >> 2),
              0u);

    // feedForward0Reference->compute(seq_len, condense_out, referenceFF0);
    feedForward0Reference->compute(seq_len, referenceCondenseAfterAddNorm, referenceFF0);

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "ffn0",
        referenceFF0,
        intermediateFF,
        (seq_len * ff_size_) >> 2);
#endif

#endif

    std::cout << "Feed Forward 1" << std::endl;
    feedForward1->compute(seq_len, intermediateFF, output);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "ff1_out.txt",
        output,
        seq_len,
        input_dim_);

#if CFG_ENABLE_DEBUG_PRINT
    printPackedPreview("ffn1_pre_addnorm", output, (seq_len * input_dim_) >> 2);
#endif


#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceFF1,
              referenceFF1 + ((seq_len * input_dim_) >> 2),
              0u);

    feedForward1Reference->compute(seq_len, referenceFF0, referenceFF1);


#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "ffn1_pre_addnorm",
        referenceFF1,
        output,
        (seq_len * input_dim_) >> 2);
#endif

    std::copy(referenceFF1,
          referenceFF1 + ((seq_len * input_dim_) >> 2),
          referenceFinalOutput);  // std::copy(first, first + count, d_first) Copies the elements in the range [first, last) into the range beginning at d_first.
#endif

    std::cout << "Add Norm" << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(condense_out, output);
#else
    addNorm->compute(condense_out, output);
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "final_out.txt",
        output,
        seq_len,
        input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
    addNorm->computeRearranged(referenceCondenseAfterAddNorm, referenceFinalOutput);
#else
    addNorm->compute(referenceCondenseAfterAddNorm, referenceFinalOutput);
#endif

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "final_output_after_addnorm",
        referenceFinalOutput,
        output,
        (seq_len * input_dim_) >> 2);
#endif

#endif

    runM5IfAvailable("m5 dumpresetstats");
}

template <std::size_t LearnerCount>
void TransformerBlock::computeGroupImpl(std::size_t seq_len,
                                        TransformerBlock** blocks,
                                        uint32_t* const* inputs,
                                        uint32_t* const* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped transformer execution is supported");

#if CFG_FULL_INTERLEAVED_PIPELINE
    if constexpr (LearnerCount == 4u) {
        computeGroup4FullInterleaved(seq_len, blocks, inputs, outputs);
        return;
    }
#endif

    runM5IfAvailable("m5 resetstats");

    for (std::size_t n = 0; n < blocks[0]->num_heads_; ++n) {
        std::cout << "Head : " << n << std::endl;

        SingleHeadSelfAttn* heads[LearnerCount];
        uint32_t* head_outputs[LearnerCount];
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner] = blocks[learner]->selfatten_[n];
            head_outputs[learner] =
                blocks[learner]->multihead_out +
                n * ((seq_len * blocks[learner]->head_hidden_size_) >> 2);
        }

        if constexpr (LearnerCount == 2u) {
            SingleHeadSelfAttn::computeGroup2(seq_len, heads, inputs, head_outputs);
        } else {
            SingleHeadSelfAttn::computeGroup4(seq_len, heads, inputs, head_outputs);
        }
    }

    uint32_t* multihead_for_condense[LearnerCount] = {};
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifndef BWMA
        Transpose::multihead_transpose(
            blocks[learner]->multihead_out,
            blocks[learner]->multihead_out_reshape,
            seq_len,
            blocks[learner]->head_hidden_size_ >> 2,
            blocks[learner]->num_heads_);

        multihead_for_condense[learner] = blocks[learner]->multihead_out_reshape;
#else
        multihead_for_condense[learner] = blocks[learner]->multihead_out;
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "multihead_out.txt",
            multihead_for_condense[learner],
            seq_len,
            blocks[learner]->num_heads_ * blocks[learner]->head_hidden_size_);
    }

    std::cout << "Condense" << std::endl;
    LinearLayer* condense_layers[LearnerCount];
    uint32_t* condense_outputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        condense_layers[learner] = blocks[learner]->condense;
        condense_outputs[learner] = blocks[learner]->condense_out;
    }

    auto tryGroupedCodebookDense = [&](LinearLayer* const layers[LearnerCount],
                                       uint32_t* const dense_inputs[LearnerCount],
                                       uint32_t* const dense_outputs[LearnerCount]) {
        if constexpr (LearnerCount == 2u) {
            return tryComputeGroupedCodebookDense2(layers, seq_len, dense_inputs, dense_outputs);
        } else {
            return tryComputeGroupedCodebookDense4(layers, seq_len, dense_inputs, dense_outputs);
        }
    };

    if (!tryGroupedCodebookDense(condense_layers, multihead_for_condense, condense_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->condense->compute(
                seq_len, multihead_for_condense[learner], blocks[learner]->condense_out);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "condense_out.txt",
            blocks[learner]->condense_out,
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceCondense,
                  blocks[learner]->referenceCondense + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  0u);

        blocks[learner]->condenseReference->compute(
            seq_len, multihead_for_condense[learner], blocks[learner]->referenceCondense);

        const std::string label =
            "condense_out_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceCondense,
                             blocks[learner]->condense_out,
                             (seq_len * blocks[learner]->input_dim_) >> 2);

        std::copy(blocks[learner]->referenceCondense,
                  blocks[learner]->referenceCondense + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  blocks[learner]->referenceCondenseAfterAddNorm);
#endif
    }

    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(inputs[learner], blocks[learner]->condense_out);
#else
        blocks[learner]->addNorm->compute(inputs[learner], blocks[learner]->condense_out);
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "after_attn_addnorm.txt",
            blocks[learner]->condense_out,
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(
            inputs[learner], blocks[learner]->referenceCondenseAfterAddNorm);
#else
        blocks[learner]->addNorm->compute(
            inputs[learner], blocks[learner]->referenceCondenseAfterAddNorm);
#endif

#if CFG_ENABLE_DEBUG_PRINT
        const std::string label =
            "condense_out_after_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceCondenseAfterAddNorm,
                             blocks[learner]->condense_out,
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif
#endif
    }

    runM5IfAvailable("m5 dumpresetstats");

    std::cout << "Feed Forward 0" << std::endl;
    LinearLayer* ff0_layers[LearnerCount];
    uint32_t* ff0_outputs[LearnerCount];
    uint32_t* ff0_inputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        ff0_layers[learner] = blocks[learner]->feedForward0;
        ff0_outputs[learner] = blocks[learner]->intermediateFF;
        ff0_inputs[learner] = blocks[learner]->condense_out;
    }
    if (!tryGroupedCodebookDense(ff0_layers, ff0_inputs, ff0_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->feedForward0->compute(
                seq_len, blocks[learner]->condense_out, blocks[learner]->intermediateFF);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        const std::string ff0_label =
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_);

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff0_out.txt",
            blocks[learner]->intermediateFF,
            seq_len,
            blocks[learner]->ff_size_);

#if CFG_ENABLE_DEBUG_PRINT
        printPackedPreview(ff0_label.c_str(), blocks[learner]->intermediateFF, (seq_len * blocks[learner]->ff_size_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceFF0,
                  blocks[learner]->referenceFF0 + ((seq_len * blocks[learner]->ff_size_) >> 2),
                  0u);

        blocks[learner]->feedForward0Reference->compute(
            seq_len, blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFF0);

#if CFG_ENABLE_DEBUG_PRINT
        comparePackedBuffers(ff0_label.c_str(),
                             blocks[learner]->referenceFF0,
                             blocks[learner]->intermediateFF,
                             (seq_len * blocks[learner]->ff_size_) >> 2);
#endif
#endif
    }

    std::cout << "Feed Forward 1" << std::endl;
    LinearLayer* ff1_layers[LearnerCount];
    uint32_t* ff1_inputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        ff1_layers[learner] = blocks[learner]->feedForward1;
        ff1_inputs[learner] = blocks[learner]->intermediateFF;
    }
    if (!tryGroupedCodebookDense(ff1_layers, ff1_inputs, outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->feedForward1->compute(
                seq_len, blocks[learner]->intermediateFF, outputs[learner]);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        const std::string ff1_label =
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff1_out.txt",
            outputs[learner],
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_ENABLE_DEBUG_PRINT
        printPackedPreview(ff1_label.c_str(), outputs[learner], (seq_len * blocks[learner]->input_dim_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceFF1,
                  blocks[learner]->referenceFF1 + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  0u);

        blocks[learner]->feedForward1Reference->compute(
            seq_len, blocks[learner]->referenceFF0, blocks[learner]->referenceFF1);

#if CFG_ENABLE_DEBUG_PRINT
        comparePackedBuffers(ff1_label.c_str(),
                             blocks[learner]->referenceFF1,
                             outputs[learner],
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif

        std::copy(blocks[learner]->referenceFF1,
                  blocks[learner]->referenceFF1 + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  blocks[learner]->referenceFinalOutput);
#endif
    }

    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(blocks[learner]->condense_out, outputs[learner]);
#else
        blocks[learner]->addNorm->compute(blocks[learner]->condense_out, outputs[learner]);
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "final_out.txt",
            outputs[learner],
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(
            blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFinalOutput);
#else
        blocks[learner]->addNorm->compute(
            blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFinalOutput);
#endif

#if CFG_ENABLE_DEBUG_PRINT
        const std::string label =
            "final_output_after_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceFinalOutput,
                             outputs[learner],
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif
#endif
    }

    runM5IfAvailable("m5 dumpresetstats");
}

#if CFG_FULL_INTERLEAVED_PIPELINE
void TransformerBlock::computeGroup4FullInterleaved(std::size_t seq_len,
                                                    TransformerBlock* blocks[4],
                                                    uint32_t* const inputs[4],
                                                    uint32_t* const outputs[4]) {
    runM5IfAvailable("m5 resetstats");

    std::string dump_dirs[4];
    std::size_t learner_ids[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        dump_dirs[learner] = blocks[learner]->dump_dir_;
        learner_ids[learner] = blocks[learner]->learner_idx_;
    }

    const std::size_t input_dim = blocks[0]->input_dim_;
    const std::size_t head_hidden_size = blocks[0]->head_hidden_size_;
    const std::size_t num_heads = blocks[0]->num_heads_;
    const std::size_t ff_size = blocks[0]->ff_size_;

    std::vector<int8_t> input_interleaved(seq_len * input_dim * 4u, 0);
    interleavePackedLearners4(seq_len, input_dim, inputs, input_interleaved.data());

    std::vector<int8_t> multihead_interleaved(
        seq_len * num_heads * head_hidden_size * 4u, 0);
    std::vector<int8_t> head_out_interleaved(
        seq_len * head_hidden_size * 4u, 0);

    for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        std::cout << "Head : " << head_idx << std::endl;

        SingleHeadSelfAttn* heads[4];
        for (std::size_t learner = 0; learner < 4u; learner++) {
            heads[learner] = blocks[learner]->selfatten_[head_idx];
        }

        std::fill(head_out_interleaved.begin(), head_out_interleaved.end(), 0);
        SingleHeadSelfAttn::computeInterleaved4D(
            seq_len,
            heads,
            input_interleaved.data(),
            head_out_interleaved.data());

        copyHeadToMultiheadInterleaved4D(
            head_out_interleaved.data(),
            multihead_interleaved.data(),
            seq_len,
            head_idx,
            head_hidden_size,
            num_heads);
    }

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "multihead_out.txt",
        multihead_interleaved.data(),
        seq_len,
        num_heads * head_hidden_size);

    std::cout << "Condense" << std::endl;
    LinearLayer* condense_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        condense_layers[learner] = blocks[learner]->condense;
    }

    std::vector<int8_t> condense_interleaved(seq_len * input_dim * 4u, 0);
    computeCodebookDenseInterleaved4D(
        "condense",
        condense_layers,
        seq_len,
        multihead_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* condense_references[4];
    uint32_t* condense_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        condense_references[learner] = blocks[learner]->condenseReference;
        condense_reference_outputs[learner] = blocks[learner]->referenceCondense;
    }
    compareInterleavedDenseReference4D(
        "condense_out",
        condense_references,
        condense_reference_outputs,
        learner_ids,
        seq_len,
        num_heads * head_hidden_size,
        input_dim,
        multihead_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "condense_out.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> condense_before_addnorm_interleaved = condense_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved4D(
        input_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference4D(
        "condense_out_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        input_interleaved.data(),
        condense_before_addnorm_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "after_attn_addnorm.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);

    runM5IfAvailable("m5 dumpresetstats");

    std::cout << "Feed Forward 0" << std::endl;
    LinearLayer* ff0_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff0_layers[learner] = blocks[learner]->feedForward0;
    }

    std::vector<int8_t> ff0_interleaved(seq_len * ff_size * 4u, 0);
    computeCodebookDenseInterleaved4D(
        "ff0",
        ff0_layers,
        seq_len,
        condense_interleaved.data(),
        ff0_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 4u; learner++) {
        const std::string ff0_label =
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview4D(
            ff0_label.c_str(),
            ff0_interleaved.data(),
            seq_len,
            ff_size,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff0_references[4];
    uint32_t* ff0_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff0_references[learner] = blocks[learner]->feedForward0Reference;
        ff0_reference_outputs[learner] = blocks[learner]->referenceFF0;
    }
    compareInterleavedDenseReference4D(
        "ffn0",
        ff0_references,
        ff0_reference_outputs,
        learner_ids,
        seq_len,
        input_dim,
        ff_size,
        condense_interleaved.data(),
        ff0_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "ff0_out.txt",
        ff0_interleaved.data(),
        seq_len,
        ff_size);

    std::cout << "Feed Forward 1" << std::endl;
    LinearLayer* ff1_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff1_layers[learner] = blocks[learner]->feedForward1;
    }

    std::vector<int8_t> ff1_interleaved(seq_len * input_dim * 4u, 0);
    computeCodebookDenseInterleaved4D(
        "ff1",
        ff1_layers,
        seq_len,
        ff0_interleaved.data(),
        ff1_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 4u; learner++) {
        const std::string ff1_label =
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview4D(
            ff1_label.c_str(),
            ff1_interleaved.data(),
            seq_len,
            input_dim,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff1_references[4];
    uint32_t* ff1_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff1_references[learner] = blocks[learner]->feedForward1Reference;
        ff1_reference_outputs[learner] = blocks[learner]->referenceFF1;
    }
    compareInterleavedDenseReference4D(
        "ffn1_pre_addnorm",
        ff1_references,
        ff1_reference_outputs,
        learner_ids,
        seq_len,
        ff_size,
        input_dim,
        ff0_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "ff1_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> ff1_before_addnorm_interleaved = ff1_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved4D(
        condense_interleaved.data(),
        ff1_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference4D(
        "final_output_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        condense_interleaved.data(),
        ff1_before_addnorm_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "final_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);

    packInterleavedLearners4(seq_len, input_dim, ff1_interleaved.data(), outputs);

    runM5IfAvailable("m5 dumpresetstats");
}
#endif

void TransformerBlock::computeGroup2(std::size_t seq_len,
                                     TransformerBlock* blocks[2],
                                     uint32_t* const inputs[2],
                                     uint32_t* const outputs[2]) {
    computeGroupImpl<2u>(seq_len, blocks, inputs, outputs);
}

void TransformerBlock::computeGroup4(std::size_t seq_len,
                                     TransformerBlock* blocks[4],
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]) {
    computeGroupImpl<4u>(seq_len, blocks, inputs, outputs);
}
