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


namespace {

// Run m5 command only if gem5 helper is available.
void runM5IfAvailable(const char* command) {
    if (std::system("command -v m5 >/dev/null 2>&1") == 0) {
        std::system(command);
    }
}

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

void TransformerBlock::computeGroup4(std::size_t seq_len,
                                     TransformerBlock* blocks[4],
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]) {
    runM5IfAvailable("m5 resetstats");

    for (std::size_t n = 0; n < blocks[0]->num_heads_; ++n) {
        std::cout << "Head : " << n << std::endl;

        SingleHeadSelfAttn* heads[4] = {
            blocks[0]->selfatten_[n],
            blocks[1]->selfatten_[n],
            blocks[2]->selfatten_[n],
            blocks[3]->selfatten_[n],
        };
        uint32_t* head_outputs[4] = {
            blocks[0]->multihead_out + n * ((seq_len * blocks[0]->head_hidden_size_) >> 2),
            blocks[1]->multihead_out + n * ((seq_len * blocks[1]->head_hidden_size_) >> 2),
            blocks[2]->multihead_out + n * ((seq_len * blocks[2]->head_hidden_size_) >> 2),
            blocks[3]->multihead_out + n * ((seq_len * blocks[3]->head_hidden_size_) >> 2),
        };

        SingleHeadSelfAttn::computeGroup4(seq_len, heads, inputs, head_outputs);
    }

    uint32_t* multihead_for_condense[4] = {nullptr, nullptr, nullptr, nullptr};
    for (std::size_t learner = 0; learner < 4; learner++) {
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
    LinearLayer* condense_layers[4] = {
        blocks[0]->condense,
        blocks[1]->condense,
        blocks[2]->condense,
        blocks[3]->condense,
    };
    uint32_t* condense_outputs[4] = {
        blocks[0]->condense_out,
        blocks[1]->condense_out,
        blocks[2]->condense_out,
        blocks[3]->condense_out,
    };
    if (!tryComputeGroupedCodebookDense4(condense_layers, seq_len, multihead_for_condense, condense_outputs)) {
        for (std::size_t learner = 0; learner < 4; learner++) {
            blocks[learner]->condense->compute(
                seq_len, multihead_for_condense[learner], blocks[learner]->condense_out);
        }
    }

    for (std::size_t learner = 0; learner < 4; learner++) {
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
    for (std::size_t learner = 0; learner < 4; learner++) {
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
    LinearLayer* ff0_layers[4] = {
        blocks[0]->feedForward0,
        blocks[1]->feedForward0,
        blocks[2]->feedForward0,
        blocks[3]->feedForward0,
    };
    uint32_t* ff0_outputs[4] = {
        blocks[0]->intermediateFF,
        blocks[1]->intermediateFF,
        blocks[2]->intermediateFF,
        blocks[3]->intermediateFF,
    };
    uint32_t* ff0_inputs[4] = {
        blocks[0]->condense_out,
        blocks[1]->condense_out,
        blocks[2]->condense_out,
        blocks[3]->condense_out,
    };
    if (!tryComputeGroupedCodebookDense4(ff0_layers, seq_len, ff0_inputs, ff0_outputs)) {
        for (std::size_t learner = 0; learner < 4; learner++) {
            blocks[learner]->feedForward0->compute(
                seq_len, blocks[learner]->condense_out, blocks[learner]->intermediateFF);
        }
    }

    for (std::size_t learner = 0; learner < 4; learner++) {
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
    LinearLayer* ff1_layers[4] = {
        blocks[0]->feedForward1,
        blocks[1]->feedForward1,
        blocks[2]->feedForward1,
        blocks[3]->feedForward1,
    };
    uint32_t* ff1_inputs[4] = {
        blocks[0]->intermediateFF,
        blocks[1]->intermediateFF,
        blocks[2]->intermediateFF,
        blocks[3]->intermediateFF,
    };
    if (!tryComputeGroupedCodebookDense4(ff1_layers, seq_len, ff1_inputs, outputs)) {
        for (std::size_t learner = 0; learner < 4; learner++) {
            blocks[learner]->feedForward1->compute(
                seq_len, blocks[learner]->intermediateFF, outputs[learner]);
        }
    }

    for (std::size_t learner = 0; learner < 4; learner++) {
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
    for (std::size_t learner = 0; learner < 4; learner++) {
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
