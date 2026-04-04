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
                                   std::size_t maxCol) {
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;

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
                maxCol));
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
        weightVector[num_heads * 3]);

    condense = condense_bundle.main;

    auto ff0_bundle = LayerFactory::create(
        "ff0",
        input_dim,
        ff_size,
        weightVector[num_heads * 3 + 1]);

    feedForward0 = ff0_bundle.main;

    auto ff1_bundle = LayerFactory::create(
        "ff1",
        ff_size,
        input_dim,
        weightVector[num_heads * 3 + 2]);

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

    std::cout << "Condense" << std::endl;
    condense->compute(seq_len, multihead_for_condense, condense_out);

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
