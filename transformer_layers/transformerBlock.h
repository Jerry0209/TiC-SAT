//
// Created by alireza on 3/2/22.
//

#include <cstddef>
#include <cstdint>

#include "selfattention.h"
#include "addNorm.h"
#include "dense.h"
#include "linearLayer.h"

#ifndef FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
#define FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H

class TransformerBlock{
public:
    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, uint32_t ** weightVector,
                     std::size_t kernelDim, std::size_t maxCol);

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    SingleHeadSelfAttn* selfatten[16];
    uint32_t* multihead_out;
    uint32_t* condense_out;
    uint32_t* intermediateFF;
    uint32_t* intermediateFFBlockWise;
    AddNormalize* addNorm;
    // Dense* condense;
    // Dense* feedForward0;
    // Dense* feedForward1;
    LinearLayer* condense;
    LinearLayer* feedForward0;
    LinearLayer* feedForward1;
#ifdef CFG_USE_CODEBOOK_GEMM
    LinearLayer* condenseReference; 
    LinearLayer* feedForward0Reference;
    LinearLayer* feedForward1Reference;
    uint32_t* referenceCondense;
    uint32_t* referenceFF0;
    uint32_t* referenceFF1;
#endif

#ifndef BWMA
    uint32_t* multihead_out_reshape;
#endif

};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
