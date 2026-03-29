#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

//#include <cstdint>
#include "debuggerFunctions.h"

#include "layerFactory.h"

// SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
//                                        uint32_t **weightVector, std::size_t kernel_dim, std::size_t max_col) {

//     pre_seq_len_ = pre_seq_len;
//     head_hidden_size_ = head_hidden_size;
//     kernel_size_ = kernel_dim;
//     max_col_ = max_col;

//     query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
//     key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
//     value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
//     softmax = new Softmax();

//     query_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
//     key_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
//     key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
//     value_layer_out = new uint32_t[pre_seq_len * head_hidden_size >> 2]();
//     attention_scores = new uint32_t[pre_seq_len * pre_seq_len >> 2]();
// }

// SingleHeadSelfAttn::~SingleHeadSelfAttn() {

//     delete[] query_layer_out;
//     delete[] key_layer_out;
//     delete[] key_transposed_layer_out;
//     delete[] value_layer_out;
//     delete[] attention_scores;

//     delete query_layer;
//     delete key_layer;
//     delete value_layer;
//     delete softmax;
// }

// void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
//     query_layer->compute(seq_len, input, query_layer_out);
//     key_layer->compute(seq_len, input, key_layer_out);
//     value_layer->compute(seq_len, input, value_layer_out);


// #ifdef BWMA
//     std::cout << "BWMA method" << std::endl;
//     Transpose::transpose_rearranged(key_layer_out, key_transposed_layer_out, head_hidden_size_,
//                                     pre_seq_len_, kernel_size_, max_col_);
// #ifdef SIMD
//     simdComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
//                           head_hidden_size_, seq_len);
// #else
//     smmComputeBWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
//                    seq_len);
// #endif
//     softmax->computeRearranged(attention_scores, seq_len, kernel_size_);
// #ifdef SIMD
//     simdComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
// #else
//     smmComputeBWMA(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);
// #endif
// #else
//     std::cout<< "RWMA method" << std::endl;
//     Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_,
//                                     pre_seq_len_);
// #ifdef SIMD
//     simdComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
//                head_hidden_size_, seq_len);
// #else
//     smmComputeRWMA(seq_len, query_layer_out, attention_scores, key_transposed_layer_out,
//                 head_hidden_size_, seq_len);
// #endif
//     softmax->compute(attention_scores, seq_len);
// #ifdef SIMD
//     simdComputeRWMA(seq_len, attention_scores, output, value_layer_out,
//                seq_len, head_hidden_size_);
// #else
//     smmComputeRWMA(seq_len, attention_scores, output, value_layer_out,
//                 seq_len, head_hidden_size_);
// #endif
// #endif

//     softmax->post_softmax(output, seq_len, head_hidden_size_);
// }

/* Flexible configuration of the number of heads and codebooked GEMM */

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t head_idx,
                                       std::size_t pre_seq_len,
                                       std::size_t input_dim,
                                       std::size_t head_hidden_size,
                                       uint32_t** weightVector,
                                       std::size_t kernel_dim,
                                       std::size_t max_col) {
    head_idx_ = head_idx;
    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;

    const std::string q_name = "q_h" + std::to_string(head_idx_);
    const std::string k_name = "k_h" + std::to_string(head_idx_);
    const std::string v_name = "v_h" + std::to_string(head_idx_);

    auto q_bundle = LayerFactory::create(q_name, input_dim, head_hidden_size, weightVector[0]);
    auto k_bundle = LayerFactory::create(k_name, input_dim, head_hidden_size, weightVector[1]);
    auto v_bundle = LayerFactory::create(v_name, input_dim, head_hidden_size, weightVector[2]);

    query_layer_ = q_bundle.main;
    key_layer_ = k_bundle.main;
    value_layer_ = v_bundle.main;

#if CFG_USE_CODEBOOK_REFERENCE
    query_reference_ = q_bundle.reference;
    key_reference_ = k_bundle.reference;
    value_reference_ = v_bundle.reference;

    query_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    value_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
#endif

    softmax_ = new Softmax();

    query_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_transposed_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    value_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    attention_scores_ = new uint32_t[(pre_seq_len * pre_seq_len) >> 2]();
}

SingleHeadSelfAttn::~SingleHeadSelfAttn() {
    delete[] query_layer_out_;
    delete[] key_layer_out_;
    delete[] key_transposed_layer_out_;
    delete[] value_layer_out_;
    delete[] attention_scores_;

#if CFG_USE_CODEBOOK_REFERENCE
    delete[] query_reference_out_;
    delete[] key_reference_out_;
    delete[] value_reference_out_;

    delete query_reference_;
    delete key_reference_;
    delete value_reference_;
#endif

    delete query_layer_;
    delete key_layer_;
    delete value_layer_;
    delete softmax_;
}

void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t* input, uint32_t* output) {
    query_layer_->compute(seq_len, input, query_layer_out_);
    key_layer_->compute(seq_len, input, key_layer_out_);
    value_layer_->compute(seq_len, input, value_layer_out_);

#if CFG_USE_CODEBOOK_REFERENCE
    std::cout << "[DEBUG] CFG_USE_CODEBOOK_REFERENCE active in SingleHeadSelfAttn" << std::endl;
    std::fill(query_reference_out_, query_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(key_reference_out_, key_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(value_reference_out_, value_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);

    query_reference_->compute(seq_len, input, query_reference_out_);
    key_reference_->compute(seq_len, input, key_reference_out_);
    value_reference_->compute(seq_len, input, value_reference_out_);

    comparePackedBuffers(("q_h" + std::to_string(head_idx_)).c_str(),
                         query_reference_out_, query_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);

    comparePackedBuffers(("k_h" + std::to_string(head_idx_)).c_str(),
                         key_reference_out_, key_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);

    comparePackedBuffers(("v_h" + std::to_string(head_idx_)).c_str(),
                         value_reference_out_, value_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);
#endif

#ifndef BWMA
    std::cout << "RWMA method" << std::endl;

    Transpose::transpose(key_layer_out_, key_transposed_layer_out_,
                         head_hidden_size_, pre_seq_len_);

#ifdef SIMD
    simdComputeRWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                    head_hidden_size_, seq_len);
#else
    smmComputeRWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                   head_hidden_size_, seq_len);
#endif

    softmax_->compute(attention_scores_, seq_len);

#ifdef SIMD
    simdComputeRWMA(seq_len, attention_scores_, output, value_layer_out_,
                    seq_len, head_hidden_size_);
#else
    smmComputeRWMA(seq_len, attention_scores_, output, value_layer_out_,
                   seq_len, head_hidden_size_);
#endif
#else
    std::cout << "BWMA method" << std::endl;

    Transpose::transpose_rearranged(key_layer_out_, key_transposed_layer_out_,
                                    head_hidden_size_, pre_seq_len_, kernel_size_, max_col_);

#ifdef SIMD
    simdComputeBWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                    head_hidden_size_, seq_len);
#else
    smmComputeBWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                   head_hidden_size_, seq_len);
#endif

    softmax_->computeRearranged(attention_scores_, seq_len, kernel_size_);

#ifdef SIMD
    simdComputeBWMA(seq_len, attention_scores_, output, value_layer_out_,
                    seq_len, head_hidden_size_);
#else
    smmComputeBWMA(seq_len, attention_scores_, output, value_layer_out_,
                   seq_len, head_hidden_size_);
#endif
#endif

    softmax_->post_softmax(output, seq_len, head_hidden_size_);
}