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
                                       std::size_t max_col,
                                       std::size_t learner_idx,
                                       std::string dump_dir) {
    head_idx_ = head_idx;
    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;
    input_dim_ = input_dim;
    learner_idx_ = learner_idx;
    dump_dir_ = dump_dir;

    const std::string q_name = "q_h" + std::to_string(head_idx_);
    const std::string k_name = "k_h" + std::to_string(head_idx_);
    const std::string v_name = "v_h" + std::to_string(head_idx_);

    auto q_bundle = LayerFactory::create(q_name, input_dim, head_hidden_size, weightVector[0], learner_idx_);
    auto k_bundle = LayerFactory::create(k_name, input_dim, head_hidden_size, weightVector[1], learner_idx_);
    auto v_bundle = LayerFactory::create(v_name, input_dim, head_hidden_size, weightVector[2], learner_idx_);

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

    // Debug for head 0
    // static bool dumped_input_h0 = false;
    // static bool dumped_q_h0_outputs = false;

    // const std::size_t rows_to_dump = std::min<std::size_t>(seq_len, 2);

    // if (head_idx_ == 0 && !dumped_input_h0) {
    //     dumped_input_h0 = true;

    //     std::size_t rows_to_dump = std::min<std::size_t>(seq_len, 2);

    //     std::cout << "\n===== DEBUG self attention input for head 0 =====\n";
    //     std::cout << "seq_len = " << seq_len << "\n";
    //     std::cout << "input_dim = " << input_dim_ << "\n";
    //     printPackedTensorAsPythonList("input_matrix_test", input, rows_to_dump, input_dim_);
    //     std::cout << "===== END DEBUG =====\n\n";
    // }
    
    query_layer_->compute(seq_len, input, query_layer_out_);
    key_layer_->compute(seq_len, input, key_layer_out_);
    value_layer_->compute(seq_len, input, value_layer_out_);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "q_h" + std::to_string(head_idx_) + ".txt",
        query_layer_out_,
        seq_len,
        head_hidden_size_);
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "k_h" + std::to_string(head_idx_) + ".txt",
        key_layer_out_,
        seq_len,
        head_hidden_size_);
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "v_h" + std::to_string(head_idx_) + ".txt",
        value_layer_out_,
        seq_len,
        head_hidden_size_);

#if CFG_USE_CODEBOOK_REFERENCE
    std::cout << "[DEBUG] CFG_USE_CODEBOOK_REFERENCE active in SingleHeadSelfAttn" << std::endl;
    std::fill(query_reference_out_, query_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(key_reference_out_, key_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(value_reference_out_, value_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);

    query_reference_->compute(seq_len, input, query_reference_out_);
    key_reference_->compute(seq_len, input, key_reference_out_);
    value_reference_->compute(seq_len, input, value_reference_out_);

    // Print q_h0 outputs only once, after both paths have been computed.
    // if (head_idx_ == 0 && !dumped_q_h0_outputs) {
    //     dumped_q_h0_outputs = true;

    //     std::cout << "\n===== DEBUG q_h0 outputs =====\n";
    //     printPackedTensorAsPythonList("q_h0_cpp_codebook", query_layer_out_, rows_to_dump, head_hidden_size_);
    //     printPackedTensorAsPythonList("q_h0_cpp_dense_ref", query_reference_out_, rows_to_dump, head_hidden_size_);
    //     std::cout << "===== END q_h0 OUTPUT DEBUG =====\n\n";
    // }

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

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "head_out_h" + std::to_string(head_idx_) + ".txt",
        output,
        seq_len,
        head_hidden_size_);
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

// Grouped self-attention entry used by TransformerBlock::computeGroup2/4().
// For Q/K/V projection we first try to fuse learners into one interleaved
// CodebookDense GEMM call. If that is not available, we fall back to normal
// per-learner LinearLayer::compute() calls.
template <std::size_t LearnerCount>
void SingleHeadSelfAttn::computeGroupImpl(std::size_t seq_len,
                                          SingleHeadSelfAttn** heads,
                                          uint32_t* const* inputs,
                                          uint32_t* const* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped self-attention is supported");

    LinearLayer* query_layers[LearnerCount];
    LinearLayer* key_layers[LearnerCount];
    LinearLayer* value_layers[LearnerCount];
    uint32_t* query_outputs[LearnerCount];
    uint32_t* key_outputs[LearnerCount];
    uint32_t* value_outputs[LearnerCount];

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        query_layers[learner] = heads[learner]->query_layer_;
        key_layers[learner] = heads[learner]->key_layer_;
        value_layers[learner] = heads[learner]->value_layer_;
        query_outputs[learner] = heads[learner]->query_layer_out_;
        key_outputs[learner] = heads[learner]->key_layer_out_;
        value_outputs[learner] = heads[learner]->value_layer_out_;
    }

    auto tryGroupedCodebookDense = [&](LinearLayer* const layers[LearnerCount],
                                       uint32_t* const dense_outputs[LearnerCount]) {
        if constexpr (LearnerCount == 2u) {
            return tryComputeGroupedCodebookDense2(layers, seq_len, inputs, dense_outputs);
        } else {
            return tryComputeGroupedCodebookDense4(layers, seq_len, inputs, dense_outputs);
        }
    };

    if (!tryGroupedCodebookDense(query_layers, query_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->query_layer_->compute(seq_len, inputs[learner], query_outputs[learner]);
        }
    }
    if (!tryGroupedCodebookDense(key_layers, key_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->key_layer_->compute(seq_len, inputs[learner], key_outputs[learner]);
        }
    }
    if (!tryGroupedCodebookDense(value_layers, value_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->value_layer_->compute(seq_len, inputs[learner], value_outputs[learner]);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        SingleHeadSelfAttn* self = heads[learner];

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "q_h" + std::to_string(self->head_idx_) + ".txt",
            self->query_layer_out_,
            seq_len,
            self->head_hidden_size_);
        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "k_h" + std::to_string(self->head_idx_) + ".txt",
            self->key_layer_out_,
            seq_len,
            self->head_hidden_size_);
        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "v_h" + std::to_string(self->head_idx_) + ".txt",
            self->value_layer_out_,
            seq_len,
            self->head_hidden_size_);

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(self->query_reference_out_,
                  self->query_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);
        std::fill(self->key_reference_out_,
                  self->key_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);
        std::fill(self->value_reference_out_,
                  self->value_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);

        self->query_reference_->compute(seq_len, inputs[learner], self->query_reference_out_);
        self->key_reference_->compute(seq_len, inputs[learner], self->key_reference_out_);
        self->value_reference_->compute(seq_len, inputs[learner], self->value_reference_out_);

        const std::string q_name =
            "q_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);
        const std::string k_name =
            "k_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);
        const std::string v_name =
            "v_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);

        comparePackedBuffers(q_name.c_str(),
                             self->query_reference_out_, self->query_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
        comparePackedBuffers(k_name.c_str(),
                             self->key_reference_out_, self->key_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
        comparePackedBuffers(v_name.c_str(),
                             self->value_reference_out_, self->value_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
#endif

#ifndef BWMA
        std::cout << "RWMA method" << std::endl;

        Transpose::transpose(self->key_layer_out_,
                             self->key_transposed_layer_out_,
                             self->head_hidden_size_,
                             self->pre_seq_len_);

#ifdef SIMD
        simdComputeRWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                        self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#else
        smmComputeRWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                       self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#endif

        self->softmax_->compute(self->attention_scores_, seq_len);

#ifdef SIMD
        simdComputeRWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                        seq_len, self->head_hidden_size_);
#else
        smmComputeRWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                       seq_len, self->head_hidden_size_);
#endif

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "head_out_h" + std::to_string(self->head_idx_) + ".txt",
            outputs[learner],
            seq_len,
            self->head_hidden_size_);
#else
        std::cout << "BWMA method" << std::endl;

        Transpose::transpose_rearranged(self->key_layer_out_, self->key_transposed_layer_out_,
                                        self->head_hidden_size_, self->pre_seq_len_,
                                        self->kernel_size_, self->max_col_);

#ifdef SIMD
        simdComputeBWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                        self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#else
        smmComputeBWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                       self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#endif

        self->softmax_->computeRearranged(self->attention_scores_, seq_len, self->kernel_size_);

#ifdef SIMD
        simdComputeBWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                        seq_len, self->head_hidden_size_);
#else
        smmComputeBWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                       seq_len, self->head_hidden_size_);
#endif
#endif

        self->softmax_->post_softmax(outputs[learner], seq_len, self->head_hidden_size_);
    }
}

void SingleHeadSelfAttn::computeGroup2(std::size_t seq_len,
                                       SingleHeadSelfAttn* heads[2],
                                       uint32_t* const inputs[2],
                                       uint32_t* const outputs[2]) {
    computeGroupImpl<2u>(seq_len, heads, inputs, outputs);
}

void SingleHeadSelfAttn::computeGroup4(std::size_t seq_len,
                                       SingleHeadSelfAttn* heads[4],
                                       uint32_t* const inputs[4],
                                       uint32_t* const outputs[4]) {
    computeGroupImpl<4u>(seq_len, heads, inputs, outputs);
}
