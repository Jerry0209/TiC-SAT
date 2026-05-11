#include "floatSelfAttention.h"

#include "floatDump.h"
#include "run_mode_config.h"

#if CFG_USE_FP32_TRANSFORMER

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace TransformerFloat {

namespace {

void matmulTransposedRhs(const float* lhs,
                         const float* rhs_rows,
                         std::size_t lhs_rows,
                         std::size_t rhs_rows_count,
                         std::size_t k_elems,
                         Matrix& output) {
    output.assign(lhs_rows * rhs_rows_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_rows_count; col++) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < k_elems; k++) {
                acc += lhs[row * k_elems + k] * rhs_rows[col * k_elems + k];
            }
            output[row * rhs_rows_count + col] = acc;
        }
    }
}

void matmulRows(const float* lhs,
                const float* rhs_rows,
                std::size_t lhs_rows,
                std::size_t rhs_cols,
                std::size_t k_elems,
                Matrix& output) {
    output.assign(lhs_rows * rhs_cols, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < k_elems; k++) {
                acc += lhs[row * k_elems + k] * rhs_rows[k * rhs_cols + col];
            }
            output[row * rhs_cols + col] = acc;
        }
    }
}

void matmulInterleavedTransposedRhs(const float* lhs_interleaved,
                                    const float* rhs_rows_interleaved,
                                    std::size_t lhs_rows,
                                    std::size_t rhs_rows_count,
                                    std::size_t k_elems,
                                    std::size_t learner_count,
                                    Matrix& output_interleaved) {
    output_interleaved.assign(lhs_rows * rhs_rows_count * learner_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_rows_count; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < k_elems; k++) {
                    const float lhs =
                        lhs_interleaved[((row * k_elems) + k) * learner_count + learner];
                    const float rhs =
                        rhs_rows_interleaved[((col * k_elems) + k) * learner_count + learner];
                    acc += lhs * rhs;
                }
                output_interleaved[((row * rhs_rows_count) + col) * learner_count + learner] =
                    acc;
            }
        }
    }
}

void matmulInterleavedRows(const float* lhs_interleaved,
                           const float* rhs_rows_interleaved,
                           std::size_t lhs_rows,
                           std::size_t rhs_cols,
                           std::size_t k_elems,
                           std::size_t learner_count,
                           Matrix& output_interleaved) {
    output_interleaved.assign(lhs_rows * rhs_cols * learner_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < k_elems; k++) {
                    const float lhs =
                        lhs_interleaved[((row * k_elems) + k) * learner_count + learner];
                    const float rhs =
                        rhs_rows_interleaved[((k * rhs_cols) + col) * learner_count + learner];
                    acc += lhs * rhs;
                }
                output_interleaved[((row * rhs_cols) + col) * learner_count + learner] = acc;
            }
        }
    }
}

} // namespace

FloatSingleHeadSelfAttn::FloatSingleHeadSelfAttn(std::size_t head_idx,
                                                 std::size_t pre_seq_len,
                                                 std::size_t input_dim,
                                                 std::size_t head_hidden_size,
                                                 std::size_t learner_idx,
                                                 std::string dump_dir)
    : head_idx_(head_idx),
      pre_seq_len_(pre_seq_len),
      input_dim_(input_dim),
      head_hidden_size_(head_hidden_size),
      learner_idx_(learner_idx),
      dump_dir_(std::move(dump_dir)),
      query_layer_("q_h" + std::to_string(head_idx), learner_idx),
      key_layer_("k_h" + std::to_string(head_idx), learner_idx),
      value_layer_("v_h" + std::to_string(head_idx), learner_idx) {}

void FloatSingleHeadSelfAttn::compute(std::size_t seq_len,
                                      const float* input,
                                      Matrix& output) const {
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix scores;

    query_layer_.compute(seq_len, input, query);
    key_layer_.compute(seq_len, input, key);
    value_layer_.compute(seq_len, input, value);

    const std::string head_suffix = std::to_string(head_idx_);
    dumpFloatMatrixIfEnabled(dump_dir_, "q_h" + head_suffix + ".txt",
                             query.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "k_h" + head_suffix + ".txt",
                             key.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "v_h" + head_suffix + ".txt",
                             value.data(), seq_len, head_hidden_size_);

    matmulTransposedRhs(query.data(), key.data(), seq_len, seq_len, head_hidden_size_, scores);
    dumpFloatMatrixIfEnabled(dump_dir_, "qk_scores_h" + head_suffix + ".txt",
                             scores.data(), seq_len, seq_len);

    softmax_.compute(scores.data(), seq_len, seq_len,
                     1.0f / std::sqrt(static_cast<float>(head_hidden_size_)));
    dumpFloatMatrixIfEnabled(dump_dir_, "softmax_qk_h" + head_suffix + ".txt",
                             scores.data(), seq_len, seq_len);

    matmulRows(scores.data(), value.data(), seq_len, head_hidden_size_, seq_len, output);
    dumpFloatMatrixIfEnabled(dump_dir_, "softmax_v_pre_post_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "head_out_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "head_out_post_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
}

template <std::size_t LearnerCount>
void FloatSingleHeadSelfAttn::computeGroupImpl(std::size_t seq_len,
                                               FloatSingleHeadSelfAttn** heads,
                                               const float* const* inputs,
                                               Matrix* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped FP32 attention is supported");

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        heads[learner]->compute(seq_len, inputs[learner], outputs[learner]);
    }
}

void FloatSingleHeadSelfAttn::computeGroup2(std::size_t seq_len,
                                            FloatSingleHeadSelfAttn* heads[2],
                                            const float* const inputs[2],
                                            Matrix outputs[2]) {
    computeGroupImpl<2u>(seq_len, heads, inputs, outputs);
}

void FloatSingleHeadSelfAttn::computeGroup4(std::size_t seq_len,
                                            FloatSingleHeadSelfAttn* heads[4],
                                            const float* const inputs[4],
                                            Matrix outputs[4]) {
    computeGroupImpl<4u>(seq_len, heads, inputs, outputs);
}

template <std::size_t LearnerCount>
void FloatSingleHeadSelfAttn::computeInterleavedImpl(std::size_t seq_len,
                                                     FloatSingleHeadSelfAttn** heads,
                                                     const float* input_interleaved,
                                                     Matrix& output_interleaved) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner interleaved FP32 attention is supported");

    std::vector<std::string> dump_dirs(LearnerCount);
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dump_dirs[learner] = heads[learner]->dump_dir_;
    }

    const std::size_t head_hidden_size = heads[0]->head_hidden_size_;
    const std::string head_suffix = std::to_string(heads[0]->head_idx_);

    Matrix query;
    Matrix key;
    Matrix value;
    heads[0]->query_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, query);
    heads[0]->key_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, key);
    heads[0]->value_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, value);

    dumpInterleavedFloatMatrices(dump_dirs, "q_h" + head_suffix + ".txt",
                                 query.data(), seq_len, head_hidden_size, LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "k_h" + head_suffix + ".txt",
                                 key.data(), seq_len, head_hidden_size, LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "v_h" + head_suffix + ".txt",
                                 value.data(), seq_len, head_hidden_size, LearnerCount);

    Matrix scores;
    matmulInterleavedTransposedRhs(query.data(), key.data(), seq_len, seq_len,
                                   head_hidden_size, LearnerCount, scores);
    dumpInterleavedFloatMatrices(dump_dirs, "qk_scores_h" + head_suffix + ".txt",
                                 scores.data(), seq_len, seq_len, LearnerCount);

    heads[0]->softmax_.computeInterleaved(
        scores.data(), seq_len, seq_len, LearnerCount,
        1.0f / std::sqrt(static_cast<float>(head_hidden_size)));
    dumpInterleavedFloatMatrices(dump_dirs, "softmax_qk_h" + head_suffix + ".txt",
                                 scores.data(), seq_len, seq_len, LearnerCount);

    matmulInterleavedRows(scores.data(), value.data(), seq_len, head_hidden_size,
                          seq_len, LearnerCount, output_interleaved);
    dumpInterleavedFloatMatrices(dump_dirs, "softmax_v_pre_post_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "head_out_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "head_out_post_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
}

void FloatSingleHeadSelfAttn::computeInterleaved2D(std::size_t seq_len,
                                                   FloatSingleHeadSelfAttn* heads[2],
                                                   const float* input_interleaved,
                                                   Matrix& output_interleaved) {
    computeInterleavedImpl<2u>(seq_len, heads, input_interleaved, output_interleaved);
}

void FloatSingleHeadSelfAttn::computeInterleaved4D(std::size_t seq_len,
                                                   FloatSingleHeadSelfAttn* heads[4],
                                                   const float* input_interleaved,
                                                   Matrix& output_interleaved) {
    computeInterleavedImpl<4u>(seq_len, heads, input_interleaved, output_interleaved);
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
