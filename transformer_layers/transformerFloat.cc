#include "transformerFloat.h"

#include "run_mode_config.h"

#if CFG_USE_FP32_TRANSFORMER

#include "../Full_NN/gemm_definitions/generated_codebook_registry.h"
#include "../Full_NN/gemm_definitions/input_matrix.h"
#include "../Full_NN/inc/gemm_exec.h"
#include "../transformer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Matrix = std::vector<float>;

constexpr float kLayerNormEps = 1.0e-5f;

std::size_t codebookSize(const GeneratedCodebookLayerView& view) {
    return static_cast<std::size_t>(1u) << view.bits_per_cb;
}

uint32_t packedIndex(const uint32_t* packed_row,
                     std::size_t elem_idx,
                     uint8_t bits_per_cb) {
    const std::size_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t mask = (1u << bits_per_cb) - 1u;
    const std::size_t word_idx = elem_idx / idxs_per_word;
    const std::size_t bit_offset = (elem_idx % idxs_per_word) * bits_per_cb;
    return (packed_row[word_idx] >> bit_offset) & mask;
}

const GeneratedCodebookLayerView& layerView(const std::string& name,
                                            std::size_t learner_count) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(name.c_str());
    if (view == nullptr) {
        throw std::runtime_error("FP32 transformer layer not found in registry: " + name);
    }
    if (view->n_learners < learner_count) {
        throw std::runtime_error(
            "FP32 transformer registry has fewer learners than requested for layer: " + name);
    }
    return *view;
}

const uint32_t* weightIdxForLearner(const GeneratedCodebookLayerView& view,
                                    std::size_t learner) {
    if (view.same_seq || view.weight_idx_by_learner == nullptr) {
        return view.weight_idx;
    }
    return view.weight_idx_by_learner + learner * getGeneratedWeightIdxCount(view);
}

const int8_t* codebookForLearner(const GeneratedCodebookLayerView& view,
                                 std::size_t learner) {
    if (view.codebooks_int8 == nullptr) {
        return view.codebook_int8;
    }
    return view.codebooks_int8 + learner * codebookSize(view);
}

const float* biasForLearner(const GeneratedCodebookLayerView& view,
                            std::size_t learner) {
    if (view.biases == nullptr) {
        return view.bias;
    }
    return view.biases + learner * view.output_size;
}

const float* codebookFp32ForLearner(const GeneratedCodebookLayerView& view,
                                    std::size_t learner,
                                    Matrix& scratch) {
#ifdef GENERATED_CODEBOOK_REGISTRY_HAS_FP32
    const float* codebook = getGeneratedCodebookFp32(&view, learner);
    if (codebook != nullptr) {
        return codebook;
    }
#endif

    const int8_t* codebook_int8 = codebookForLearner(view, learner);
    if (codebook_int8 == nullptr) {
        return nullptr;
    }

    const std::size_t cb_size = codebookSize(view);
    scratch.resize(cb_size);
    for (std::size_t idx = 0; idx < cb_size; idx++) {
        scratch[idx] = static_cast<float>(codebook_int8[idx]);
    }
    return scratch.data();
}

const float* codebookFp32Interleaved(const GeneratedCodebookLayerView& view,
                                     std::size_t learner_count,
                                     Matrix& scratch) {
#ifdef GENERATED_CODEBOOK_REGISTRY_HAS_FP32
    const float* codebook = getGeneratedCodebookFp32Interleaved(&view);
    if (codebook != nullptr) {
        return codebook;
    }
#endif

    const std::size_t cb_size = codebookSize(view);
    scratch.assign(cb_size * learner_count, 0.0f);
    for (std::size_t cb_idx = 0; cb_idx < cb_size; cb_idx++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            const int8_t* learner_codebook = codebookForLearner(view, learner);
            if (learner_codebook == nullptr) {
                return nullptr;
            }
            scratch[cb_idx * learner_count + learner] =
                static_cast<float>(learner_codebook[cb_idx]);
        }
    }
    return scratch.data();
}

const float* biasFp32Interleaved(const GeneratedCodebookLayerView& view,
                                 std::size_t learner_count,
                                 Matrix& scratch) {
    if (view.bias_interleaved != nullptr) {
        return view.bias_interleaved;
    }
    if (view.biases == nullptr && view.bias == nullptr) {
        return nullptr;
    }

    scratch.assign(view.output_size * learner_count, 0.0f);
    for (std::size_t out_idx = 0; out_idx < view.output_size; out_idx++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            const float* learner_bias = biasForLearner(view, learner);
            scratch[out_idx * learner_count + learner] =
                (learner_bias == nullptr) ? 0.0f : learner_bias[out_idx];
        }
    }
    return scratch.data();
}

void saveFloatMatrixText(const std::string& path,
                         const float* data,
                         std::size_t rows,
                         std::size_t cols) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::cout << path << " Not saved" << std::endl;
        return;
    }

    fout << std::setprecision(9);
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            if (col != 0u) {
                fout << " ";
            }
            fout << data[row * cols + col];
        }
        fout << "\n";
    }
}

void dumpFloatMatrixIfEnabled(const std::string& dump_dir,
                              const std::string& filename,
                              const float* data,
                              std::size_t rows,
                              std::size_t cols) {
#if CFG_GEM5_PROFILE_REGIONS
    (void)dump_dir;
    (void)filename;
    (void)data;
    (void)rows;
    (void)cols;
#else
    if (dump_dir.empty()) {
        return;
    }
    std::filesystem::create_directories(dump_dir);
    saveFloatMatrixText(dump_dir + "/" + filename, data, rows, cols);
#endif
}

void dumpInterleavedFloatMatrices(const std::vector<std::string>& dump_dirs,
                                  const std::string& filename,
                                  const float* interleaved,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t learner_count) {
    Matrix tmp(rows * cols, 0.0f);
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        for (std::size_t row = 0; row < rows; row++) {
            for (std::size_t col = 0; col < cols; col++) {
                tmp[row * cols + col] =
                    interleaved[((row * cols) + col) * learner_count + learner];
            }
        }
        dumpFloatMatrixIfEnabled(dump_dirs[learner], filename, tmp.data(), rows, cols);
    }
}

void printFloatPreview(const std::string& label,
                       const float* data,
                       std::size_t size) {
#if CFG_ENABLE_DEBUG_PRINT
    const std::size_t preview = std::min<std::size_t>(size, 8u);
    std::cout << label << " preview (first " << preview << " float values):" << std::endl;
    std::cout << std::setprecision(6);
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << label << "[" << idx << "] = " << data[idx] << std::endl;
    }
#else
    (void)label;
    (void)data;
    (void)size;
#endif
}

void printInterleavedFloatPreview(const std::string& label,
                                  const float* interleaved,
                                  std::size_t logical_size,
                                  std::size_t learner_count,
                                  std::size_t learner) {
#if CFG_ENABLE_DEBUG_PRINT
    const std::size_t preview = std::min<std::size_t>(logical_size, 8u);
    std::cout << label << " preview (first " << preview << " float values):" << std::endl;
    std::cout << std::setprecision(6);
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << label << "[" << idx << "] = "
                  << interleaved[idx * learner_count + learner] << std::endl;
    }
#else
    (void)label;
    (void)interleaved;
    (void)logical_size;
    (void)learner_count;
    (void)learner;
#endif
}

void dense1D(const std::string& layer_name,
             std::size_t learner,
             const float* input,
             std::size_t rows,
             Matrix& output) {
    const auto& view = layerView(layer_name, learner + 1u);
    const uint32_t* weight_idx = weightIdxForLearner(view, learner);
    Matrix codebook_scratch;
    const float* codebook = codebookFp32ForLearner(view, learner, codebook_scratch);
    const float* bias = biasForLearner(view, learner);

    if (weight_idx == nullptr || codebook == nullptr) {
        throw std::runtime_error("FP32 transformer registry entry is incomplete: " + layer_name);
    }

    output.assign(rows * view.output_size, 0.0f);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(rows);
    layer.input_size = static_cast<uint16_t>(view.input_size);
    layer.output_size = static_cast<uint16_t>(view.output_size);
    layer.n_words_row = static_cast<uint16_t>(view.n_words_row);

#ifdef SIMD
    gemm_exec_compact_sve(layer, input, weight_idx, codebook, bias, output.data(), view.bits_per_cb);
#else
    gemm_exec_compact(layer, input, weight_idx, codebook, bias, output.data(), view.bits_per_cb);
#endif
}

void denseInterleaved(const std::string& layer_name,
                      std::size_t learner_count,
                      const float* input_interleaved,
                      std::size_t rows,
                      Matrix& output_interleaved) {
    const auto& view = layerView(layer_name, learner_count);
    output_interleaved.assign(rows * view.output_size * learner_count, 0.0f);

    Matrix codebook_scratch;
    Matrix bias_scratch;
    const float* codebook_interleaved =
        codebookFp32Interleaved(view, learner_count, codebook_scratch);
    const float* bias_interleaved =
        biasFp32Interleaved(view, learner_count, bias_scratch);
    if (codebook_interleaved == nullptr) {
        throw std::runtime_error("FP32 transformer registry entry is incomplete: " + layer_name);
    }

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(rows);
    layer.input_size = static_cast<uint16_t>(view.input_size);
    layer.output_size = static_cast<uint16_t>(view.output_size);
    layer.n_words_row = static_cast<uint16_t>(view.n_words_row);

    if (learner_count == 2u) {
        if (!view.same_seq) {
            throw std::runtime_error("FP32 2D interleaved path currently requires SAME_SEQ: " + layer_name);
        }
#ifdef SIMD
        gemm_exec_compact_sve_fp32_interleaved_2D_same_seq(
            layer, input_interleaved, view.weight_idx, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
        gemm_exec_compact_fp32_interleaved_2D_same_seq(
            layer, input_interleaved, view.weight_idx, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
        return;
    }

    if (learner_count == 4u) {
        if (view.same_seq) {
#ifdef SIMD
            gemm_exec_compact_sve_fp32_interleaved_4D_same_seq(
                layer, input_interleaved, view.weight_idx, codebook_interleaved,
                bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
            gemm_exec_compact_fp32_interleaved_4D_same_seq(
                layer, input_interleaved, view.weight_idx, codebook_interleaved,
                bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
            return;
        }

        const uint32_t* weight_idx_interleaved =
            getGeneratedCodebookWeightIdxInterleaved(&view);
        if (weight_idx_interleaved == nullptr) {
            throw std::runtime_error("FP32 4D diff-seq path is missing interleaved indexes: " + layer_name);
        }
#ifdef SIMD
        gemm_exec_compact_sve_fp32_interleaved_4D_diff_seq(
            layer, input_interleaved, weight_idx_interleaved, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
        gemm_exec_compact_fp32_interleaved_4D_diff_seq(
            layer, input_interleaved, weight_idx_interleaved, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
        return;
    }

    throw std::runtime_error("FP32 interleaved path supports only 2D or 4D learners: " + layer_name);
}

void addNorm1D(const float* residual,
               float* candidate,
               std::size_t rows,
               std::size_t cols) {
    for (std::size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        float* out_row = candidate + row * cols;
        const float* residual_row = residual + row * cols;

        for (std::size_t col = 0; col < cols; col++) {
            out_row[col] += residual_row[col];
            sum += out_row[col];
        }

        const float mean = sum / static_cast<float>(cols);
        float variance = 0.0f;
        for (std::size_t col = 0; col < cols; col++) {
            const float diff = out_row[col] - mean;
            variance += diff * diff;
        }

        const float inv_sd = 1.0f / std::sqrt(variance / static_cast<float>(cols) + kLayerNormEps);
        for (std::size_t col = 0; col < cols; col++) {
            out_row[col] = (out_row[col] - mean) * inv_sd;
        }
    }
}

void addNormInterleaved(const float* residual_interleaved,
                        float* candidate_interleaved,
                        std::size_t rows,
                        std::size_t cols,
                        std::size_t learner_count) {
    for (std::size_t row = 0; row < rows; row++) {
        std::vector<float> sum(learner_count, 0.0f);

        for (std::size_t col = 0; col < cols; col++) {
            float* out_slot = candidate_interleaved + ((row * cols + col) * learner_count);
            const float* residual_slot =
                residual_interleaved + ((row * cols + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] += residual_slot[learner];
                sum[learner] += out_slot[learner];
            }
        }

        std::vector<float> mean(learner_count, 0.0f);
        std::vector<float> variance(learner_count, 0.0f);
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            mean[learner] = sum[learner] / static_cast<float>(cols);
        }

        for (std::size_t col = 0; col < cols; col++) {
            const float* out_slot = candidate_interleaved + ((row * cols + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                const float diff = out_slot[learner] - mean[learner];
                variance[learner] += diff * diff;
            }
        }

        for (std::size_t col = 0; col < cols; col++) {
            float* out_slot = candidate_interleaved + ((row * cols + col) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                const float inv_sd =
                    1.0f / std::sqrt(variance[learner] / static_cast<float>(cols) + kLayerNormEps);
                out_slot[learner] = (out_slot[learner] - mean[learner]) * inv_sd;
            }
        }
    }
}

void softmaxRows(float* scores,
                 std::size_t rows,
                 std::size_t cols,
                 float scale) {
    for (std::size_t row = 0; row < rows; row++) {
        float* row_ptr = scores + row * cols;
        float max_value = row_ptr[0] * scale;
        for (std::size_t col = 1; col < cols; col++) {
            max_value = std::max(max_value, row_ptr[col] * scale);
        }

        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] = std::exp(row_ptr[col] * scale - max_value);
            sum += row_ptr[col];
        }

        const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
        for (std::size_t col = 0; col < cols; col++) {
            row_ptr[col] *= inv_sum;
        }
    }
}

void softmaxRowsInterleaved(float* scores,
                            std::size_t rows,
                            std::size_t cols,
                            std::size_t learner_count,
                            float scale) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            float max_value = scores[(row * cols) * learner_count + learner] * scale;
            for (std::size_t col = 1; col < cols; col++) {
                const float value =
                    scores[((row * cols) + col) * learner_count + learner] * scale;
                max_value = std::max(max_value, value);
            }

            float sum = 0.0f;
            for (std::size_t col = 0; col < cols; col++) {
                float& slot = scores[((row * cols) + col) * learner_count + learner];
                slot = std::exp(slot * scale - max_value);
                sum += slot;
            }

            const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);
            for (std::size_t col = 0; col < cols; col++) {
                scores[((row * cols) + col) * learner_count + learner] *= inv_sum;
            }
        }
    }
}

void attentionFromQKV1D(const float* query,
                        const float* key,
                        const float* value,
                        std::size_t seq_len,
                        std::size_t head_hidden_size,
                        Matrix& scores,
                        Matrix& output) {
    scores.assign(seq_len * seq_len, 0.0f);
    output.assign(seq_len * head_hidden_size, 0.0f);

    for (std::size_t row = 0; row < seq_len; row++) {
        for (std::size_t col = 0; col < seq_len; col++) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < head_hidden_size; k++) {
                acc += query[row * head_hidden_size + k] *
                       key[col * head_hidden_size + k];
            }
            scores[row * seq_len + col] = acc;
        }
    }

    softmaxRows(scores.data(), seq_len, seq_len,
                1.0f / std::sqrt(static_cast<float>(head_hidden_size)));

    for (std::size_t row = 0; row < seq_len; row++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            float acc = 0.0f;
            for (std::size_t key_idx = 0; key_idx < seq_len; key_idx++) {
                acc += scores[row * seq_len + key_idx] *
                       value[key_idx * head_hidden_size + feature];
            }
            output[row * head_hidden_size + feature] = acc;
        }
    }
}

void attention1D(std::size_t learner,
                 std::size_t head_idx,
                 const float* input,
                 const std::string& dump_dir,
                 Matrix& output) {
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix scores;

    dense1D("q_h" + std::to_string(head_idx), learner, input, D_SEQ, query);
    dense1D("k_h" + std::to_string(head_idx), learner, input, D_SEQ, key);
    dense1D("v_h" + std::to_string(head_idx), learner, input, D_SEQ, value);

    dumpFloatMatrixIfEnabled(dump_dir, "q_h" + std::to_string(head_idx) + ".txt",
                             query.data(), D_SEQ, D_Q);
    dumpFloatMatrixIfEnabled(dump_dir, "k_h" + std::to_string(head_idx) + ".txt",
                             key.data(), D_SEQ, D_Q);
    dumpFloatMatrixIfEnabled(dump_dir, "v_h" + std::to_string(head_idx) + ".txt",
                             value.data(), D_SEQ, D_Q);

    attentionFromQKV1D(query.data(), key.data(), value.data(), D_SEQ, D_Q, scores, output);

    dumpFloatMatrixIfEnabled(dump_dir, "softmax_qk_h" + std::to_string(head_idx) + ".txt",
                             scores.data(), D_SEQ, D_SEQ);
    dumpFloatMatrixIfEnabled(dump_dir, "softmax_v_pre_post_h" + std::to_string(head_idx) + ".txt",
                             output.data(), D_SEQ, D_Q);
    dumpFloatMatrixIfEnabled(dump_dir, "head_out_h" + std::to_string(head_idx) + ".txt",
                             output.data(), D_SEQ, D_Q);
    dumpFloatMatrixIfEnabled(dump_dir, "head_out_post_h" + std::to_string(head_idx) + ".txt",
                             output.data(), D_SEQ, D_Q);
}

void matmulInterleavedTransposedRhs(const float* lhs_interleaved,
                                    const float* rhs_rows_interleaved,
                                    std::size_t lhs_rows,
                                    std::size_t rhs_rows,
                                    std::size_t k_elems,
                                    std::size_t learner_count,
                                    Matrix& output_interleaved) {
    output_interleaved.assign(lhs_rows * rhs_rows * learner_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_rows; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < k_elems; k++) {
                    const float lhs =
                        lhs_interleaved[((row * k_elems) + k) * learner_count + learner];
                    const float rhs =
                        rhs_rows_interleaved[((col * k_elems) + k) * learner_count + learner];
                    acc += lhs * rhs;
                }
                output_interleaved[((row * rhs_rows) + col) * learner_count + learner] = acc;
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

void copyHeadToMultiheadInterleaved(const float* head_interleaved,
                                    float* multihead_interleaved,
                                    std::size_t learner_count,
                                    std::size_t head_idx) {
    const std::size_t multihead_cols = NUM_HEAD * D_Q;
    const std::size_t head_col_base = head_idx * D_Q;

    for (std::size_t seq = 0; seq < D_SEQ; seq++) {
        for (std::size_t feature = 0; feature < D_Q; feature++) {
            const float* in_slot = head_interleaved + ((seq * D_Q + feature) * learner_count);
            float* out_slot =
                multihead_interleaved + ((seq * multihead_cols + head_col_base + feature) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void interleaveInputs(const std::vector<Matrix>& inputs,
                      std::size_t rows,
                      std::size_t cols,
                      Matrix& output_interleaved) {
    const std::size_t learner_count = inputs.size();
    output_interleaved.assign(rows * cols * learner_count, 0.0f);
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                output_interleaved[((row * cols) + col) * learner_count + learner] =
                    inputs[learner][row * cols + col];
            }
        }
    }
}

void deinterleaveOutputs(const Matrix& input_interleaved,
                         std::size_t rows,
                         std::size_t cols,
                         std::size_t learner_count,
                         std::vector<Matrix>& outputs) {
    outputs.assign(learner_count, Matrix(rows * cols, 0.0f));
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                outputs[learner][row * cols + col] =
                    input_interleaved[((row * cols) + col) * learner_count + learner];
            }
        }
    }
}

void transformerBlock1D(std::size_t learner,
                        const Matrix& input,
                        const std::string& dump_dir,
                        Matrix& output) {
    Matrix multihead(D_SEQ * D_MODEL, 0.0f);

    for (std::size_t head = 0; head < NUM_HEAD; head++) {
        std::cout << "Head : " << head << std::endl;
        Matrix head_out;
        attention1D(learner, head, input.data(), dump_dir, head_out);
        for (std::size_t seq = 0; seq < D_SEQ; seq++) {
            for (std::size_t feature = 0; feature < D_Q; feature++) {
                multihead[seq * D_MODEL + head * D_Q + feature] =
                    head_out[seq * D_Q + feature];
            }
        }
    }

    dumpFloatMatrixIfEnabled(dump_dir, "multihead_out.txt", multihead.data(), D_SEQ, D_MODEL);

    std::cout << "Condense" << std::endl;
    Matrix condense;
    dense1D("condense", learner, multihead.data(), D_SEQ, condense);
    dumpFloatMatrixIfEnabled(dump_dir, "condense_out.txt", condense.data(), D_SEQ, D_MODEL);

    std::cout << "Add Norm" << std::endl;
    addNorm1D(input.data(), condense.data(), D_SEQ, D_MODEL);
    dumpFloatMatrixIfEnabled(dump_dir, "after_attn_addnorm.txt", condense.data(), D_SEQ, D_MODEL);

    std::cout << "Feed Forward 0" << std::endl;
    Matrix ff0;
    dense1D("ff0", learner, condense.data(), D_SEQ, ff0);
    dumpFloatMatrixIfEnabled(dump_dir, "ff0_out.txt", ff0.data(), D_SEQ, D_FF);
    printFloatPreview("ffn0_learner" + std::to_string(learner), ff0.data(), ff0.size());

    std::cout << "Feed Forward 1" << std::endl;
    Matrix ff1;
    dense1D("ff1", learner, ff0.data(), D_SEQ, ff1);
    dumpFloatMatrixIfEnabled(dump_dir, "ff1_out.txt", ff1.data(), D_SEQ, D_MODEL);
    printFloatPreview("ffn1_pre_addnorm_learner" + std::to_string(learner), ff1.data(), ff1.size());

    std::cout << "Add Norm" << std::endl;
    output = ff1;
    addNorm1D(condense.data(), output.data(), D_SEQ, D_MODEL);
    dumpFloatMatrixIfEnabled(dump_dir, "final_out.txt", output.data(), D_SEQ, D_MODEL);
}

void transformerBlockGroupedFullInterleaved(std::size_t learner_count,
                                            const Matrix& input,
                                            const std::vector<std::string>& dump_dirs,
                                            std::vector<Matrix>& outputs) {
    std::vector<Matrix> per_learner_inputs(learner_count, input);
    Matrix input_interleaved;
    interleaveInputs(per_learner_inputs, D_SEQ, D_MODEL, input_interleaved);

    Matrix multihead_interleaved(D_SEQ * D_MODEL * learner_count, 0.0f);

    for (std::size_t head = 0; head < NUM_HEAD; head++) {
        std::cout << "Head : " << head << std::endl;

        Matrix query;
        Matrix key;
        Matrix value;
        denseInterleaved("q_h" + std::to_string(head), learner_count,
                         input_interleaved.data(), D_SEQ, query);
        denseInterleaved("k_h" + std::to_string(head), learner_count,
                         input_interleaved.data(), D_SEQ, key);
        denseInterleaved("v_h" + std::to_string(head), learner_count,
                         input_interleaved.data(), D_SEQ, value);

        dumpInterleavedFloatMatrices(dump_dirs, "q_h" + std::to_string(head) + ".txt",
                                     query.data(), D_SEQ, D_Q, learner_count);
        dumpInterleavedFloatMatrices(dump_dirs, "k_h" + std::to_string(head) + ".txt",
                                     key.data(), D_SEQ, D_Q, learner_count);
        dumpInterleavedFloatMatrices(dump_dirs, "v_h" + std::to_string(head) + ".txt",
                                     value.data(), D_SEQ, D_Q, learner_count);

        Matrix attention_scores;
        matmulInterleavedTransposedRhs(query.data(), key.data(), D_SEQ, D_SEQ, D_Q,
                                       learner_count, attention_scores);
        softmaxRowsInterleaved(attention_scores.data(), D_SEQ, D_SEQ, learner_count,
                               1.0f / std::sqrt(static_cast<float>(D_Q)));

        dumpInterleavedFloatMatrices(dump_dirs, "softmax_qk_h" + std::to_string(head) + ".txt",
                                     attention_scores.data(), D_SEQ, D_SEQ, learner_count);

        Matrix head_out;
        matmulInterleavedRows(attention_scores.data(), value.data(), D_SEQ, D_Q, D_SEQ,
                              learner_count, head_out);
        dumpInterleavedFloatMatrices(dump_dirs, "softmax_v_pre_post_h" + std::to_string(head) + ".txt",
                                     head_out.data(), D_SEQ, D_Q, learner_count);
        dumpInterleavedFloatMatrices(dump_dirs, "head_out_h" + std::to_string(head) + ".txt",
                                     head_out.data(), D_SEQ, D_Q, learner_count);
        dumpInterleavedFloatMatrices(dump_dirs, "head_out_post_h" + std::to_string(head) + ".txt",
                                     head_out.data(), D_SEQ, D_Q, learner_count);
        copyHeadToMultiheadInterleaved(head_out.data(), multihead_interleaved.data(),
                                       learner_count, head);
    }

    dumpInterleavedFloatMatrices(dump_dirs, "multihead_out.txt", multihead_interleaved.data(),
                                 D_SEQ, D_MODEL, learner_count);

    std::cout << "Condense" << std::endl;
    Matrix condense;
    denseInterleaved("condense", learner_count, multihead_interleaved.data(), D_SEQ, condense);
    dumpInterleavedFloatMatrices(dump_dirs, "condense_out.txt", condense.data(),
                                 D_SEQ, D_MODEL, learner_count);

    std::cout << "Add Norm" << std::endl;
    addNormInterleaved(input_interleaved.data(), condense.data(), D_SEQ, D_MODEL, learner_count);
    dumpInterleavedFloatMatrices(dump_dirs, "after_attn_addnorm.txt", condense.data(),
                                 D_SEQ, D_MODEL, learner_count);

    std::cout << "Feed Forward 0" << std::endl;
    Matrix ff0;
    denseInterleaved("ff0", learner_count, condense.data(), D_SEQ, ff0);
    dumpInterleavedFloatMatrices(dump_dirs, "ff0_out.txt", ff0.data(),
                                 D_SEQ, D_FF, learner_count);
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        printInterleavedFloatPreview("ffn0_learner" + std::to_string(learner),
                                     ff0.data(), D_SEQ * D_FF, learner_count, learner);
    }

    std::cout << "Feed Forward 1" << std::endl;
    Matrix ff1;
    denseInterleaved("ff1", learner_count, ff0.data(), D_SEQ, ff1);
    dumpInterleavedFloatMatrices(dump_dirs, "ff1_out.txt", ff1.data(),
                                 D_SEQ, D_MODEL, learner_count);
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        printInterleavedFloatPreview("ffn1_pre_addnorm_learner" + std::to_string(learner),
                                     ff1.data(), D_SEQ * D_MODEL, learner_count, learner);
    }

    std::cout << "Add Norm" << std::endl;
    addNormInterleaved(condense.data(), ff1.data(), D_SEQ, D_MODEL, learner_count);
    dumpInterleavedFloatMatrices(dump_dirs, "final_out.txt", ff1.data(),
                                 D_SEQ, D_MODEL, learner_count);

    deinterleaveOutputs(ff1, D_SEQ, D_MODEL, learner_count, outputs);
}

void transformerBlockGroupedNonInterleaved(std::size_t learner_count,
                                           const Matrix& input,
                                           const std::vector<std::string>& dump_dirs,
                                           std::vector<Matrix>& outputs) {
    outputs.assign(learner_count, Matrix(D_SEQ * D_MODEL, 0.0f));
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        transformerBlock1D(learner, input, dump_dirs[learner], outputs[learner]);
    }
}

Matrix loadInputMatrix() {
    static_assert(GEMM_M == D_SEQ, "input_matrix.h GEMM_M must match D_SEQ");
    static_assert(GEMM_K == D_MODEL, "input_matrix.h GEMM_K must match D_MODEL");
    return Matrix(input_matrix, input_matrix + (D_SEQ * D_MODEL));
}

} // namespace

namespace TransformerFloat {

void run(std::size_t learner_count, const std::string& c_output_root) {
    if (learner_count == 0u) {
        learner_count = 1u;
    }

    std::cout << "CFG_USE_FP32_TRANSFORMER = 1" << std::endl;
    std::cout << "FP32 Transformer uses generated codebook registry; Dense reference is disabled."
              << std::endl;

    Matrix input = loadInputMatrix();
    std::vector<std::string> dump_dirs(learner_count);
    for (std::size_t learner = 0; learner < learner_count; learner++) {
        dump_dirs[learner] = c_output_root + "/learner" + std::to_string(learner);
        std::filesystem::create_directories(dump_dirs[learner]);
        dumpFloatMatrixIfEnabled(dump_dirs[learner], "input_matrix.txt",
                                 input.data(), D_SEQ, D_MODEL);
    }

    std::vector<Matrix> outputs;
    if (learner_count == 2u || learner_count == 4u) {
#if CFG_FULL_INTERLEAVED_PIPELINE
        transformerBlockGroupedFullInterleaved(learner_count, input, dump_dirs, outputs);
#else
        transformerBlockGroupedNonInterleaved(learner_count, input, dump_dirs, outputs);
#endif
    } else {
        outputs.assign(learner_count, Matrix());
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            if (learner_count > 1u) {
                std::cout << "\n=============== LEARNER " << learner
                          << " ===============\n" << std::endl;
            }
            transformerBlock1D(learner, input, dump_dirs[learner], outputs[learner]);
        }
    }

    for (std::size_t learner = 0; learner < outputs.size(); learner++) {
        dumpFloatMatrixIfEnabled(dump_dirs[learner], "output.txt",
                                 outputs[learner].data(), D_SEQ, D_MODEL);
    }
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
