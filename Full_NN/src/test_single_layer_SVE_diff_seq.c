#include "../gemm_definitions/generated_codebook_registry.h"

#include <gemm_exec.h>

#ifndef SIMD
#error "test_single_layer_SVE_diff_seq requires -DSIMD."
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t unpackIndex(const uint32_t *packed_row,
                            std::size_t elem_idx,
                            uint8_t bits_per_cb) {
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask =
        (bits_per_cb >= 32u) ? UINT32_MAX : ((1u << bits_per_cb) - 1u);
    const uint32_t word_idx = static_cast<uint32_t>(elem_idx / idxs_per_word);
    const uint32_t shift =
        static_cast<uint32_t>((elem_idx % idxs_per_word) * bits_per_cb);
    return (packed_row[word_idx] >> shift) & idx_mask;
}

static std::vector<int8_t> makeLearnerInput(std::size_t learner,
                                            std::size_t rows,
                                            std::size_t cols) {
    std::vector<int8_t> input(rows * cols, 0);
    for (std::size_t r = 0; r < rows; r++) {
        for (std::size_t c = 0; c < cols; c++) {
            const int value =
                static_cast<int>(((learner + 1u) * 5u + r * 3u + c * 2u) % 9u) - 4;
            input[r * cols + c] = static_cast<int8_t>(value);
        }
    }
    return input;
}

static std::vector<int8_t> buildInterleavedInput(
    const std::vector<std::vector<int8_t>>& inputs,
    std::size_t seq_len,
    std::size_t input_size) {
    std::vector<int8_t> interleaved(seq_len * input_size * 4u, 0);
    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t in_idx = 0; in_idx < input_size; in_idx++) {
            const std::size_t base = ((seq * input_size) + in_idx) * 4u;
            for (std::size_t learner = 0; learner < 4u; learner++) {
                interleaved[base + learner] =
                    inputs[learner][seq * input_size + in_idx];
            }
        }
    }
    return interleaved;
}

static std::vector<uint32_t> buildInterleavedWeightIdx(
    const std::vector<const uint32_t*>& weight_idx_by_learner,
    std::size_t weight_idx_count) {
    std::vector<uint32_t> interleaved(weight_idx_count * 4u, 0);
    for (std::size_t word = 0; word < weight_idx_count; word++) {
        for (std::size_t learner = 0; learner < 4u; learner++) {
            interleaved[word * 4u + learner] = weight_idx_by_learner[learner][word];
        }
    }
    return interleaved;
}

static std::vector<int8_t> buildInterleavedCodebook(
    const std::vector<const int8_t*>& codebooks_by_learner,
    std::size_t codebook_size) {
    std::vector<int8_t> interleaved(codebook_size * 4u, 0);
    for (std::size_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (std::size_t learner = 0; learner < 4u; learner++) {
            interleaved[cb_idx * 4u + learner] =
                codebooks_by_learner[learner][cb_idx];
        }
    }
    return interleaved;
}

static std::vector<int8_t> buildDenseWeights(const uint32_t *weight_idx,
                                             const int8_t *codebook,
                                             std::size_t output_size,
                                             std::size_t input_size,
                                             std::size_t n_words_row,
                                             uint8_t bits_per_cb) {
    std::vector<int8_t> dense(output_size * input_size, 0);
    for (std::size_t out_idx = 0; out_idx < output_size; out_idx++) {
        const uint32_t *packed_row = &weight_idx[out_idx * n_words_row];
        for (std::size_t in_idx = 0; in_idx < input_size; in_idx++) {
            const uint32_t cb_idx = unpackIndex(packed_row, in_idx, bits_per_cb);
            dense[out_idx * input_size + in_idx] = codebook[cb_idx];
        }
    }
    return dense;
}

static bool compareFlat(const std::vector<int32_t>& expected,
                        const std::vector<int32_t>& actual,
                        const std::string& label,
                        std::size_t seq_len,
                        std::size_t output_size) {
    if (expected.size() != actual.size()) {
        std::cout << label << "_size_match = false\n";
        return false;
    }

    for (std::size_t idx = 0; idx < expected.size(); idx++) {
        if (expected[idx] != actual[idx]) {
            const std::size_t seq = idx / output_size;
            const std::size_t out_idx = idx % output_size;
            std::cout << label << "_match = false\n";
            std::cout << label << "_first_mismatch_seq = " << seq << "\n";
            std::cout << label << "_first_mismatch_out = " << out_idx << "\n";
            std::cout << label << "_expected = " << expected[idx] << "\n";
            std::cout << label << "_actual = " << actual[idx] << "\n";
            return false;
        }
    }

    (void)seq_len;
    std::cout << label << "_match = true\n";
    return true;
}

static bool compareInterleaved(const std::vector<std::vector<int32_t>>& expected,
                               const std::vector<int32_t>& actual_interleaved,
                               const std::string& label,
                               std::size_t seq_len,
                               std::size_t output_size) {
    for (std::size_t learner = 0; learner < 4u; learner++) {
        for (std::size_t seq = 0; seq < seq_len; seq++) {
            for (std::size_t out_idx = 0; out_idx < output_size; out_idx++) {
                const std::size_t flat_idx = seq * output_size + out_idx;
                const std::size_t inter_idx = flat_idx * 4u + learner;
                if (expected[learner][flat_idx] != actual_interleaved[inter_idx]) {
                    std::cout << label << "_match = false\n";
                    std::cout << label << "_first_mismatch_learner = " << learner << "\n";
                    std::cout << label << "_first_mismatch_seq = " << seq << "\n";
                    std::cout << label << "_first_mismatch_out = " << out_idx << "\n";
                    std::cout << label << "_expected = " << expected[learner][flat_idx] << "\n";
                    std::cout << label << "_actual = " << actual_interleaved[inter_idx] << "\n";
                    return false;
                }
            }
        }
    }

    std::cout << label << "_match = true\n";
    return true;
}

static void printPreview(const std::vector<int32_t>& values,
                         const std::string& label,
                         std::size_t count) {
    const std::size_t preview = values.size() < count ? values.size() : count;
    std::cout << label << " = [";
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << values[idx];
        if (idx + 1 != preview) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

static bool runSingleLayerDiffSeq(const std::string& layer_name,
                                  std::size_t seq_len) {
    const GeneratedCodebookLayerView *view =
        findGeneratedCodebookLayer(layer_name.c_str());
    if (view == nullptr) {
        throw std::runtime_error("Layer not found in generated registry: " + layer_name);
    }
    if (view->n_learners < 4u) {
        throw std::runtime_error("Layer does not provide 4 learners: " + layer_name);
    }
    if ((view->input_size > UINT16_MAX) || (view->output_size > UINT16_MAX) ||
        (view->n_words_row > UINT16_MAX) || (seq_len > UINT16_MAX)) {
        throw std::runtime_error("Layer dimensions exceed gemm_t uint16_t fields");
    }

    gemm_t gemm_layer;
    gemm_layer.seq_len = static_cast<uint16_t>(seq_len);
    gemm_layer.input_size = static_cast<uint16_t>(view->input_size);
    gemm_layer.output_size = static_cast<uint16_t>(view->output_size);
    gemm_layer.n_words_row = static_cast<uint16_t>(view->n_words_row);

    const std::size_t codebook_size = getGeneratedCodebookSize(*view);
    const std::size_t weight_idx_count = getGeneratedWeightIdxCount(*view);

    std::vector<const uint32_t*> weight_idx_ptrs;
    std::vector<const int8_t*> codebook_ptrs;
    std::vector<std::vector<int8_t>> inputs;
    std::vector<std::vector<int8_t>> dense_weights;
    std::vector<std::vector<int32_t>> dense_reference;
    std::vector<std::vector<int32_t>> compact_reference;

    weight_idx_ptrs.reserve(4u);
    codebook_ptrs.reserve(4u);
    inputs.reserve(4u);
    dense_weights.reserve(4u);
    dense_reference.reserve(4u);
    compact_reference.reserve(4u);

    for (std::size_t learner = 0; learner < 4u; learner++) {
        const uint32_t *weight_idx = getGeneratedCodebookWeightIdx(view, learner);
        const int8_t *codebook = getGeneratedCodebookInt8(view, learner);
        if ((weight_idx == nullptr) || (codebook == nullptr)) {
            throw std::runtime_error("Registry entry is missing learner-specific data");
        }

        weight_idx_ptrs.push_back(weight_idx);
        codebook_ptrs.push_back(codebook);
        inputs.push_back(makeLearnerInput(learner, seq_len, view->input_size));
        dense_weights.push_back(buildDenseWeights(weight_idx,
                                                 codebook,
                                                 view->output_size,
                                                 view->input_size,
                                                 view->n_words_row,
                                                 view->bits_per_cb));
        dense_reference.emplace_back(seq_len * view->output_size, 0);
        compact_reference.emplace_back(seq_len * view->output_size, 0);

        gemm_exec_noCB_int(gemm_layer,
                           inputs.back().data(),
                           dense_weights.back().data(),
                           nullptr,
                           dense_reference.back().data());
        gemm_exec_compact_int(gemm_layer,
                              inputs.back().data(),
                              weight_idx,
                              codebook,
                              nullptr,
                              compact_reference.back().data(),
                              view->bits_per_cb);
    }

    std::vector<int8_t> input_interleaved =
        buildInterleavedInput(inputs, seq_len, view->input_size);
    std::vector<uint32_t> weight_idx_interleaved =
        buildInterleavedWeightIdx(weight_idx_ptrs, weight_idx_count);
    std::vector<int8_t> codebook_interleaved =
        buildInterleavedCodebook(codebook_ptrs, codebook_size);
    std::vector<int32_t> scalar_interleaved(seq_len * view->output_size * 4u, 0);
    std::vector<int32_t> sve_interleaved(seq_len * view->output_size * 4u, 0);

    const uint32_t *registry_weight_idx_interleaved =
        getGeneratedCodebookWeightIdxInterleaved(view);
    bool registry_weight_idx_match = (registry_weight_idx_interleaved != nullptr);
    if (registry_weight_idx_interleaved != nullptr) {
        for (std::size_t idx = 0; idx < weight_idx_interleaved.size(); idx++) {
            if (weight_idx_interleaved[idx] != registry_weight_idx_interleaved[idx]) {
                registry_weight_idx_match = false;
                break;
            }
        }
    }

    const int8_t *registry_codebook_interleaved =
        getGeneratedCodebookInt8Interleaved(view);
    bool registry_codebook_match = (registry_codebook_interleaved != nullptr);
    if (registry_codebook_interleaved != nullptr) {
        for (std::size_t idx = 0; idx < codebook_interleaved.size(); idx++) {
            if (codebook_interleaved[idx] != registry_codebook_interleaved[idx]) {
                registry_codebook_match = false;
                break;
            }
        }
    }

    gemm_exec_compact_int_interleaved_4D_diff_seq(gemm_layer,
                                                  input_interleaved.data(),
                                                  weight_idx_interleaved.data(),
                                                  codebook_interleaved.data(),
                                                  nullptr,
                                                  scalar_interleaved.data(),
                                                  view->bits_per_cb);

    gemm_exec_compact_int_sve_interleaved_4D_diff_seq(gemm_layer,
                                                      input_interleaved.data(),
                                                      weight_idx_interleaved.data(),
                                                      codebook_interleaved.data(),
                                                      nullptr,
                                                      sve_interleaved.data(),
                                                      view->bits_per_cb);

    std::cout << "layer_name = \"" << layer_name << "\"\n";
    std::cout << "seq_len = " << seq_len << "\n";
    std::cout << "input_size = " << view->input_size << "\n";
    std::cout << "output_size = " << view->output_size << "\n";
    std::cout << "n_words_row = " << view->n_words_row << "\n";
    std::cout << "bits_per_cb = " << static_cast<int>(view->bits_per_cb) << "\n";
    std::cout << "n_learners_in_registry = " << view->n_learners << "\n";
    std::cout << "same_seq = " << (view->same_seq ? "true" : "false") << "\n";
    std::cout << "registry_weight_idx_interleaved_match = "
              << (registry_weight_idx_match ? "true" : "false") << "\n";
    std::cout << "registry_codebook_interleaved_match = "
              << (registry_codebook_match ? "true" : "false") << "\n";
    printPreview(dense_reference[0], "dense_ref_learner0_row0", 8u);
    printPreview(compact_reference[0], "compact_ref_learner0_row0", 8u);

    bool ok = true;
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ok = compareFlat(dense_reference[learner],
                         compact_reference[learner],
                         "dense_vs_compact_learner" + std::to_string(learner),
                         seq_len,
                         view->output_size) && ok;
    }
    ok = compareInterleaved(dense_reference,
                            scalar_interleaved,
                            "dense_vs_interleaved_scalar",
                            seq_len,
                            view->output_size) && ok;
    ok = compareInterleaved(dense_reference,
                            sve_interleaved,
                            "dense_vs_interleaved_sve",
                            seq_len,
                            view->output_size) && ok;
    ok = compareInterleaved(compact_reference,
                            sve_interleaved,
                            "compact_vs_interleaved_sve",
                            seq_len,
                            view->output_size) && ok;

    return ok;
}

int main(int argc, char **argv) {
    try {
        std::string layer_name = "q_h0";
        std::size_t seq_len = 2;

        if (argc >= 2) {
            layer_name = argv[1];
        }
        if (argc >= 3) {
            seq_len = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
        }

        return runSingleLayerDiffSeq(layer_name, seq_len) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

/*
/home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++ \
  -x c++ -std=c++17 -O2 -Wall -march=armv8-a+sve -DSIMD \
  Full_NN/src/test_single_layer_SVE_diff_seq.c \
  Full_NN/src/gemm_exec.c \
  Full_NN/src/gemm_SVE.c \
  -IFull_NN/inc -IFull_NN/gemm_definitions \
  -o /tmp/test_single_layer_SVE_diff_seq_aarch64

/home/thu/opt/qemu-sve/bin/qemu-aarch64 \
  -cpu max,sve=on,sve-default-vector-length=16 \
  -L /home/thu/miniforge3/envs/gem5_env/bin/../aarch64-conda-linux-gnu/sysroot \
  /tmp/test_single_layer_SVE_diff_seq_aarch64 q_h0 2

SYSROOT=$(/home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++ -print-sysroot)
/home/thu/opt/qemu-sve/bin/qemu-aarch64 \
  -cpu max,sve=on,sve-default-vector-length=16 \
  -L "$SYSROOT" \
  /tmp/test_single_layer_SVE_diff_seq_aarch64_static q_h0 2 \
  < /tmp/test_single_layer_SVE_diff_seq_aarch64_static

*/
