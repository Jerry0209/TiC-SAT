#include "../gemm_definitions/generated_codebook_registry.h"

#include <gemm_exec.h>
#include <gemm_SVE.h>

#ifndef SIMD
#error "test_single_layer_SVE requires -DSIMD and Full_NN/src/gemm_SVE.c in the build."
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int8_t> makeDefaultInput(std::size_t rows, std::size_t cols) {
    std::vector<int8_t> input(rows * cols, 0);

    for (std::size_t c = 0; c < cols; c++) {
        input[c] = static_cast<int8_t>(static_cast<int>(c % 8) - 4);
    }

    if (rows >= 2) {
        for (std::size_t c = 0; c < cols; c++) {
            input[cols + c] = static_cast<int8_t>(static_cast<int>(c % 5) - 2);
        }
    }

    for (std::size_t r = 2; r < rows; r++) {
        for (std::size_t c = 0; c < cols; c++) {
            input[r * cols + c] =
                static_cast<int8_t>(static_cast<int>((r + c) % 7) - 3);
        }
    }

    return input;
}

static bool compareOutputs(const std::vector<int32_t>& expected,
                           const std::vector<int32_t>& actual) {
    if (expected.size() != actual.size()) {
        std::cout << "sve_vs_scalar_size_match = false\n";
        return false;
    }

    for (std::size_t idx = 0; idx < expected.size(); idx++) {
        if (expected[idx] != actual[idx]) {
            std::cout << "sve_vs_scalar_match = false\n";
            std::cout << "first_mismatch_index = " << idx << "\n";
            std::cout << "expected_scalar = " << expected[idx] << "\n";
            std::cout << "actual_sve = " << actual[idx] << "\n";
            return false;
        }
    }

    std::cout << "sve_vs_scalar_match = true\n";
    return true;
}

static void printPreview(const std::vector<int32_t>& values,
                         std::size_t cols,
                         const std::string& label) {
    const std::size_t preview = (cols < 8) ? cols : 8;
    std::cout << label << "_row0_first_" << preview << " = [";
    for (std::size_t idx = 0; idx < preview; idx++) {
        std::cout << values[idx];
        if (idx + 1 != preview) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

static bool runSingleLayerSVE(const std::string& layer_name, std::size_t seq_len) {
    const GeneratedCodebookLayerView *layer =
        findGeneratedCodebookLayer(layer_name.c_str());
    if (layer == nullptr) {
        throw std::runtime_error("Layer not found in generated registry: " + layer_name);
    }
    if (layer->codebook_int8 == nullptr) {
        throw std::runtime_error("Layer has no int8 codebook: " + layer_name);
    }
    if ((layer->input_size > UINT16_MAX) || (layer->output_size > UINT16_MAX) ||
        (layer->n_words_row > UINT16_MAX) || (seq_len > UINT16_MAX)) {
        throw std::runtime_error("Layer dimensions exceed gemm_t uint16_t fields");
    }

    gemm_t gemm_layer;
    gemm_layer.seq_len = static_cast<uint16_t>(seq_len);
    gemm_layer.input_size = static_cast<uint16_t>(layer->input_size);
    gemm_layer.output_size = static_cast<uint16_t>(layer->output_size);
    gemm_layer.n_words_row = static_cast<uint16_t>(layer->n_words_row);

    std::vector<int8_t> input = makeDefaultInput(seq_len, layer->input_size);
    std::vector<int32_t> scalar_out(seq_len * layer->output_size, 0);
    std::vector<int32_t> sve_out(seq_len * layer->output_size, 0);

    gemm_exec_compact_int(gemm_layer,
                          input.data(),
                          layer->weight_idx,
                          layer->codebook_int8,
                          nullptr,
                          scalar_out.data(),
                          layer->bits_per_cb);

    gemm_exec_compact_int_sve(gemm_layer,
                              input.data(),
                              layer->weight_idx,
                              layer->codebook_int8,
                              nullptr,
                              sve_out.data(),
                              layer->bits_per_cb);

    std::cout << "layer_name = \"" << layer_name << "\"\n";
    std::cout << "seq_len = " << seq_len << "\n";
    std::cout << "input_size = " << layer->input_size << "\n";
    std::cout << "output_size = " << layer->output_size << "\n";
    std::cout << "n_words_row = " << layer->n_words_row << "\n";
    std::cout << "bits_per_cb = " << static_cast<int>(layer->bits_per_cb) << "\n";
    printPreview(scalar_out, layer->output_size, "scalar_acc");
    printPreview(sve_out, layer->output_size, "sve_acc");

    return compareOutputs(scalar_out, sve_out);
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

        return runSingleLayerSVE(layer_name, seq_len) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

// Example build from /home/thu/TiC-SAT:
//   aarch64-conda-linux-gnu-g++ -x c++ -std=c++17 -O2 -Wall -march=armv8-a+sve -DSIMD
//   Full_NN/src/test_single_layer_SVE.c
//   Full_NN/src/gemm_exec.c
//   Full_NN/src/gemm_SVE.c
//   -IFull_NN/inc
//   -IFull_NN/gemm_definitions
//   -o /tmp/test_single_layer_SVE_aarch64
//
// SYSROOT=$(aarch64-conda-linux-gnu-g++ -print-sysroot)
// Run on SVE-capable hardware, gem5, or a qemu-aarch64 build with SVE enabled.
// qemu-aarch64 -L "$SYSROOT" /tmp/test_single_layer_SVE_aarch64 q_h0 2


/*
/home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++ \
  -x c++ -std=c++17 -O2 -Wall -march=armv8-a+sve -DSIMD \
  Full_NN/src/test_single_layer_SVE.c \
  Full_NN/src/gemm_exec.c \
  Full_NN/src/gemm_SVE.c \
  -IFull_NN/inc -IFull_NN/gemm_definitions \
  -o /tmp/test_single_layer_SVE_aarch64

/home/thu/opt/qemu-sve/bin/qemu-aarch64 \
  -cpu max,sve=on,sve-default-vector-length=16 \
  -L /home/thu/miniforge3/envs/gem5_env/bin/../aarch64-conda-linux-gnu/sysroot \
  /tmp/test_single_layer_SVE_aarch64 q_h0 2 \
  < /tmp/test_single_layer_SVE_aarch64
*/


/* 
/home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++ \
  -x c++ -std=c++17 -O2 -Wall -march=armv8-a+sve -DSIMD \
  Full_NN/src/test_single_layer_SVE.c \
  Full_NN/src/gemm_exec.c \
  Full_NN/src/gemm_SVE.c \
  -IFull_NN/inc \
  -IFull_NN/gemm_definitions \
  -o /tmp/test_single_layer_SVE_aarch64


  /home/thu/opt/qemu-sve/bin/qemu-aarch64 \
  -cpu max,sve=on,sve-default-vector-length=16 \
  -L /home/thu/miniforge3/envs/gem5_env/bin/../aarch64-conda-linux-gnu/sysroot \
  /tmp/test_single_layer_SVE_aarch64 q_h0 2 \
  < /tmp/test_single_layer_SVE_aarch64


  /home/thu/opt/qemu-sve/bin/qemu-aarch64 \
  -cpu max,sve=on,sve-default-vector-length=16 \
  -L /home/thu/miniforge3/envs/gem5_env/bin/../aarch64-conda-linux-gnu/sysroot \
  /tmp/test_single_layer_SVE_aarch64 ff0 4 \
  < /tmp/test_single_layer_SVE_aarch64
 */
