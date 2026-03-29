#include "../../transformer_layers/registry_adapter.h"
#include "../../transformer_layers/codebookDense.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

// -------------------------
// Helpers
// -------------------------

static std::vector<uint32_t> packInt8Matrix(const std::vector<int8_t>& src,
                                            std::size_t rows,
                                            std::size_t cols) {
    if (src.size() != rows * cols) {
        throw std::runtime_error("packInt8Matrix: src size mismatch");
    }
    if ((cols % 4) != 0) {
        throw std::runtime_error("packInt8Matrix: cols must be divisible by 4");
    }

    std::vector<uint32_t> packed(rows * (cols / 4), 0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c_word = 0; c_word < cols / 4; ++c_word) {
            uint32_t word = 0;
            for (std::size_t b = 0; b < 4; ++b) {
                std::size_t c = c_word * 4 + b;
                int8_t v = src[r * cols + c];
                word |= (static_cast<uint32_t>(static_cast<uint8_t>(v)) << (8 * b));
            }
            packed[r * (cols / 4) + c_word] = word;
        }
    }

    return packed;
}

static std::vector<int8_t> unpackInt8Matrix(const std::vector<uint32_t>& packed,
                                            std::size_t rows,
                                            std::size_t cols) {
    if ((cols % 4) != 0) {
        throw std::runtime_error("unpackInt8Matrix: cols must be divisible by 4");
    }
    if (packed.size() != rows * (cols / 4)) {
        throw std::runtime_error("unpackInt8Matrix: packed size mismatch");
    }

    std::vector<int8_t> dst(rows * cols, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::size_t word_idx = r * (cols / 4) + (c / 4);
            std::size_t byte_idx = c % 4;
            uint32_t word = packed[word_idx];
            dst[r * cols + c] = static_cast<int8_t>((word >> (8 * byte_idx)) & 0xFF);
        }
    }

    return dst;
}

static void printInt8Matrix(const std::vector<int8_t>& mat,
                            std::size_t rows,
                            std::size_t cols,
                            const std::string& name) {
    std::cout << name << " (" << rows << " x " << cols << "):\n";
    for (std::size_t r = 0; r < rows; ++r) {
        std::cout << "row " << r << ": ";
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << static_cast<int>(mat[r * cols + c]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

static void printPythonMatrix(const std::vector<int8_t>& mat,
                              std::size_t rows,
                              std::size_t cols,
                              const std::string& var_name) {
    std::cout << var_name << " = [\n";
    for (std::size_t r = 0; r < rows; ++r) {
        std::cout << "    [";
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << static_cast<int>(mat[r * cols + c]);
            if (c + 1 != cols) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (r + 1 != rows) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "]\n\n";
}

static std::vector<int8_t> makeDefaultInput(std::size_t rows, std::size_t cols) {
    std::vector<int8_t> input(rows * cols, 0);

    for (std::size_t c = 0; c < cols; ++c) {
        input[0 * cols + c] = static_cast<int8_t>(static_cast<int>(c % 8) - 4);   // -4..3 repeat
    }

    if (rows >= 2) {
        for (std::size_t c = 0; c < cols; ++c) {
            input[1 * cols + c] = static_cast<int8_t>(static_cast<int>(c % 5) - 2); // -2..2 repeat
        }
    }

    for (std::size_t r = 2; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            input[r * cols + c] = static_cast<int8_t>((static_cast<int>((r + c) % 7)) - 3);
        }
    }

    return input;
}

// -------------------------
// Single-layer test
// -------------------------

static void run_single_layer_test(const std::string& layer_name,
                                  std::size_t seq_len) {
    auto cfg = makeCodebookDenseConfigFromRegistry(layer_name.c_str());
    CodebookDense layer(cfg);

    const std::size_t in_dim = cfg.input_size;
    const std::size_t out_dim = cfg.output_size;

    std::vector<int8_t> input_int8 = makeDefaultInput(seq_len, in_dim);
    std::vector<uint32_t> input_packed = packInt8Matrix(input_int8, seq_len, in_dim);
    std::vector<uint32_t> output_packed(seq_len * (out_dim / 4), 0);

    layer.compute(seq_len, input_packed.data(), output_packed.data());

    std::vector<int8_t> output_int8 = unpackInt8Matrix(output_packed, seq_len, out_dim);

    std::cout << "layer_name_test = \"" << layer_name << "\"\n";
    std::cout << "seq_len_test = " << seq_len << "\n";
    std::cout << "input_dim_test = " << in_dim << "\n";
    std::cout << "output_dim_test = " << out_dim << "\n\n";

    printInt8Matrix(input_int8, seq_len, in_dim, "input_int8");
    printInt8Matrix(output_int8, seq_len, out_dim, "output_int8");

    printPythonMatrix(input_int8, seq_len, in_dim, "input_matrix_test");
    printPythonMatrix(output_int8, seq_len, out_dim, "output_matrix_test");
}

int main(int argc, char** argv) {
    try {
        std::string layer_name = "q_h0";
        std::size_t seq_len = 2;

        if (argc >= 2) {
            layer_name = argv[1];
        }
        if (argc >= 3) {
            seq_len = static_cast<std::size_t>(std::stoul(argv[2]));
        }

        run_single_layer_test(layer_name, seq_len);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// cd /home/thu/TiC-SAT
// source ~/miniforge3/etc/profile.d/conda.sh
// conda activate gem5_env

// aarch64-conda-linux-gnu-g++ -x c++ -std=c++17 -O2 -Wall \
//   Full_NN/src/test_single_layer.c \
//   transformer_layers/codebookDense.cc \
//   Full_NN/src/gemm_exec.c \
//   -I. \
//   -IFull_NN/inc \
//   -IFull_NN/gemm_definitions \
//   -o /tmp/test_single_layer_aarch64

// SYSROOT=$(aarch64-conda-linux-gnu-g++ -print-sysroot)
// qemu-aarch64 -L "$SYSROOT" /tmp/test_single_layer_aarch64

// qemu-aarch64 -L "$SYSROOT" /tmp/test_single_layer_aarch64 q_h0 2